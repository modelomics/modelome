"""Exact public Diffusion Policy checkpoint inventory (non-Hub release)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_SITE = "https://diffusion-policy.cs.columbia.edu"
_ROOTS = ("image", "low_dim")
_BEST_CHECKPOINT = re.compile(
    r"^epoch=(?P<epoch>\d+)-test_mean_score=(?P<score>-?\d+(?:\.\d+)?)\.ckpt$"
)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.=-]{1,160}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _IndexLinks(HTMLParser):
    """Read only link targets from the upstream web-server directory index."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


class DiffusionPolicyCheckpointIndexAdapter:
    """Enumerate named best-run checkpoints from Diffusion Policy's official site.

    The official README publishes recursively indexed experiment directories at
    `/data/experiments/{image,low_dim}/`. This adapter follows only the documented
    task/method/train_N/checkpoints path shape and admits scored `epoch=...ckpt`
    files. It ignores mutable `latest.ckpt` aliases and never downloads weights.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers scored epoch checkpoints in the official Diffusion Policy image and "
        "low_dim experiment directory indexes. It excludes latest aliases, other "
        "projects, undocumented folder layouts, and checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "diffusion-policy-checkpoints",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_directories: int = 1024,
        max_entries: int = 20_000,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_directories", max_directories),
            ("max_entries", max_entries),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_directories = max_directories
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "diffusion-policy-checkpoint-index-v1",
                "site": _SITE,
                "roots": list(_ROOTS),
                "best_checkpoint_pattern": _BEST_CHECKPOINT.pattern,
                "max_response_bytes": max_response_bytes,
                "max_directories": max_directories,
                "max_entries": max_entries,
            }
        )

    @property
    def repository_url(self) -> str:
        return "https://github.com/real-stanford/diffusion_policy"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state  # The site directory indexes are small enough for one bounded snapshot.
        pending = [f"{_SITE}/data/experiments/{root}/" for root in _ROOTS]
        visited: set[str] = set()
        checkpoints: dict[str, tuple[str, str]] = {}
        index_digests: list[str] = []
        while pending:
            index_url = pending.pop(0)
            if index_url in visited:
                continue
            if len(visited) >= self.max_directories:
                raise ValueError(f"{self.name}: directory index exceeds {self.max_directories}")
            visited.add(index_url)
            response: HttpResponse = self.client.get(index_url, headers={"Accept": "text/html"})
            if response.status != 200:
                raise ValueError(f"{self.name}: directory index returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: directory index exceeds response limit")
            index_digests.append(content_hash(response.body))
            parser = _IndexLinks()
            parser.feed(response.text())
            parser.close()
            for href in parser.hrefs:
                resolved = _within_site(index_url, href, self.name)
                if resolved is None:
                    continue
                relative = _relative_to_experiment_root(resolved)
                if relative is None:
                    continue
                components = relative.rstrip("/").split("/") if relative else []
                if resolved.endswith("/"):
                    if (
                        _is_expected_directory(components)
                        and resolved not in visited
                        and resolved not in pending
                    ):
                        pending.append(resolved)
                    continue
                match = _BEST_CHECKPOINT.fullmatch(components[-1]) if components else None
                if match and len(components) == 6 and components[4] == "checkpoints":
                    handle = "/".join(components)
                    checkpoints[handle] = (resolved, match.group("score"))
                    if len(checkpoints) > self.max_entries:
                        raise ValueError(
                            f"{self.name}: checkpoint inventory exceeds {self.max_entries}"
                        )
        if not checkpoints:
            raise ValueError(f"{self.name}: no scored checkpoints found")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        ordered = sorted(checkpoints.items())
        records = tuple(self._record(handle, url, score) for handle, (url, score) in ordered)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "index_count": len(visited),
                "index_sha256": content_hash(index_digests),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, handle: str, weight_url: str, score: str) -> SourceRecord:
        identifier = Identifier("diffusion-policy:checkpoint", handle)
        local_id = f"checkpoint:{content_hash(handle)[:24]}"
        filename = handle.rsplit("/", 1)[-1]
        root, task, method, run, _, _ = handle.split("/")
        model = ModelHint(
            local_id=local_id,
            name=f"{task} {method} {run} {filename}",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=handle,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(handle)[:24]}",
            model_local_id=local_id,
            identifiers=(Identifier("diffusion-policy:checkpoint-release", handle),),
            metadata={
                "experiment_root": root,
                "task": task,
                "method": method,
                "training_run": run,
                "checkpoint_file": filename,
                "test_mean_score": score,
                "weight_url": weight_url,
            },
            locator=handle,
        )
        page_url = f"{_SITE}/data/experiments/{root}/{task}/{method}/{run}/checkpoints/"
        return SourceRecord(
            source_record_id=f"checkpoint:{content_hash(handle)[:24]}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(weight_url),
            title=f"Diffusion Policy {task} {method} {filename}",
            raw={"checkpoint_path": handle, "weight_url": weight_url},
            text=f"Official Diffusion Policy checkpoint {handle}; test mean score {score}.",
            identifiers=(identifier,),
            links=(
                Link(page_url, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(self.repository_url, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(weight_url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _relative_to_experiment_root(url: str) -> str | None:
    path = unquote(urlsplit(url).path)
    prefix = "/data/experiments/"
    if not path.startswith(prefix):
        return None
    relative = path[len(prefix):]
    if not any(relative.startswith(f"{root}/") for root in _ROOTS):
        return None
    return relative


def _within_site(index_url: str, href: str, source: str) -> str | None:
    parts = urlsplit(href)
    if parts.scheme or parts.netloc or parts.query or parts.fragment:
        return None
    resolved = urljoin(index_url, href)
    target = urlsplit(resolved)
    if target.scheme != "https" or target.netloc != urlsplit(_SITE).netloc:
        return None
    path = unquote(target.path)
    if any(component in {".", ".."} for component in path.split("/")):
        raise ValueError(f"{source}: directory index contains a traversal link")
    if not path.startswith("/data/experiments/"):
        return None
    return resolved


def _is_expected_directory(components: list[str]) -> bool:
    if not components or not all(_SAFE_COMPONENT.fullmatch(part) for part in components):
        return False
    if len(components) in {1, 2, 3}:
        return True
    if len(components) == 4:
        return bool(re.fullmatch(r"train_\d+", components[3]))
    return (
        len(components) == 5
        and bool(re.fullmatch(r"train_\d+", components[3]))
        and components[4] == "checkpoints"
    )


__all__ = ["DiffusionPolicyCheckpointIndexAdapter"]
