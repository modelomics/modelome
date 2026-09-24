"""First-party Uni-MOF checkpoint rows from its README."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class UniMofCheckpointSourceAdapter:
    """Enumerate exact ``.pt`` download rows in dptech-corp/Uni-MOF README.

    This reads the first-party README at a pinned Git revision, admits only
    literal GitHub release assets ending in ``.pt``, and never fetches weights.
    Other model families and non-checkpoint files in the same tables are ignored.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only .pt checkpoint URLs explicitly listed in the official "
        "dptech-corp/Uni-MOF README. It does not cover other Uni-Mol releases, "
        "third-party fine-tunes, or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "unimof-checkpoints",
        repository: str = "dptech-corp/Uni-MOF",
        branch: str = "main",
        max_response_bytes: int = 2 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("name and owner/repository are required")
        if not branch.strip() or max_response_bytes <= 0:
            raise ValueError("branch and positive max_response_bytes are required")
        self.name, self.repository, self.branch = name, repository, branch
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "unimof-readme-checkpoints-v1",
            "repository": repository,
            "branch": branch,
            "path": "README.md",
            "max_response_bytes": max_response_bytes,
        })

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/README.md"
        )
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        entries = _checkpoint_rows(response.text(), self.repository)
        if not entries:
            raise ValueError(f"{self.name}: no first-party .pt checkpoint rows found")
        records = tuple(self._record(name, url, revision, source_url) for name, url in entries)
        return SourcePage(
            records,
            {"completed_revision": revision, "checked_at": checked,
             "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, name: str, weight_url: str, revision: str,
                source_url: str) -> SourceRecord:
        handle = name.removesuffix(".pt")
        model_id = f"model:{handle}"
        namespace = "unimof:checkpoint"
        page_url = f"{self.repository_url}/blob/{revision}/README.md"
        model = ModelHint(
            model_id, handle, identifiers=(Identifier(namespace, handle),),
            aliases=(name,), status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}", model_id, revision=revision,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={"repository": self.repository, "revision": revision,
                      "checkpoint_filename": name, "weight_url": weight_url},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}", kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(page_url), title=f"Uni-MOF {handle}",
            raw={"repository": self.repository, "revision": revision,
                 "readme_url": source_url, "checkpoint_filename": name,
                 "weight_url": weight_url},
            text=f"Uni-MOF checkpoint listed in its README: {name}",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(page_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(self.repository_url, "source_implementation", crawl=False,
                     model_local_ids=(model_id,)),
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,), releases=(release,),
        )


class ChempropCheMeleonCheckpointSourceAdapter:
    """Index the CheMeleon file explicitly downloaded by Chemprop's docs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the CheMeleon message-passing checkpoint linked in Chemprop's "
        "first-party foundation-model tutorial. It does not enumerate user-trained "
        "Chemprop checkpoints or download model bytes."
    )
    documentation_url = (
        "https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html"
    )
    weight_url = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    record_url = "https://zenodo.org/records/15460715"

    def __init__(
        self,
        *,
        name: str = "chemprop-chemeleon-checkpoint",
        max_response_bytes: int = 8 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive max_response_bytes are required")
        self.name = name
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "chemprop-chemeleon-doc-checkpoint-v1",
            "documentation_url": self.documentation_url,
            "weight_url": self.weight_url,
            "max_response_bytes": max_response_bytes,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.documentation_url, headers={"Accept": "text/html,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: documentation returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: documentation exceeds response limit")
        if not _has_chemeleon_url(response.text()):
            raise ValueError(f"{self.name}: official documentation no longer lists the checkpoint")
        digest = content_hash(response.body)
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("documentation_sha256"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=1)

        handle = "chemeleon_mp"
        model_id = f"model:{handle}"
        namespace = "chemprop:checkpoint"
        model = ModelHint(
            model_id, "CheMeleon", identifiers=(Identifier(namespace, handle),),
            aliases=("chemeleon_mp.pt",), status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}", model_id,
            identifiers=(Identifier(f"{namespace}:release", "15460715"),),
            metadata={"zenodo_record": "15460715", "checkpoint_filename": "chemeleon_mp.pt",
                      "weight_url": self.weight_url,
                      "documentation_sha256": digest},
        )
        record = SourceRecord(
            source_record_id=f"checkpoint:{handle}", kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(self.record_url), title="Chemprop CheMeleon",
            raw={"documentation_url": self.documentation_url,
                 "documentation_sha256": digest, "zenodo_record": "15460715",
                 "checkpoint_filename": "chemeleon_mp.pt", "weight_url": self.weight_url},
            text="Chemprop's first-party tutorial explicitly downloads the CheMeleon checkpoint.",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(self.record_url, "model_card", crawl=False,
                     model_local_ids=(model_id,)),
                Link(self.documentation_url, "documentation", crawl=False,
                     model_local_ids=(model_id,)),
                Link("https://github.com/chemprop/chemprop", "source_implementation",
                     crawl=False, model_local_ids=(model_id,)),
                Link(self.weight_url, "weights", crawl=False,
                     model_local_ids=(model_id,)),
            ),
            models=(model,), releases=(release,),
        )
        return SourcePage(
            (record,),
            {"documentation_sha256": digest, "checked_at": checked, "model_count": 1},
            True, upstream_count=1, authoritative_snapshot=True,
        )


def _has_chemeleon_url(documentation: str) -> bool:
    """Require the one exact file URL and do not infer other Zenodo assets."""
    return documentation.count(ChempropCheMeleonCheckpointSourceAdapter.weight_url) > 0


def _checkpoint_rows(markdown: str, repository: str) -> tuple[tuple[str, str], ...]:
    """Read markdown table rows with one exact release asset in the last cell."""
    expected_prefix = f"https://github.com/{repository}/releases/download/"
    found: dict[str, str] = {}
    for line in markdown.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        urls = re.findall(r"https://[^\s|)]+", line)
        if len(urls) != 1:
            continue
        url = urls[0].rstrip(".,;")
        parts = urlsplit(url)
        filename = parts.path.rsplit("/", 1)[-1]
        if (
            not url.startswith(expected_prefix)
            or parts.hostname != "github.com"
            or not filename.endswith(".pt")
            or len(cells) < 2
        ):
            continue
        label = cells[0].strip("` ")
        if not label or label.casefold() in {"model", "weight", "file"}:
            label = filename
        name = filename
        prior = found.get(name)
        if prior is not None and prior != url:
            raise ValueError(f"conflicting checkpoint URL for {name}")
        found[name] = url
    return tuple(sorted(found.items()))


__all__ = [
    "ChempropCheMeleonCheckpointSourceAdapter",
    "UniMofCheckpointSourceAdapter",
]
