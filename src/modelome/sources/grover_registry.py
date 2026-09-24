"""Pinned reader for Tencent AI Lab's first-party GROVER fine-tuned assets."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_HEADING = "## The Reproducibility Issue"
_ROW = re.compile(
    r"^-\s+(?P<dataset>[A-Za-z0-9]+):?\s+"
    r"\[BASE\]\((?P<base>https://ai\.tencent\.com/[^)]+\.tar\.gz)\),\s+"
    r"\[LARGE\]\((?P<large>https://ai\.tencent\.com/[^)]+\.tar\.gz)\)\s*$"
)


class GroverCheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate direct GROVER BASE/LARGE finetune bundles from the official README.

    Handles are the source's explicit checkpoint directory name plus dataset,
    such as ``grover_base_ft_refine/bbbp``. This adapter does not include the
    separately linked OneDrive/Google Drive pretraining pages.
    """

    coverage_limitation = (
        "Covers only BASE and LARGE direct .tar.gz fine-tuning links in the first-party "
        "GROVER README reproducibility list at a pinned commit. It does not infer model "
        "variants, follow cloud-drive pages, or download checkpoint bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("repository", "tencent-ailab/grover")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "tencent-grover:finetuned-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "tencent-grover-finetuned-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "BASE/LARGE direct tar.gz rows under reproducibility heading",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(), next_state=next_state, complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_readme(response.text(), self.name, self.source_path, self.max_entries)
        records = tuple(self._record(row, revision, response.body) for row in checkpoints)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records, next_state=next_state, complete=True,
            upstream_count=len(records), authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        return handle

def _parse_readme(document: str, source: str, path: str, maximum: int) -> tuple[_Checkpoint, ...]:
    rows: list[_Checkpoint] = []
    active = False
    found_heading = False
    handles: set[str] = set()
    for line_number, line in enumerate(document.splitlines(), start=1):
        if line.startswith("## "):
            active = line.strip() == _HEADING
            found_heading |= active
            continue
        if not active or not line.startswith("-"):
            continue
        match = _ROW.fullmatch(line.strip())
        if match is None:
            # The configured section contains prose and a single link list; any
            # dataset-like bullet with malformed structure should fail closed.
            if re.match(r"^-\s+[A-Za-z0-9]+:", line):
                raise ValueError(f"{source}: malformed checkpoint row at line {line_number}")
            continue
        dataset = match.group("dataset").casefold()
        for size, url in (("base", match.group("base")), ("large", match.group("large"))):
            handle = f"grover_{size}_ft_refine/{dataset}"
            parsed = urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname != "ai.tencent.com":
                raise ValueError(f"{source}: {handle!r} has an untrusted artifact host")
            expected = (
                "/ailab/ml/ml-data/grover-models/finetune/"
                f"grover_{size}_ft_refine/{dataset}.tar.gz"
            )
            if parsed.path != expected or parsed.query or parsed.fragment:
                raise ValueError(f"{source}: {handle!r} URL does not match its declared identity")
            if handle in handles:
                raise ValueError(f"{source}: duplicate checkpoint handle {handle!r}")
            handles.add(handle)
            rows.append(_Checkpoint(handle, url, f"{path}:line:{line_number}"))
            if len(rows) > maximum:
                raise ValueError(f"{source}: registry exceeds {maximum} entries")
    if not found_heading:
        raise ValueError(f"{source}: reproducibility checkpoint section is missing")
    if not rows:
        raise ValueError(f"{source}: checkpoint list contains no matching entries")
    return tuple(rows)
