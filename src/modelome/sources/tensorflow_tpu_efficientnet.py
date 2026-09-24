"""TensorFlow TPU's first-party EfficientNet checkpoint matrix."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_REPOSITORY = "tensorflow/tpu"
_PATH = "models/official/efficientnet/README.md"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CHECKPOINT_LINK = re.compile(r"\[(?P<label>ckpt)\]\((?P<url>https?://[^)]+)\)", re.I)
_SECTION = "## 2. Using Pretrained EfficientNet Checkpoints"
_NEXT_HEADING = re.compile(r"^#{1,2}\s+")
_VARIANT = re.compile(r"^(?:B[0-8]|L2(?:-\d+)?)$")
_CHECKPOINT_VARIANTS = {"ckpts", "ckptsaug", "randaug", "advprop", "noisystudent"}


class TensorFlowTPUEfficientNetSourceAdapter:
    """Parse the published transposed table; do not fetch any checkpoints."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the source-declared checkpoint links in TensorFlow TPU's official "
        "EfficientNet README matrix. It excludes child README inventories, arbitrary "
        "links, evaluation data, and model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "tensorflow-tpu-efficientnet-checkpoints",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 500,
        client: Any | None = None,
    ) -> None:
        if max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("response and entry limits must be positive")
        self.name = name
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "tensorflow-tpu-efficientnet-matrix-v1",
            "repository": _REPOSITORY,
            "path": _PATH,
            "section": _SECTION,
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/master"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid commit revision")
        checked = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage((), {**state, "checked_at": checked}, True,
                              upstream_count=state.get("model_count"))

        source_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/"
            f"{quote(_PATH, safe='/')}"
        )
        document = self.client.get(source_url, headers={"Accept": "text/plain"})
        if document.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {document.status}")
        if len(document.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        entries = _parse_checkpoint_matrix(document.text(), self.max_entries)
        if not entries:
            raise ValueError(f"{self.name}: no checkpoint rows found in expected section")
        records = tuple(
            self._record(method, variant, url, revision, source_url)
            for method, variant, url in entries
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked,
                "source_sha256": content_hash(document.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, method: str, variant: str, checkpoint: str, revision: str, source_url: str
    ) -> SourceRecord:
        identity = f"{_PATH}:{method}:{variant}"
        release_identity = f"{identity}:{checkpoint}"
        model_id = f"model:{content_hash(identity)[:24]}"
        locator = f"{method} / EfficientNet-{variant}"
        model = ModelHint(
            local_id=model_id,
            name=f"EfficientNet-{variant}",
            aliases=(locator,),
            identifiers=(Identifier("tensorflow-tpu:efficientnet", identity),),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(release_identity)[:24]}",
            model_local_id=model_id,
            revision=revision,
            identifiers=(
                Identifier("tensorflow-tpu:efficientnet-release", release_identity),
            ),
            metadata={
                "training_method": method,
                "variant": variant,
                "checkpoint": checkpoint,
                "document": _PATH,
            },
            locator=locator,
        )
        github_url = f"https://github.com/{_REPOSITORY}/blob/{revision}/{_PATH}"
        return SourceRecord(
            source_record_id=f"tensorflow-tpu-efficientnet:{content_hash(release_identity)[:24]}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(github_url),
            title=locator,
            raw={
                "repository": _REPOSITORY,
                "revision": revision,
                "document": _PATH,
                "training_method": method,
                "variant": variant,
                "checkpoint": checkpoint,
            },
            text=(
                f"TensorFlow TPU publishes the {method} EfficientNet-{variant} "
                f"checkpoint at {checkpoint}. Source: {source_url}"
            ),
            identifiers=model.identifiers,
            links=(
                Link(github_url, relation="model_card", crawl=False),
                Link(
                    checkpoint,
                    relation="weights",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(
                    f"https://github.com/{_REPOSITORY}",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoint_matrix(
    markdown: str, max_entries: int = 500
) -> tuple[tuple[str, str, str], ...]:
    """Map each linked cell to its declared training-method row and model column."""
    lines = markdown.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == _SECTION)
    except StopIteration:
        return ()
    end = next(
        (i for i in range(start + 1, len(lines)) if _NEXT_HEADING.match(lines[i])),
        len(lines),
    )
    headers: list[str] = []
    entries = []
    for line in lines[start + 1 : end]:
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            continue
        if any(_VARIANT.fullmatch(cell) for cell in cells[1:]):
            headers = cells
            continue
        if not headers or not cells[0]:
            continue
        method = re.sub(r"\s+", " ", cells[0]).strip()
        for index, cell in enumerate(cells[1:], start=1):
            if index >= len(headers) or not _VARIANT.fullmatch(headers[index]):
                continue
            variant = headers[index]
            for match in _CHECKPOINT_LINK.finditer(cell):
                url = match.group("url").strip()
                parts = urlsplit(url)
                if (
                    parts.scheme != "https"
                    or parts.hostname != "storage.googleapis.com"
                    or len(parts.path.split("/")) < 5
                    or parts.path.split("/")[1:3]
                    != ["cloud-tpu-checkpoints", "efficientnet"]
                    or parts.path.split("/")[3] not in _CHECKPOINT_VARIANTS
                    or parts.username
                    or parts.password
                ):
                    continue
                entries.append((method, variant, url))
                if len(entries) > max_entries:
                    raise ValueError(
                        "tensorflow-tpu-efficientnet-checkpoints: entry limit exceeded"
                    )
    return tuple(dict.fromkeys(entries))
