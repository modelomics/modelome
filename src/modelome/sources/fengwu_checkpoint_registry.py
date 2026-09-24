"""Read the OpenEarthLab FengWu checkpoint declarations from its first-party README."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

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

_SECTION = "## Downloading trained models"
_SHARE = r"https://pjlab-my\.sharepoint\.cn/:u:/g/personal/[A-Za-z0-9_]+/[A-Za-z0-9_-]+"
_ROW = re.compile(
    rf"^Fengwu (?P<variant>without transfer learning|with transfer learning) "
    rf"\((?P<filename>fengwu_v[12]\.onnx)"
    r"(?:, finetune the model with analysis data up to 2021)?\): "
    rf"\[Onedrive\((?P<url>{_SHARE})\)\]$"
)
_EXPECTED = {
    "Fengwu without transfer learning": "fengwu_v1.onnx",
    "Fengwu with transfer learning": "fengwu_v2.onnx",
}


class FengWuCheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate the two exact ONNX model share pages declared by OpenEarthLab."""

    coverage_limitation = (
        "Covers only FengWu v1 and v2 share-page URLs listed under the first-party "
        "OpenEarthLab/FengWu README's trained-model section. These are OneDrive share "
        "pages, not direct file URLs; the adapter does not resolve them or download files."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "openearthlab-fengwu-checkpoint-registry")
        kwargs.setdefault("repository", "OpenEarthLab/FengWu")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "fengwu:checkpoint")
        super().__init__(**kwargs)

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_readme(response.text(), self.name, self.source_path)
        records = tuple(self._record(item, revision, response.body) for item in checkpoints)
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
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _model_name(self, handle: str) -> str:
        return f"FengWu {handle.removeprefix('fengwu_').removesuffix('.onnx')}"

    def _record(self, checkpoint: _Checkpoint, revision: str, source: bytes):
        record = super()._record(checkpoint, revision, source)
        links = tuple(
            replace(link, relation="checkpoint_download_page")
            if link.relation == "weights"
            else link
            for link in record.links
        )
        raw = dict(record.raw)
        raw["checkpoint_share_url"] = raw.pop("weight_url")
        releases = tuple(
            replace(
                release,
                metadata={
                    key: value for key, value in release.metadata.items() if key != "weight_url"
                }
                | {"checkpoint_share_url": checkpoint.url},
            )
            for release in record.releases
        )
        return replace(record, links=links, raw=raw, releases=releases)


def _parse_readme(document: str, source: str, path: str) -> tuple[_Checkpoint, ...]:
    if document.count(_SECTION) != 1:
        raise ValueError(f"{source}: expected exactly one trained-model section")
    section = document.split(_SECTION, 1)[1].split("## ", 1)[0]
    checkpoints: list[_Checkpoint] = []
    seen: set[str] = set()
    for line_number, line in enumerate(section.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        match = _ROW.fullmatch(stripped)
        if match is None:
            if "fengwu_v" in stripped.casefold():
                raise ValueError(f"{source}: malformed checkpoint row at line {line_number}")
            continue
        variant, filename, url = match.group("variant", "filename", "url")
        expected = _EXPECTED[f"Fengwu {variant}"]
        if filename != expected or filename in seen:
            raise ValueError(f"{source}: unexpected or duplicate checkpoint {filename!r}")
        seen.add(filename)
        checkpoints.append(
            _Checkpoint(handle=filename, url=url, locator=f"{path}:trained-models:{line_number}")
        )
    if seen != set(_EXPECTED.values()):
        raise ValueError(f"{source}: expected the FengWu v1 and v2 checkpoint rows")
    return tuple(checkpoints)
