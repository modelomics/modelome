"""Read Huawei's Pangu-Weather checkpoint links from its official repository."""

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

_SECTION = "#### Downloading trained models"
_ROW = re.compile(
    r"^The (?P<hours>[0-9]+)-hour model \(pangu_weather_(?P<filename_hours>[0-9]+)\.onnx\): "
    r"\[Google drive\]\((?P<url>https://drive\.google\.com/file/d/"
    r"(?P<file_id>[A-Za-z0-9_-]+)/view\?usp=share_link)\)"
    r"(?:/\[Baidu netdisk\]\(https://pan\.baidu\.com/[^)]+\))?"
)
_HORIZONS = {"1", "3", "6", "24"}


class PanguWeatherCheckpointRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate Pangu-Weather's four exact ONNX checkpoints and share pages."""

    coverage_limitation = (
        "Covers only the four Pangu-Weather model links in the official repository's "
        "trained-model section. The links are Google Drive share pages, not direct file "
        "URLs; this adapter does not resolve them, follow redirects, or download files."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "huawei-pangu-weather-checkpoint-registry")
        kwargs.setdefault("repository", "198808xc/Pangu-Weather")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "pangu-weather:checkpoint")
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
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        rows = _parse_readme(response.text(), self.name, self.source_path)
        records = tuple(self._record(row, revision, response.body) for row in rows)
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
        hours = handle.removeprefix("pangu_weather_").removesuffix(".onnx")
        return f"Pangu-Weather {hours}-hour forecast"

    def _record(self, checkpoint: _Checkpoint, revision: str, source: bytes):
        record = super()._record(checkpoint, revision, source)
        # A Google Drive /view page is a source-declared location, not a direct
        # checkpoint file URL. Keep it as an artifact page without resolving it.
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
    section = document.split(_SECTION, 1)[1].split("####", 1)[0]
    rows: list[_Checkpoint] = []
    horizons: set[str] = set()
    for line_number, line in enumerate(section.splitlines(), 1):
        match = _ROW.fullmatch(line.strip())
        if match is None:
            if "pangu_weather_" in line:
                raise ValueError(f"{source}: invalid model link in trained-model section")
            continue
        hours = match.group("hours")
        filename_hours = match.group("filename_hours")
        if hours != filename_hours or hours not in _HORIZONS:
            raise ValueError(f"{source}: unsupported checkpoint horizon {hours!r}")
        if hours in horizons:
            raise ValueError(f"{source}: duplicate {hours}-hour checkpoint")
        horizons.add(hours)
        rows.append(
            _Checkpoint(
                handle=f"pangu_weather_{hours}.onnx",
                url=match.group("url"),
                locator=f"{path}:trained-models:{line_number}",
            )
        )
    if horizons != _HORIZONS:
        raise ValueError(f"{source}: expected Pangu-Weather 1, 3, 6, and 24-hour checkpoints")
    return tuple(rows)
