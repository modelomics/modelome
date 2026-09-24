"""Read exact WeatherNext checkpoint patterns from the first-party model README."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

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

_BUCKET_PREFIX = "https://storage.googleapis.com/dm_graphcast/weathernext2/params/"
_DECLARATION = re.compile(
    r"`(?P<pattern>WeatherNext(?:2|Cyclones(?:_Mini)?)_<20(?:23|24|25)"
    r"(?:_model\{1,2,3,4\})?\.npz)`"
)
_EXPECTED_PATTERNS = {
    "WeatherNext2_<2025_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2025_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2024_model{1,2,3,4}.npz",
    "WeatherNextCyclones_<2023_model{1,2,3,4}.npz",
    "WeatherNextCyclones_Mini_<2024.npz",
    "WeatherNextCyclones_Mini_<2023.npz",
}


class WeatherNext2CheckpointRegistrySourceAdapter(
    StaticJsonCheckpointRegistrySourceAdapter
):
    """Expand only the finite artifact patterns declared in Google DeepMind's README."""

    coverage_limitation = (
        "Covers the 18 checkpoint object names declared by the first-party "
        "google-deepmind/weathernext README: four model seeds for WeatherNext2 2025, "
        "WeatherNextCyclones 2023–2025, and one checkpoint each for Cyclones Mini "
        "2023–2024. Object paths use the bucket and params prefix shown by the "
        "first-party demo, which reads known objects anonymously. The adapter does "
        "not enumerate/list the bucket, check object existence, or download weights."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "google-deepmind-weathernext2-checkpoint-registry")
        kwargs.setdefault("repository", "google-deepmind/weathernext")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "weathernext2:checkpoint")
        kwargs.setdefault("max_entries", 18)
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
        family, year, seed = handle.split("/")
        family_name = {
            "weathernext2": "WeatherNext2",
            "cyclones": "WeatherNext Cyclones",
            "cyclones-mini": "WeatherNext Cyclones Mini",
        }[family]
        if seed.startswith("model-"):
            return f"{family_name} {year} {seed.replace('-', ' ')}"
        return f"{family_name} {year}"


def _parse_readme(document: str, source: str, path: str) -> tuple[_Checkpoint, ...]:
    patterns = [match.group("pattern") for match in _DECLARATION.finditer(document)]
    if len(patterns) != len(_EXPECTED_PATTERNS) or set(patterns) != _EXPECTED_PATTERNS:
        raise ValueError(f"{source}: README must declare the six expected weight patterns")

    checkpoints: list[_Checkpoint] = []
    for pattern in patterns:
        multi_seed = "_model{1,2,3,4}.npz" in pattern
        prefix, suffix = pattern.split("{1,2,3,4}", 1) if multi_seed else (pattern[:-4], ".npz")
        for seed in (1, 2, 3, 4) if multi_seed else (None,):
            filename = f"{prefix}{seed}{suffix}" if seed is not None else f"{prefix}{suffix}"
            family, year, model_seed = _handle_parts(filename)
            handle = f"{family}/{year}/{model_seed}"
            url = _BUCKET_PREFIX + quote(filename, safe="._-")
            checkpoints.append(
                _Checkpoint(handle=handle, url=url, locator=f"{path}:weight-pattern:{pattern}")
            )
    if len(checkpoints) != 18 or len({item.handle for item in checkpoints}) != 18:
        raise ValueError(f"{source}: expected 18 unique WeatherNext checkpoint paths")
    return tuple(checkpoints)


def _handle_parts(filename: str) -> tuple[str, str, str]:
    if filename.startswith("WeatherNext2_<"):
        family = "weathernext2"
        remainder = filename.removeprefix("WeatherNext2_<")
    elif filename.startswith("WeatherNextCyclones_Mini_<"):
        family = "cyclones-mini"
        remainder = filename.removeprefix("WeatherNextCyclones_Mini_<")
    elif filename.startswith("WeatherNextCyclones_<"):
        family = "cyclones"
        remainder = filename.removeprefix("WeatherNextCyclones_<")
    else:
        raise ValueError(f"unexpected WeatherNext model filename {filename!r}")
    year, suffix = remainder.split("_", 1) if "_" in remainder else (remainder[:-4], "")
    if suffix:
        seed = suffix.removeprefix("model").removesuffix(".npz")
        model_seed = f"model-{seed}"
    else:
        model_seed = "single"
    if year not in {"2023", "2024", "2025"} or model_seed not in {
        "model-1",
        "model-2",
        "model-3",
        "model-4",
        "single",
    }:
        raise ValueError(f"unexpected WeatherNext model filename {filename!r}")
    return family, year, model_seed
