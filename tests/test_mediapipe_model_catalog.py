from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.mediapipe_model_catalog import (
    MediaPipeModelCatalogSourceAdapter,
    _parse_entries,
)

_SHA = "a" * 40
_DOC = """# MediaPipe Models
### [Face Detection](https://google-ai-edge.github.io/mediapipe/solutions/face_detection)
* Short-range: [TFLite model](https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite)
* Quantized: [TFLite model](https://github.com/google-ai-edge/mediapipe/tree/master/example/face_quant.tflite)
### [Objectron](https://google-ai-edge.github.io/mediapipe/solutions/objectron)
* [TFLite model for shoes](https://storage.googleapis.com/mediapipe-assets/object_detection_3d_sneakers.tflite)
* [Model card](https://example.com/card)
"""


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if url.endswith("/commits/master"):
            body = json.dumps({"sha": _SHA}).encode()
        else:
            body = _DOC.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_keeps_only_exact_task_or_tflite_asset_links() -> None:
    parsed = _parse_entries(_DOC, maximum=10, source="test")

    assert [entry[2] for entry in parsed] == [
        "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        "https://github.com/google-ai-edge/mediapipe/tree/master/example/face_quant.tflite",
        "https://storage.googleapis.com/mediapipe-assets/object_detection_3d_sneakers.tflite",
    ]
    assert parsed[0][0] == "Face Detection"
    assert parsed[2][0] == "Objectron"


def test_parser_deduplicates_tracking_query_variants_of_one_artifact_url() -> None:
    document = """### Face Detection
* [Short-range](https://storage.googleapis.com/mediapipe-assets/model.tflite?utm_source=docs)
* [Short-range](https://storage.googleapis.com/mediapipe-assets/model.tflite)
"""

    parsed = _parse_entries(document, maximum=10, source="test")

    assert len(parsed) == 1
    assert parsed[0][2] == "https://storage.googleapis.com/mediapipe-assets/model.tflite"


def test_parser_rejects_conflicting_labels_after_url_canonicalization() -> None:
    document = """### Face Detection
* [Short-range](https://storage.googleapis.com/mediapipe-assets/model.tflite?utm_source=docs)
* [Full-range](https://storage.googleapis.com/mediapipe-assets/model.tflite)
"""

    with pytest.raises(ValueError, match="duplicate artifact URL"):
        _parse_entries(document, maximum=10, source="test")


def test_same_filename_in_distinct_asset_paths_keeps_distinct_source_ids() -> None:
    document = """### Family A
* [Model](https://storage.googleapis.com/mediapipe-assets/family-a/model.tflite)
### Family B
* [Model](https://storage.googleapis.com/mediapipe-assets/family-b/model.tflite)
"""
    parsed = _parse_entries(document, maximum=10, source="test")
    adapter = MediaPipeModelCatalogSourceAdapter(client=object())
    records = tuple(
        adapter._record(entry, _SHA, b"index", "https://example.test/catalog")
        for entry in parsed
    )

    assert len(records) == 2
    assert len({record.source_record_id for record in records}) == 2
    assert len({record.models[0].local_id for record in records}) == 2
    assert len({record.releases[0].local_id for record in records}) == 2
    assert len({record.models[0].identifiers for record in records}) == 2
    result = build_entries(
        source_record_to_entry_seed(record, source="mediapipe") for record in records
    )
    assert len(result.entries) == 2


def test_adapter_emits_exact_binary_links_with_revisioned_first_party_source() -> None:
    client = _Client()
    adapter = MediaPipeModelCatalogSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/google-ai-edge/mediapipe/commits/master",
        f"https://raw.githubusercontent.com/google-ai-edge/mediapipe/{_SHA}/docs/solutions/models.md",
    ]
    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 3
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (
        Identifier(
            "mediapipe:artifact",
            "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite",
        ),
    )
    assert record.releases[0].metadata["revision"] == _SHA
    assert record.releases[0].metadata["family"] == "Face Detection"
    artifact_links = [link for link in record.links if link.relation == "model_artifact"]
    assert len(artifact_links) == 1
    assert artifact_links[0].url == record.releases[0].metadata["artifact_url"]


@pytest.mark.parametrize(
    "document,maximum",
    [
        ("### Family\n[weights](http://example.com/model.tflite)", 5),
        ("### Family\n[weights](https://example.com/model.tflite)", 0),
        (
            "### Family\n[weights](https://example.com/model.tflite)\n[weights](https://example.com/model.task)",
            1,
        ),
    ],
)
def test_parser_rejects_unsafe_or_overlarge_asset_lists(document: str, maximum: int) -> None:
    with pytest.raises(ValueError):
        _parse_entries(document, maximum=maximum, source="test")
