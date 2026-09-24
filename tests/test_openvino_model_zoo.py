from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.openvino_model_zoo import OpenVinoModelZooSourceAdapter

_REVISION = "a" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/openvino")


def _tree(*, truncated: bool = False) -> dict[str, Any]:
    return {
        "truncated": truncated,
        "tree": [
            {
                "path": "models/public/bert-base-ner/model.yml",
                "type": "blob",
            },
            {
                "path": "models/public/bert-base-ner/README.md",
                "type": "blob",
            },
            {
                "path": "models/intel/face-detection-0001/model.yml",
                "type": "blob",
            },
        ],
    }


_BERT_MANIFEST = """\
description: >-
  bert-base-ner is a fine-tuned BERT model.
  Paper: https://arxiv.org/abs/1810.04805
task_type: named_entity_recognition
framework: pytorch
license: https://huggingface.co/dslim/bert-base-NER
files:
  - name: bert-base-ner/pytorch_model.bin
    size: 433316646
    checksum: abcdef012345
    original_source: https://huggingface.co/dslim/bert-base-NER/resolve/main/pytorch_model.bin
    source: https://storage.example.test/bert-base-ner/pytorch_model.bin
  - name: bert-base-ner/config.json
    size: 829
    source: https://storage.example.test/bert-base-ner/config.json
"""

_FACE_MANIFEST = """\
description: Face detection model.
task_type: face_detection
framework: dldt
files:
  - name: FP16/face-detection.xml
    size: 1024
    checksum: deadbeef
    source: https://storage.example.test/face-detection.xml
"""


def test_openvino_model_zoo_enumerates_every_pinned_manifest_and_resource() -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_FACE_MANIFEST),
        _response(_BERT_MANIFEST),
    )
    adapter = OpenVinoModelZooSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["model_count"] == 2
    by_name = {record.title: record for record in page.records}
    bert = by_name["bert-base-ner"]
    assert bert.models[0].identifiers == (
        Identifier("openvino:model", "public/bert-base-ner"),
    )
    assert bert.releases[0].revision == _REVISION
    assert bert.releases[0].metadata["task_type"] == "named_entity_recognition"
    assert bert.releases[0].metadata["files"][0]["checksum"] == "abcdef012345"
    assert {link.relation for link in bert.links} >= {
        "model_card",
        "metadata",
        "documentation",
        "weights",
        "original_model_source",
        "license",
    }
    assert all(not link.crawl for link in bert.links if link.relation == "weights")
    assert "https://arxiv.org/abs/1810.04805" in bert.text
    relation = bert.model_relations[0]
    assert relation.predicate == "converted_from"
    assert relation.target.identifiers == (
        Identifier("huggingface:model", "dslim/bert-base-NER"),
    )
    face = by_name["face-detection-0001"]
    assert next(link for link in face.links if "face-detection.xml" in link.url).relation == (
        "model_artifact"
    )
    assert client.calls[1][0].endswith(f"/{_REVISION}?recursive=1")


def test_openvino_model_zoo_skips_manifest_fetches_when_the_commit_is_unchanged() -> None:
    first_client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_FACE_MANIFEST),
        _response(_BERT_MANIFEST),
    )
    adapter = OpenVinoModelZooSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_openvino_model_zoo_rejects_a_truncated_recursive_tree() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_tree(truncated=True)))
    adapter = OpenVinoModelZooSourceAdapter(client=client)

    with pytest.raises(ValueError, match="recursive repository tree is truncated"):
        adapter.fetch_page({})

