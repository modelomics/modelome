from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openml_run_checkpoints import OpenMLRunOnnxCheckpointSourceAdapter


class _Client:
    def __init__(self, *payloads: Any) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps(payload).encode(), "https://www.openml.org")
            for payload in payloads
        ]
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return self.responses.pop(0)


def test_openml_run_onnx_source_emits_weights_only_for_declared_exact_file_id() -> None:
    client = _Client({
        "run": {
            "run_id": 10594206,
            "flow_id": 92,
            "flow_name": "keras.SceneClassifier",
            "output_files": {"predictions": 12, "onnx_model": 778},
        }
    })
    adapter = OpenMLRunOnnxCheckpointSourceAdapter(run_ids=[10594206], client=client)

    page = adapter.fetch_page({})

    assert client.calls == ["https://www.openml.org/api/v1/json/run/10594206"]
    assert page.complete is True
    assert page.next_state == {"index": 1}
    assert len(page.records) == 1
    record = page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.source_record_id == "openml-run-onnx:10594206:778"
    assert record.canonical_url == "https://api.openml.org/data/download/778/model.onnx"
    assert record.models[0].name == "keras.SceneClassifier"
    assert record.models[0].identifiers == (
        Identifier("openml:trained-run-model", "10594206"),
    )
    assert record.releases[0].identifiers == (Identifier("openml:file", "778"),)
    assert record.releases[0].metadata == {
        "format": "onnx",
        "run_id": "10594206",
        "flow_id": "92",
    }
    assert all(not link.crawl for link in record.links)


def test_openml_run_onnx_source_skips_runs_without_declared_model_file() -> None:
    client = _Client({
        "run": {"run_id": 23, "flow_id": 7, "output_files": {"predictions": 33}}
    })
    adapter = OpenMLRunOnnxCheckpointSourceAdapter(run_ids=[23], client=client)

    page = adapter.fetch_page({})

    assert page.records == ()
    assert page.complete is True


def test_openml_run_onnx_source_accepts_v1_output_file_rows() -> None:
    client = _Client({
        "run": {
            "run_id": "25",
            "outputfile": [
                {"file_name": "predictions.arff", "file_id": "70"},
                {"file_name": "model.onnx", "file_id": "71"},
            ],
        }
    })
    adapter = OpenMLRunOnnxCheckpointSourceAdapter(run_ids=[25], client=client)

    page = adapter.fetch_page({})

    assert page.records[0].canonical_url.endswith("/71/model.onnx")


@pytest.mark.parametrize("payload", [[], {"run": None}, {"run": {"id": 1, "output_files": "bad"}}])
def test_openml_run_onnx_source_rejects_malformed_detail_response(payload: Any) -> None:
    adapter = OpenMLRunOnnxCheckpointSourceAdapter(run_ids=[1], client=_Client(payload))
    with pytest.raises(ValueError, match="response|output-files"):
        adapter.fetch_page({})


@pytest.mark.parametrize("run_ids", [[], [1, 1], [0], ["bad"]])
def test_openml_run_onnx_source_requires_unique_positive_run_ids(run_ids: list[Any]) -> None:
    with pytest.raises(ValueError):
        OpenMLRunOnnxCheckpointSourceAdapter(run_ids=run_ids, client=_Client())
