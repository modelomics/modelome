from __future__ import annotations

import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.opencv_dnn_model_index import OpenCVDnnModelIndexSourceAdapter

_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/models.yml"
_PROPOSAL = Path(__file__).parents[1] / "config/proposals/opencv_dnn_model_index.toml"


class _Client:
    def __init__(self, body: str, *, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        return HttpResponse(
            self.status,
            {"ETag": '"fixture"'},
            self.body,
            url,
        )


_INDEX = '''\
%YAML 1.0
---
# A declared Caffe checkpoint.
opencv_fd:
  load_info:
    url: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10.caffemodel"
    sha1: "15aa726b4d46d9f023526d85537db81cbc8dd566"
  model: "opencv_face_detector.caffemodel"
  config: "opencv_face_detector.prototxt"
  sample: "object_detection"
# A TensorFlow archive with a declared extracted graph path.
ssd_tf:
  load_info:
    url: "http://download.tensorflow.org/models/object_detection/ssd_mobilenet.tar.gz"
    sha1: "9e4bcdd98f4c6572747679e4ce570de4f03a70e2"
    download_sha: "6157ddb6da55db2da89dd561eceb7f944928e317"
    download_name: "ssd_mobilenet.tar.gz"
    member: "ssd_mobilenet/frozen_inference_graph.pb"
  model: "ssd_mobilenet.pb"
  config: "ssd_mobilenet.pbtxt"
  sample: "object_detection"
# No load_info.url: not a checkpoint entry.
documented_only:
  model: "model.onnx"
'''


def test_opencv_dnn_index_extracts_exact_checkpoint_urls_and_metadata() -> None:
    client = _Client(_INDEX)
    adapter = OpenCVDnnModelIndexSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["entry_count"] == 2
    assert client.calls == [(_URL, {"Accept": "text/yaml,text/plain"})]
    caffe, tensorflow = page.records
    assert caffe.identifiers[0].namespace == "opencv:dnn-sample-model"
    assert caffe.identifiers[0].value == "opencv_fd"
    assert caffe.releases[0].metadata["format"] == "caffemodel"
    assert caffe.releases[0].metadata["sha1"] == (
        "15aa726b4d46d9f023526d85537db81cbc8dd566"
    )
    assert any(
        link.url.endswith("res10.caffemodel") and link.relation == "weights"
        for link in caffe.links
    )
    assert tensorflow.releases[0].metadata["format"] == "pb"
    assert tensorflow.releases[0].metadata["download_sha"] == (
        "6157ddb6da55db2da89dd561eceb7f944928e317"
    )
    assert tensorflow.releases[0].metadata["download_name"] == "ssd_mobilenet.tar.gz"
    assert tensorflow.releases[0].metadata["member"] == (
        "ssd_mobilenet/frozen_inference_graph.pb"
    )
    assert tensorflow.raw["model_filename"] == "ssd_mobilenet.pb"
    assert tensorflow.raw["config_filename"] == "ssd_mobilenet.pbtxt"


def test_same_checkpoint_basename_under_distinct_paths_keeps_models_separate() -> None:
    document = '''\
model_a:
  load_info:
    url: "https://weights.example/a/shared.onnx"
  model: "shared.onnx"
model_b:
  load_info:
    url: "https://weights.example/b/shared.onnx"
  model: "shared.onnx"
'''
    records = OpenCVDnnModelIndexSourceAdapter(client=_Client(document)).fetch_page({}).records

    assert len(records) == 2
    assert len({record.source_record_id for record in records}) == 2
    assert len({record.releases[0].local_id for record in records}) == 2
    result = build_entries(
        source_record_to_entry_seed(record, source="opencv-dnn-index")
        for record in records
    )
    assert len(result.entries) == 2


def test_opencv_dnn_index_honors_conditional_request_and_size_bounds() -> None:
    unchanged_client = _Client("", status=304)
    unchanged = OpenCVDnnModelIndexSourceAdapter(client=unchanged_client).fetch_page(
        {"etag": '"old"', "entry_count": 7}
    )
    assert unchanged.records == ()
    assert unchanged.next_state["checked_at"]
    assert unchanged_client.calls[0][1]["If-None-Match"] == '"old"'

    oversized = OpenCVDnnModelIndexSourceAdapter(
        client=_Client(_INDEX),
        max_response_bytes=10,
    )
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        oversized.fetch_page({})


def test_opencv_dnn_proposal_is_disabled_and_matches_adapter() -> None:
    source = tomllib.loads(_PROPOSAL.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = OpenCVDnnModelIndexSourceAdapter(
        name=source["name"],
        url=source["url"],
        repository_url=source["repository_url"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
    )
    assert adapter.name == source["name"]
    assert adapter.url == _URL
