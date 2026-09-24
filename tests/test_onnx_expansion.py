from __future__ import annotations

from datetime import UTC, datetime

from modelome.http import HttpResponse
from modelome.sources.onnx_model_zoo import OnnxModelZooSourceAdapter


class Client:
    def __init__(self, body: str) -> None:
        self.body = body.encode()
        self.calls = 0

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls += 1
        return HttpResponse(status=200, headers={}, body=self.body, url=url)


def test_onnx_zoo_captures_only_first_party_assets_beneath_model_identity() -> None:
    body = (
        "# ONNX Model Zoo\n"
        "|Model|Reference|Assets|\n|---|---|---|\n"
        "|[Tiny](validated/vision/tiny)|[Paper](https://arxiv.org/abs/2401.00001)|"
        "[ONNX](model.onnx) [Weights](https://github.com/onnx/models/blob/main/"
        "validated/vision/tiny/weights.data) [Other](../other/model.onnx) "
        "[Foreign](https://github.com/someone/other/blob/main/validated/vision/tiny/fake.onnx)|\n"
    )
    client = Client(body)
    page = OnnxModelZooSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 22, tzinfo=UTC)
    ).fetch_page({})

    assert client.calls == 1
    assert len(page.records) == 1
    record = page.records[0]
    assert record.raw["model_path"] == "validated/vision/tiny"
    assert record.raw["first_party_assets"] == [
        "validated/vision/tiny/model.onnx",
        "validated/vision/tiny/weights.data",
    ]
    assert (
        "https://github.com/onnx/models/blob/main/validated/vision/tiny/model.onnx",
        "inference_artifact",
    ) in {(link.url, link.relation) for link in record.links}
    assert not any(
        "other/model.onnx" in path or "fake.onnx" in path
        for path in record.raw["first_party_assets"]
    )
