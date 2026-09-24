from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.onnx_model_zoo import OnnxModelZooSourceAdapter

NOW = datetime(2026, 9, 21, 19, 30, tzinfo=UTC)
URL = "https://raw.githubusercontent.com/onnx/models/main/README.md"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Any = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        return self.responses.pop(0)


def response(
    body: str = "",
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=headers or {},
        body=body.encode(),
        url=URL,
    )


def test_onnx_model_zoo_preserves_per_model_artifact_and_paper_evidence() -> None:
    body = (
        "# ONNX Model Zoo\n\n"
        "### Image Classification\n"
        "|Model Class |Reference |Description |Huggingface Spaces|\n"
        "|-|-|-|-|\n"
        "|<b>[ResNet](validated/vision/classification/resnet)</b>|[He et al.]("
        "https://arxiv.org/abs/1512.03385)|A CNN model.|[Space]("
        "https://huggingface.co/spaces/onnx/ResNet)|\n"
        "|<b>[MobileNet](validated/vision/classification/mobilenet)</b>|[Sandler et al.]("
        "https://doi.org/10.48550/arXiv.1801.04381)|A mobile CNN.| |\n\n"
        "### Context only\n"
        "|Model Class |Reference |\n"
        "|-|-|\n"
        "|A model without a validated artifact|[Paper](https://arxiv.org/abs/2001.00001)|\n"
    )
    client = QueuedClient(
        response(
            body,
            headers={
                "ETag": '"onnx-zoo-r1"',
                "Last-Modified": "Mon, 21 Sep 2026 19:00:00 GMT",
            },
        )
    )
    adapter = OnnxModelZooSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["entry_count"] == 2
    # Table heading rows and the unlinked narrative row are deliberately not
    # admitted: only entries with an exact local validated-artifact path count.
    assert page.next_state["ignored_table_rows"] == 3
    assert page.next_state["etag"] == '"onnx-zoo-r1"'

    resnet = page.records[0]
    assert resnet.source_record_id == "model:validated/vision/classification/resnet"
    assert resnet.kind is ArtifactKind.MODEL_CARD
    assert resnet.canonical_url == (
        "https://github.com/onnx/models/tree/main/validated/vision/classification/resnet"
    )
    assert resnet.identifiers == (
        Identifier("onnx:model-zoo-artifact", "validated/vision/classification/resnet"),
    )
    assert resnet.models[0].name == "ResNet"
    assert resnet.models[0].identifiers == (
        Identifier("onnx:model-zoo", "validated/vision/classification/resnet"),
    )
    assert resnet.models[0].status is ModelStatus.RELEASED
    assert resnet.raw["historical_archive"] is True
    assert resnet.releases[0].metadata["format"] == "ONNX"
    links = {(link.url, link.relation) for link in resnet.links}
    assert ("https://arxiv.org/abs/1512.03385", "paper_reference") in links
    assert ("https://huggingface.co/spaces/onnx/ResNet", "artifact_reference") in links
    assert ("https://github.com/onnx/models", "source_repository") in links

    mobile = page.records[1]
    assert ("https://doi.org/10.48550/arXiv.1801.04381", "paper_reference") in {
        (link.url, link.relation) for link in mobile.links
    }


def test_onnx_model_zoo_reuses_conditional_catalog_headers() -> None:
    client = QueuedClient(response(status=304))
    adapter = OnnxModelZooSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page(
        {
            "etag": '"onnx-zoo-r1"',
            "http_last_modified": "Mon, 21 Sep 2026 19:00:00 GMT",
        }
    )

    assert page.records == ()
    assert page.complete is True
    assert page.next_state["checked_at"] == "2026-09-21T19:30:00Z"
    assert client.calls == [
        (
            URL,
            {
                "Accept": "text/markdown,text/plain",
                "If-None-Match": '"onnx-zoo-r1"',
                "If-Modified-Since": "Mon, 21 Sep 2026 19:00:00 GMT",
            },
        )
    ]
