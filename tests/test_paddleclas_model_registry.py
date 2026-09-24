from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.paddleclas_model_registry import (
    PaddleClasModelRegistrySourceAdapter,
)

_REVISION = "2" * 40
_SOURCE = b'''
IMN_MODEL_BASE_DOWNLOAD_URL = "https://weights.example.test/imn/{}_infer.tar"
IMN_MODEL_SERIES = {
    "ResNet": ["ResNet18", "ResNet50"],
    "MobileNet": ["MobileNetV3"],
}
PULC_MODEL_BASE_DOWNLOAD_URL = "https://weights.example.test/pulc/{}_infer.tar"
PULC_MODELS = ["car_exists"]
SHITU_MODEL_BASE_DOWNLOAD_URL = "https://weights.example.test/shitu/{}_infer.tar"
SHITU_MODELS = ["PP-ShiTuV2"]
'''


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(body: bytes, *, url: str = "https://fixtures.test/paddleclas") -> HttpResponse:
    return HttpResponse(200, {"etag": '"fixture"'}, body, url)


def _commit() -> HttpResponse:
    return _response(("{\"sha\": \"" + _REVISION + "\"}").encode())


def test_paddleclas_registry_enumerates_only_explicit_inference_models() -> None:
    client = _QueuedClient(_commit(), _response(_SOURCE))
    adapter = PaddleClasModelRegistrySourceAdapter(
        repository="example/PaddleClas",
        branch="release/2.6",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot
    assert page.upstream_count == 4
    assert page.next_state["catalog_counts"] == {"IMN": 3, "PULC": 1}
    assert client.calls[1][0].endswith(f"/{_REVISION}/paddleclas.py")
    resnet = next(record for record in page.records if record.title == "ResNet50")
    assert resnet.identifiers == (Identifier("paddleclas:model", "IMN:ResNet50"),)
    assert resnet.models[0].status.value == "released"
    assert resnet.models[0].locator == "paddleclas.py:IMN_MODEL_SERIES['ResNet'][1]"
    assert resnet.releases[0].identifiers == (
        Identifier("paddleclas:inference-model", "IMN:ResNet50"),
    )
    assert resnet.releases[0].metadata["family"] == "ResNet"
    assert (
        "https://weights.example.test/imn/ResNet50_infer.tar",
        "weights",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in resnet.links}
    assert all(link.model_local_ids == (resnet.models[0].local_id,) for link in resnet.links)
    assert all(record.title != "PP-ShiTuV2" for record in page.records)


def test_paddleclas_registry_skips_source_when_commit_is_unchanged() -> None:
    client = _QueuedClient(_commit())
    adapter = PaddleClasModelRegistrySourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 41})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 41
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "source, message",
    [
        (
            b"IMN_MODEL_BASE_DOWNLOAD_URL = 'https://weights.test/{}_{}.tar'\n"
            b"IMN_MODEL_SERIES = {'ResNet': ['ResNet18']}\n",
            "one '{}' slot",
        ),
        (
            b"IMN_MODEL_BASE_DOWNLOAD_URL = 'https://weights.test/{}_infer.tar'\n"
            b"IMN_MODEL_SERIES = {'ResNet': dynamic_names}\n",
            "expected a literal model list",
        ),
        (
            b"SHITU_MODEL_BASE_DOWNLOAD_URL = 'https://weights.test/{}_infer.tar'\n"
            b"SHITU_MODELS = ['PP-ShiTuV2']\n",
            "neither a literal IMN nor PULC",
        ),
    ],
)
def test_paddleclas_registry_fails_closed_on_nonliteral_or_unpaired_shapes(
    source: bytes, message: str
) -> None:
    client = _QueuedClient(_commit(), _response(source))

    with pytest.raises(ValueError, match=message):
        PaddleClasModelRegistrySourceAdapter(client=client).fetch_page({})
