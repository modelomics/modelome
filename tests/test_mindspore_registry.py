from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.mindspore_registry import MindSporeModelZooSourceAdapter

_ROOT = "https://download.mindspore.cn/model_zoo/official/"
_LENET = "https://download.mindspore.cn/model_zoo/official/cv/lenet/"
_LITE = "https://download.mindspore.cn/model_zoo/official/lite/"
_PACKAGE = (
    "https://download.mindspore.cn/model_zoo/official/cv/lenet/"
    "lenet_ascend_0.5.0_mnist_official_classification_20200716/"
)
_CHECKPOINT = _PACKAGE + "lenet.ckpt"


class _Client:
    def __init__(self, documents: Mapping[str, str]) -> None:
        self.documents = documents
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if url not in self.documents:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(200, {}, self.documents[url].encode(), url)


def _adapter(client: _Client) -> MindSporeModelZooSourceAdapter:
    return MindSporeModelZooSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_registry_emits_only_direct_checkpoint_files_inside_official_tree() -> None:
    client = _Client(
        {
            _ROOT: '<a href="cv/">cv/</a><a href="../community/">community/</a>',
            _ROOT + "cv/": (
                '<a href="lenet/">lenet/</a>'
                '<a href="weights.ckpt">weights.ckpt</a>'
                '<a href="README.md">README.md</a>'
            ),
            _LENET: (
                '<a href="lenet_ascend_0.5.0_mnist_official_classification_20200716/">package/</a>'
                '<a href="lenet.ckpt.meta">metadata</a>'
            ),
            _PACKAGE: (
                '<a href="lenet.ckpt">lenet.ckpt</a>'
                '<a href="training.zip">training.zip</a>'
                '<a href="https://example.test/model.ckpt">external</a>'
            ),
        }
    )

    page = _adapter(client).fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert {record.canonical_url for record in page.records} == {
        _ROOT + "cv/weights.ckpt",
        _CHECKPOINT,
    }
    lenet = next(record for record in page.records if record.canonical_url == _CHECKPOINT)
    assert lenet.kind is ArtifactKind.WEIGHTS
    assert lenet.models[0].status is ModelStatus.RELEASED
    assert lenet.models[0].identifiers == (
        Identifier(
            "mindspore:modelzoo-checkpoint",
            "cv/lenet/lenet_ascend_0.5.0_mnist_official_classification_20200716/lenet.ckpt",
        ),
    )
    assert lenet.releases[0].metadata["artifact_url"] == _CHECKPOINT


def test_registry_rechecks_indexes_and_skips_unchanged_snapshot() -> None:
    client = _Client(
        {
            _ROOT: '<a href="cv/">cv/</a>',
            _ROOT + "cv/": '<a href="weights.ckpt">weights.ckpt</a>',
        }
    )
    adapter = _adapter(client)
    first = adapter.fetch_page({})

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.upstream_count == 1
    assert len(client.calls) == 4


def test_registry_normalizes_apache_sort_links_and_admits_official_lite_ms_weights() -> None:
    model_url = _LITE + "mobilenetv2_openimage_lite/"
    client = _Client(
        {
            _ROOT: (
                '<a href="lite/?C=N&amp;O=D">sorted lite/</a>'
                '<a href="lite/">lite/</a>'
            ),
            _LITE: '<a href="mobilenetv2_openimage_lite/">model/</a>',
            model_url: (
                '<a href="mobilenetv2.ms">mobilenetv2.ms</a>'
                '<a href="mobilenetv2.mindir">mobilenetv2.mindir</a>'
                '<a href="readme.txt">readme.txt</a>'
            ),
        }
    )

    page = _adapter(client).fetch_page({})

    assert client.calls.count(_LITE) == 1
    assert {record.canonical_url for record in page.records} == {
        model_url + "mobilenetv2.ms",
        model_url + "mobilenetv2.mindir",
    }
    ms_record = next(record for record in page.records if record.canonical_url.endswith(".ms"))
    assert ms_record.models[0].name == "mobilenetv2"
