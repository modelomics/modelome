from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

_REVISION = "e" * 40


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
    return HttpResponse(200, {}, body, "https://fixtures.test/markdown-catalog")


_DOCUMENT = """\
# Released Models

## Speech-to-Text

Acoustic Model | Training Data | Download
:-------------:| :------------: | :------:
[Example ASR](https://weights.example.test/asr.tar.gz) | LibriSpeech | [source](https://github.com/example/asr)

## Language Model based on NGram

Model | Dataset | Download
----- | ------- | --------
[Not a neural model](https://weights.example.test/ngram.klm) | Web | [download](https://weights.example.test/ngram.klm)

## Text-to-Speech

| Model Type | Dataset | Pretrained Models |
| :--------- | :------ | :---------------- |
""" + (
    "| FastSpeech2 | CSMSC | [checkpoint](https://weights.example.test/fs2.zip) / "
    "[ONNX](https://weights.example.test/fs2.onnx) |\n"
)


def _adapter(
    *responses: HttpResponse,
    model_header_pattern: str | None = None,
    model_name_pattern: str | None = None,
    checkpoint_row_pattern: str | None = None,
    identity_include_heading: bool = True,
    identity_context_columns: tuple[int, ...] | None = None,
) -> MarkdownModelTableSourceAdapter:
    return MarkdownModelTableSourceAdapter(
        name="example-markdown-catalog",
        repository="example/models",
        branch="main",
        document_path="docs/models.md",
        provider_namespace="example:model",
        model_header_pattern=model_header_pattern,
        model_name_pattern=model_name_pattern,
        dataset_column=None if identity_context_columns is not None else 1,
        identity_context_columns=identity_context_columns,
        checkpoint_row_pattern=checkpoint_row_pattern,
        identity_include_heading=identity_include_heading,
        excluded_headings=("NGram",),
        client=_QueuedClient(*responses),
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )


def test_markdown_table_catalog_emits_only_explicit_neural_artifact_rows() -> None:
    adapter = _adapter(_response({"sha": _REVISION}), _response(_DOCUMENT))

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["model_count"] == 2
    assert page.next_state["skipped_nonmodel_rows"] == 1
    asr, tts = page.records
    assert asr.kind is ArtifactKind.MODEL_CARD
    assert asr.source_record_id == "model:Speech-to-Text / Example ASR / LibriSpeech"
    assert asr.identifiers == (
        Identifier("example:model", "Speech-to-Text / Example ASR / LibriSpeech"),
    )
    assert {(link.url, link.relation, link.crawl) for link in asr.links} >= {
        ("https://weights.example.test/asr.tar.gz", "weights", False),
        ("https://github.com/example/asr", "source_implementation", False),
    }
    assert tts.source_record_id == "model:Text-to-Speech / FastSpeech2 / CSMSC"
    assert {(link.url, link.relation) for link in tts.links} >= {
        ("https://weights.example.test/fs2.zip", "weights"),
        ("https://weights.example.test/fs2.onnx", "inference_artifact"),
    }


def test_markdown_table_catalog_skips_document_when_commit_is_unchanged() -> None:
    first = _adapter(_response({"sha": _REVISION}), _response(_DOCUMENT))
    first_page = first.fetch_page({})
    second_client = _QueuedClient(_response({"sha": _REVISION}))
    first.client = second_client

    second_page = first.fetch_page(first_page.next_state)

    assert second_page.complete is True
    assert second_page.records == ()
    assert len(second_client.calls) == 1


def test_markdown_table_catalog_retains_artifacts_from_repeated_model_rows() -> None:
    conflicting = _DOCUMENT + (
        "| FastSpeech2 | CSMSC | "
        "[other](https://weights.example.test/other.zip) |\n"
    )
    adapter = _adapter(_response({"sha": _REVISION}), _response(conflicting))

    page = adapter.fetch_page({})

    tts = next(record for record in page.records if record.title == "FastSpeech2")
    assert ("https://weights.example.test/other.zip", "weights", False) in {
        (link.url, link.relation, link.crawl) for link in tts.links
    }


def test_markdown_table_catalog_can_use_a_source_specific_checkpoint_header() -> None:
    document = """\
# Segmentation releases

## PASCAL VOC

Checkpoint name | Score | Archive
--------------- | ----- | -------
[example_deeplab_checkpoint](https://weights.example.test/deeplab.tar.gz) | 82 | download
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Checkpoint name$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "example_deeplab_checkpoint"
    assert (
        "https://weights.example.test/deeplab.tar.gz",
        "weights",
        False,
    ) in {(link.url, link.relation, link.crawl) for link in record.links}


def test_markdown_table_catalog_can_extract_a_source_declared_model_prefix() -> None:
    document = (
        "# Segment Anything\n\n## Checkpoints\n\n| Model | Download |\n| --- | --- |\n"
        "| sam2.1_hiera_tiny <br /> ([config](sam2/configs/tiny.yaml), "
        "[checkpoint](https://weights.example.test/tiny.pt)) | 38.9 |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model$",
        model_name_pattern=r"^(sam2(?:\.1)?_hiera_[a-z_]+)",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.records[0].title == "sam2.1_hiera_tiny"
    assert page.records[0].models[0].identifiers == (
        Identifier("example:model", "Checkpoints / sam2.1_hiera_tiny / 38.9"),
    )
    assert ("https://weights.example.test/tiny.pt", "weights") in {
        (link.url, link.relation) for link in page.records[0].links
    }


def test_markdown_table_catalog_retains_source_declared_metrics_as_evaluation_evidence() -> None:
    document = (
        "# Speech models\n\n| Model Name | Checkpoint | Metrics |\n| --- | --- | --- |\n"
        "| ExampleSpeech | [checkpoint](https://weights.example.test/model.pt) | "
        "[metrics](https://weights.example.test/metrics.zip) |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model Name$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert {
        (link.url, link.relation) for link in page.records[0].links
    } >= {
        ("https://weights.example.test/model.pt", "weights"),
        ("https://weights.example.test/metrics.zip", "benchmark_results"),
    }


def test_markdown_table_catalog_retains_source_declared_logs_as_evidence() -> None:
    document = (
        "# Vision models\n\n| Model Name | Download |\n| --- | --- |\n"
        "| ExampleVision | [model](https://weights.example.test/model.pth) "
        "[logs](https://weights.example.test/training.txt) |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model Name$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert {
        (link.url, link.relation) for link in page.records[0].links
    } >= {
        ("https://weights.example.test/model.pth", "weights"),
        ("https://weights.example.test/training.txt", "training_log"),
    }


def test_markdown_table_catalog_uses_generic_link_column_headers_as_relation_evidence() -> None:
    document = (
        "# Vision models\n\n| Model Name | Weight | Log |\n| --- | --- | --- |\n"
        "| ExampleVision | [link](https://weights.example.test/model) | "
        "[link](https://logs.example.test/run) |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model Name$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert {
        (link.url, link.relation) for link in page.records[0].links
    } >= {
        ("https://weights.example.test/model", "weights"),
        ("https://logs.example.test/run", "training_log"),
    }


def test_markdown_table_catalog_can_unify_identical_model_rows_across_headings() -> None:
    document = (
        "# Example models\n\n## Classification\n\n"
        "| Model | Initialized checkpoint | Weight |\n| --- | --- | --- |\n"
        "| ExampleBase | [pretrain](https://weights.example.test/pretrain.pt) | "
        "[link](https://weights.example.test/classification.pt) |\n\n"
        "## Segmentation\n\n"
        "| Model | Initialized checkpoint | Weight |\n| --- | --- | --- |\n"
        "| ExampleBase | [pretrain](https://weights.example.test/pretrain.pt) | "
        "[link](https://weights.example.test/segmentation.pt) |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model$",
        identity_include_heading=False,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "model:ExampleBase / pretrain"
    assert {
        (link.url, link.relation) for link in record.links
    } >= {
        ("https://weights.example.test/pretrain.pt", "weights"),
        ("https://weights.example.test/classification.pt", "weights"),
        ("https://weights.example.test/segmentation.pt", "weights"),
    }


def test_markdown_table_catalog_can_preserve_multiple_source_context_columns() -> None:
    document = (
        "# I-JEPA\n\n| arch. | patch | resolution | epochs | data | download |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| ViT-H | 14x14 | 224x224 | 300 | ImageNet-1K | "
        "[full checkpoint](https://weights.example.test/vith14.pth.tar) |\n"
        "| ViT-H | 16x16 | 448x448 | 300 | ImageNet-1K | "
        "[full checkpoint](https://weights.example.test/vith16.pth.tar) |\n"
    )
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^arch\.$",
        identity_context_columns=(1, 2, 3, 4),
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    assert {record.source_record_id for record in page.records} == {
        "model:I-JEPA / ViT-H / 14x14 / 224x224 / 300 / ImageNet-1K",
        "model:I-JEPA / ViT-H / 16x16 / 448x448 / 300 / ImageNet-1K",
    }


def test_markdown_table_catalog_recognizes_pkl_checkpoint_urls_without_weight_labels() -> None:
    document = """\
# Video models

| Model Name | Download |
| --- | --- |
| ExampleVideo | [link](https://weights.example.test/model.pkl) |
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Model Name$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert ("https://weights.example.test/model.pkl", "weights") in {
        (link.url, link.relation) for link in page.records[0].links
    }


def test_markdown_table_catalog_accepts_direct_artifact_hosts_without_filename_suffix() -> None:
    document = """\
# Fashion Model Zoo

## Attribute Prediction

Backbone | Pooling | Download (Google)
-------- | ------- | -----------------
VGG-16 | Global | [model](https://drive.google.com/file/d/vgg16/view)

## Virtual Try-on

Model type | Dataset | Download (Google)
---------- | ------- | -----------------
CP-VTON | VTON | [GMM](https://drive.google.com/file/d/gmm/view) [TOM](https://drive.google.com/file/d/tom/view)
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^(?:Backbone|Model type)$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    vgg = next(record for record in page.records if record.title == "VGG-16")
    try_on = next(record for record in page.records if record.title == "CP-VTON")
    assert ("https://drive.google.com/file/d/vgg16/view", "weights", False) in {
        (link.url, link.relation, link.crawl) for link in vgg.links
    }
    assert {
        (link.url, link.relation, link.crawl)
        for link in try_on.links
    } >= {
        ("https://drive.google.com/file/d/gmm/view", "model_artifact", False),
        ("https://drive.google.com/file/d/tom/view", "model_artifact", False),
    }


def test_markdown_table_catalog_ignores_commented_rows_and_normalizes_nested_labels() -> None:
    document = """\
# Re-ID Model Zoo

<!--
| Method | Download |
| ------ | -------- |
| Hidden | [model](https://weights.example.test/hidden.pth) |
-->

| Method | Download |
| ------ | -------- |
| [UDA_TP](../tools/uda_tp) | [[model]](https://drive.google.com/file/d/uda-tp/view) |
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Method$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.records[0].title == "UDA_TP"
    assert all(record.title != "Hidden" for record in page.records)
    assert ("https://drive.google.com/file/d/uda-tp/view", "model_artifact") in {
        (link.url, link.relation) for link in page.records[0].links
    }


def test_markdown_table_catalog_accepts_visible_embedded_html_rows() -> None:
    document = """\
# Fixture catalog

## Classification

<table><tbody>
<tr><th>model</th><th>download</th></tr>
<!-- non-rendered example:
<tr><td>HiddenNet</td><td><a href="https://example.test/hidden.pth">model</a></td></tr>
-->
<tr>
  <td><a href="configs/visible.yaml">VisibleNet</a></td>
  <td><a href="https://weights.example.test/visible.pyth">model</a></td>
</tr>
</tbody></table>
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^model$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "VisibleNet"
    assert record.models[0].identifiers == (
        Identifier("example:model", "Classification / VisibleNet / model"),
    )
    assert {
        (link.url, link.relation)
        for link in record.links
        if link.relation not in {"model_card", "source_repository"}
    } == {("https://weights.example.test/visible.pyth", "weights")}


def test_markdown_table_catalog_ignores_nested_html_tables() -> None:
    document = """\
# Fixture catalog

## Classification

<table><tbody>
<tr><th>model</th><th>download</th></tr>
<tr>
  <td>VisibleNet <table><tr><td>unrelated layout</td></tr></table></td>
  <td><a href="https://weights.example.test/visible.pyth">model</a></td>
</tr>
</tbody></table>
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^model$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert page.records[0].title == "VisibleNet"
    assert {link.url for link in page.records[0].links} >= {
        "https://weights.example.test/visible.pyth"
    }


def test_markdown_table_catalog_reads_transposed_html_checkpoint_columns() -> None:
    document = """\
# MAE releases

<table><tbody>
<th></th>
<th>ViT-Base</th>
<th>ViT-Large</th>
<tr>
  <td>pre-trained checkpoint</td>
  <td><a href="https://weights.example.test/vit-base.pth">download</a></td>
  <td><a href="https://weights.example.test/vit-large.pth">download</a></td>
</tr>
<tr><td>md5</td><td>abc</td><td>def</td></tr>
</tbody></table>
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        checkpoint_row_pattern=r"^pre-trained checkpoint$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 2
    assert [record.title for record in page.records] == ["ViT-Base", "ViT-Large"]
    assert {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    } == {
        "https://weights.example.test/vit-base.pth",
        "https://weights.example.test/vit-large.pth",
    }


def test_markdown_table_catalog_normalizes_legacy_tex_display_table_cells() -> None:
    document = """\
# Legacy detection zoo

### Mask R-CNN

^{_{backbone}} | ^{_{model id}} | ^{_{download links}}
--- | --- | ---
^{_{X-101-32x8d-FPN}} | ^{_{37121596}} | ^{_{[model](https://weights.example.test/x101.pkl)}}
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^backbone$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "X-101-32x8d-FPN"
    assert record.models[0].identifiers == (
        Identifier("example:model", "Mask R-CNN / X-101-32x8d-FPN / 37121596"),
    )
    assert ("https://weights.example.test/x101.pkl", "weights") in {
        (link.url, link.relation) for link in record.links
    }


def test_markdown_table_catalog_keeps_explicit_row_papers_as_paper_references() -> None:
    document = """\
# Released models

Method | Paper | Download
------ | ----- | --------
ExampleNet | [paper](https://arxiv.org/abs/2401.12345) | [model](https://weights.example.test/example.pth)
"""
    adapter = _adapter(
        _response({"sha": _REVISION}),
        _response(document),
        model_header_pattern=r"^Method$",
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    assert {
        (link.url, link.relation)
        for link in page.records[0].links
        if link.relation not in {"model_card", "source_repository"}
    } == {
        ("https://arxiv.org/abs/2401.12345", "paper_reference"),
        ("https://weights.example.test/example.pth", "weights"),
    }
