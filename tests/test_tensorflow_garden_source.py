from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.tensorflow_garden import TensorFlowGardenSourceAdapter

REVISION = "a" * 40


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self, url: str, *, params: Any = None, headers: dict[str, str] | None = None
    ) -> HttpResponse:
        assert params is None
        self.calls.append(url)
        return self.responses.pop(0)


def response(value: str | dict[str, Any]) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(status=200, headers={}, body=body, url="https://example.test")


def test_garden_emits_exact_checkpoint_linked_model_rows_and_advances_one_doc() -> None:
    vision = "\n".join(
        [
            "| Model | Resolution | Download |",
            "|---|---|---|",
            "| ResNet-50 | 224 | [ckpt](https://storage.googleapis.com/garden/model.tar.gz) |",
            "| Foo | 5 | [config](https://example.test/cfg) |",
        ]
    )
    client = Client(response({"sha": REVISION}), response(vision))
    adapter = TensorFlowGardenSourceAdapter(client=client)
    page = adapter.fetch_page({})

    assert page.complete is False
    assert page.next_state["doc_index"] == 1
    assert len(page.records) == 1
    record = page.records[0]
    assert record.title == "ResNet-50"
    assert record.identifiers == (
        Identifier(
            "tensorflow-model-garden:model",
            "official/vision/MODEL_GARDEN.md:root:ResNet-50",
        ),
    )
    assert record.releases[0].metadata["checkpoints"] == [
        "https://storage.googleapis.com/garden/model.tar.gz"
    ]
    assert client.calls == [
        adapter.commit_url,
        adapter.raw_url(REVISION, "official/vision/MODEL_GARDEN.md"),
    ]


def test_garden_pins_revision_until_all_docs_are_scanned_and_identity_survives_insertions() -> None:
    first_doc = "\n".join(
        [
            "| Model | Checkpoint |",
            "|---|---|",
            "| Added model | [ckpt](https://example.test/new) |",
            "| ResNet-50 | [ckpt](https://example.test/r50) |",
        ]
    )
    nlp = "\n".join(
        [
            "#### BERT",
            (
                "| Model | Configuration | Training Data | Checkpoint & Vocabulary | "
                "Kaggle SavedModels |"
            ),
            "|---|---|---|---|---|",
            (
                "| BERT-base uncased English | config | Wiki | "
                "[checkpoint](https://storage.googleapis.com/tf_model_garden/nlp/bert.tar.gz) | "
                "[saved](https://tfhub.dev/tf/bert/1) |"
            ),
        ]
    )
    client = Client(response({"sha": REVISION}), response(first_doc), response(nlp))
    adapter = TensorFlowGardenSourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.records[1].identifiers == (
        Identifier(
            "tensorflow-model-garden:model",
            "official/vision/MODEL_GARDEN.md:root:ResNet-50",
        ),
    )
    assert len(second.records) == 1
    assert second.records[0].title == "BERT-base uncased English"
    assert (
        Identifier("tensorflow:model-garden-nlp-model", "BERT / BERT-base uncased English")
        in second.records[0].models[0].identifiers
    )
    assert second.records[0].releases[0].metadata["checkpoints"] == [
        "https://storage.googleapis.com/tf_model_garden/nlp/bert.tar.gz",
        "https://tfhub.dev/tf/bert/1",
    ]
    assert second.records[0].releases[0].version == "1"
    assert client.calls == [
        adapter.commit_url,
        adapter.raw_url(REVISION, "official/vision/MODEL_GARDEN.md"),
        adapter.raw_url(REVISION, "official/nlp/docs/pretrained_models.md"),
    ]


def test_garden_identity_does_not_include_table_row_position() -> None:
    markdown = "\n".join(
        [
            "| Model | Checkpoint |",
            "|---|---|",
            "| Other | [ckpt](https://example.test/other) |",
            "| Stable | [ckpt](https://example.test/stable) |",
        ]
    )
    shifted = (
        "| Model | Checkpoint |\n|---|---|\n| Stable | [ckpt](https://example.test/stable) |\n"
    )
    adapter = TensorFlowGardenSourceAdapter(client=Client())
    original = next(
        record
        for record in adapter._records("docs.md", REVISION, markdown)
        if record.title == "Stable"
    )
    moved = next(
        record
        for record in adapter._records("docs.md", REVISION, shifted)
        if record.title == "Stable"
    )
    assert original.identifiers == moved.identifiers


def test_garden_captures_first_party_nlp_experiment_checkpoint_without_sentencepiece() -> None:
    document = "\n".join(
        [
            "| NAME | EXPERIMENT | TASK_CONFIG | MODEL_CONFIG | EXRTRA_PARAMS |",
            "|---|---|---|---|---|",
            (
                "| BERT-base GLUE/MNLI-matched finetune | bert/sentence_prediction | "
                "glue.yaml | bert.yaml | data and bert-base ckpt "
                "task.init_checkpoint=gs://tf_model_garden/nlp/bert/uncased/bert_model.ckpt |"
            ),
            (
                "| Transformer-large WMT14/en-de scratch | wmt_transformer/large | | | "
                "ende-32k sentencepiece "
                "task.sentencepiece_model_path='gs://tf_model_garden/nlp/transformer/model.model' |"
            ),
        ]
    )
    adapter = TensorFlowGardenSourceAdapter(client=Client())

    records = adapter._records("official/nlp/MODEL_GARDEN.md", REVISION, document)

    assert len(records) == 1
    assert records[0].title == "BERT-base GLUE/MNLI-matched finetune"
    assert records[0].releases[0].metadata["checkpoints"] == [
        "https://storage.googleapis.com/tf_model_garden/nlp/bert/uncased/bert_model.ckpt"
    ]


def test_garden_checkpoint_change_adds_release_to_same_model() -> None:
    adapter = TensorFlowGardenSourceAdapter(client=Client())
    before = adapter._records(
        "docs.md",
        REVISION,
        "| Model | Checkpoint |\n|---|---|\n| Stable | [ckpt](https://example.test/v1) |",
    )[0]
    after = adapter._records(
        "docs.md",
        REVISION,
        "| Model | Checkpoint |\n|---|---|\n| Stable | [ckpt](https://example.test/v2) |",
    )[0]

    assert before.models[0].identifiers == after.models[0].identifiers
    assert before.releases[0].identifiers != after.releases[0].identifiers


def test_garden_only_sets_tfhub_version_when_checkpoint_declarations_agree() -> None:
    adapter = TensorFlowGardenSourceAdapter(client=Client())

    def release(checkpoints: str):
        return adapter._records(
            "docs.md",
            REVISION,
            f"| Model | Checkpoint |\n|---|---|\n| Stable | {checkpoints} |",
        )[0].releases[0]

    assert release("[Hub](https://tfhub.dev/tf/bert/1)").version == "1"
    assert release("[Hub](https://tfhub.dev/tf/bert)").version is None
    assert release(
        "[Hub A](https://tfhub.dev/tf/bert/1) [Hub B](https://tfhub.dev/tf/bert/2)"
    ).version is None
    assert release("[ckpt](https://storage.googleapis.com/model/v1/model.tar.gz)").version is None


def test_garden_rejects_bad_cursor_and_invalid_commit_sha() -> None:
    with pytest.raises(ValueError, match="invalid document cursor"):
        TensorFlowGardenSourceAdapter(client=Client()).fetch_page({"doc_index": -1})
    with pytest.raises(ValueError, match="SHA-1"):
        TensorFlowGardenSourceAdapter(client=Client(response({"sha": "bad"}))).fetch_page({})


def test_garden_captures_official_vision_readme_yolov7_checkpoint_row() -> None:
    # The current official/vision/README.md contains this published checkpoint
    # table, but the older MODEL_GARDEN.md document does not. The table's model
    # column is named "Variant" rather than "Model".
    document = "\n".join(
        [
            "### YOLOv7 (Trained from scratch)",
            "| Variant | Resolution | Epochs | FLOPs (B) | Params (M) | Box AP | Download |",
            "|---|---|---|---|---|---|---|",
            (
                "| YOLOv7 | 640x640 | 300 | 53.16 | 44.57 | 50.5 | "
                "[config](https://github.com/tensorflow/models/blob/master/"
                "official/projects/yolo/configs/experiments/yolov7/detection/yolov7.yaml) | "
                "[ckpt](https://storage.googleapis.com/tf_model_garden/vision/"
                "yolo/yolov7/yolov7.tar.gz) |"
            ),
        ]
    )
    adapter = TensorFlowGardenSourceAdapter(client=Client())

    records = adapter._records("official/vision/README.md", REVISION, document)

    assert len(records) == 1
    record = records[0]
    assert record.title == "YOLOv7"
    assert record.raw["document"] == "official/vision/README.md"
    assert record.releases[0].metadata["checkpoints"] == [
        "https://storage.googleapis.com/tf_model_garden/vision/yolo/yolov7/yolov7.tar.gz"
    ]


def test_garden_fetches_added_vision_readme_at_pinned_revision() -> None:
    markdown = "\n".join(
        [
            "### YOLOv7 (Trained from scratch)",
            "| Variant | Resolution | Epochs | FLOPs (B) | Params (M) | Box AP | Download |",
            "|---|---|---|---|---|---|---|",
            (
                "| YOLOv7 | 640x640 | 300 | 53.16 | 44.57 | 50.5 | "
                "[config](https://github.com/tensorflow/models/blob/master/"
                "official/projects/yolo/configs/experiments/yolov7/detection/yolov7.yaml) | "
                "[ckpt](https://storage.googleapis.com/tf_model_garden/vision/"
                "yolo/yolov7/yolov7.tar.gz) |"
            ),
        ]
    )
    client = Client(response(markdown))
    adapter = TensorFlowGardenSourceAdapter(client=client)

    page = adapter.fetch_page({"revision": REVISION, "doc_index": 3})

    assert [record.title for record in page.records] == ["YOLOv7"]
    assert page.records[0].raw["revision"] == REVISION
    assert client.calls == [adapter.raw_url(REVISION, "official/vision/README.md")]


def test_garden_resolves_first_party_config_initialization_checkpoint() -> None:
    table = "\n".join(
        [
            "### RetinaNet (ImageNet pretrained)",
            "| Backbone | Resolution | Download |",
            "|---|---|---|",
            (
                "| R50-FPN | 640x640 | "
                "[config](https://github.com/tensorflow/models/blob/master/"
                "official/vision/configs/retinanet.py#L187-L258) | "
                "[ckpt](https://storage.googleapis.com/tf_model_garden/vision/"
                "retinanet/retinanet-resnet50fpn.tar.gz) |"
            ),
        ]
    )
    first_client = Client(response(table))
    adapter = TensorFlowGardenSourceAdapter(client=first_client)

    document_page = adapter.fetch_page({"revision": REVISION, "doc_index": 3})

    assert document_page.records[0].raw["config_paths"] == [
        "official/vision/configs/retinanet.py"
    ]
    assert document_page.next_state["config_queue"][0]["path"] == (
        "official/vision/configs/retinanet.py"
    )

    config = """
    task=RetinaNetTask(
        init_checkpoint='gs://cloud-tpu-checkpoints/vision-2.0/resnet50_imagenet/ckpt-28080',
        init_checkpoint_modules='backbone',
    )
    """
    config_client = Client(response(config))
    adapter.client = config_client
    config_page = adapter.fetch_page(document_page.next_state)

    assert config_page.complete is True
    assert len(config_page.records) == 1
    record = config_page.records[0]
    assert record.models[0].identifiers == document_page.records[0].models[0].identifiers
    assert record.releases[0].metadata == {
        "config_path": "official/vision/configs/retinanet.py",
        "checkpoint": (
            "https://storage.googleapis.com/cloud-tpu-checkpoints/vision-2.0/"
            "resnet50_imagenet/ckpt-28080"
        ),
        "checkpoint_modules": "backbone",
        "revision": REVISION,
    }
    assert config_client.calls == [
        adapter.raw_url(REVISION, "official/vision/configs/retinanet.py")
    ]


@pytest.mark.parametrize(
    ("rows", "config"),
    [
        (
            "| Model A | [config](https://github.com/tensorflow/models/blob/master/"
            "official/vision/configs/shared.py) | [ckpt](https://weights.test/a) |\n"
            "| Model B | [config](https://github.com/tensorflow/models/blob/master/"
            "official/vision/configs/shared.py) | [ckpt](https://weights.test/b) |",
            "init_checkpoint='gs://bucket/a.ckpt'",
        ),
        (
            "| Model A | [config](https://github.com/tensorflow/models/blob/master/"
            "official/vision/configs/shared.py) | [ckpt](https://weights.test/a) |",
            "init_checkpoint='gs://bucket/a.ckpt'\n"
            "other.init_checkpoint='gs://bucket/b.ckpt'",
        ),
    ],
)
def test_garden_skips_ambiguous_linked_config_checkpoint_associations(
    rows: str, config: str
) -> None:
    from modelome.sources.tensorflow_garden import _append_config_queue

    markdown = "| Model | Config | Checkpoint |\n|---|---|---|\n" + rows
    adapter = TensorFlowGardenSourceAdapter(
        client=Client(response(markdown), response(config))
    )
    document_page = adapter.fetch_page({"revision": REVISION, "doc_index": 3})
    queue = _append_config_queue([], document_page.records)
    assert len(queue) == 1

    config_page = adapter.fetch_page(
        {
            "revision": REVISION,
            "doc_index": 4,
            "config_queue": queue,
            "config_index": 0,
        }
    )

    assert config_page.records == ()


def test_garden_reads_literal_yaml_init_checkpoint_without_fetching_artifact() -> None:
    from modelome.sources.tensorflow_garden import _declared_config_checkpoints

    assert _declared_config_checkpoints(
        "task:\n  init_checkpoint: gs://tf_model_garden/nlp/bert/bert_model.ckpt\n"
    ) == (
        (
            "https://storage.googleapis.com/tf_model_garden/nlp/bert/bert_model.ckpt",
            None,
        ),
    )


def test_garden_reads_parenthesized_multiline_python_checkpoint_literal() -> None:
    from modelome.sources.tensorflow_garden import _declared_config_checkpoints

    assert _declared_config_checkpoints(
        "task.init_checkpoint=(\n"
        "    'gs://tf_model_garden/vision/resnet/ckpt-42'\n"
        ")\n"
    ) == (
        (
            "https://storage.googleapis.com/tf_model_garden/vision/resnet/ckpt-42",
            None,
        ),
    )
    # The parser only accepts a single literal, never evaluates Python.
    assert _declared_config_checkpoints(
        "task.init_checkpoint=(prefix + 'gs://bucket/ckpt')\n"
    ) == ()
