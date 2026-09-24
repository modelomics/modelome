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
