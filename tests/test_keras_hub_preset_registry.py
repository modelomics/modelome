from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.keras_hub_preset_registry import KerasHubPresetRegistrySourceAdapter

_REVISION = "d" * 40


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


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/keras-hub")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    root = f"keras-hub-{_REVISION}"
    with zipfile.ZipFile(output, "w") as package:
        for path, text in files.items():
            package.writestr(f"{root}/{path}", text)
    return output.getvalue()


_BERT = '''\
backbone_presets_with_weights = {
    "bert_base_en": {
        "metadata": {
            "description": "BERT base " + "English",
            "params": 110000000,
            "path": "bert",
        },
        "kaggle_handle": "kaggle://keras/bert/keras/bert_base_en/3",
        "hf_handle": "hf://keras/bert-base-en",
    },
}

backbone_presets = {
    **backbone_presets_with_weights,
}
'''

_VIT = '''\
task_presets = {
    "vit_base_imagenet": {
        "metadata": {"path": "vit"},
        "kaggle_handle": "kaggle://keras/vit/keras/vit_base_imagenet",
    },
}
'''


def _catalog_archive() -> bytes:
    return _archive(
        {
            "keras_hub/src/models/bert/bert_presets.py": _BERT,
            "keras_hub/src/models/vision/vit_presets.py": _VIT,
            "keras_hub/src/models/bert/bert_backbone.py": "class BertBackbone: pass\n",
        }
    )


def test_preset_registry_enumerates_literal_provider_handles() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = KerasHubPresetRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["preset_file_count"] == 2
    assert page.next_state["preset_count"] == 2
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")

    bert, vit = page.records
    assert bert.kind is ArtifactKind.MODEL_CARD
    assert bert.source_record_id.startswith("preset:")
    assert bert.identifiers == (
        Identifier(
            "keras-hub:preset",
            "keras_hub/src/models/bert/bert_presets.py:backbone_presets:bert_base_en",
        ),
    )
    assert bert.models[0].name == "bert_base_en"
    assert bert.models[0].status is ModelStatus.RELEASED
    assert bert.raw["preset_source_sha256"]
    assert [release.identifiers[0].namespace for release in bert.releases] == [
        "keras-hub:preset-handle",
        "keras-hub:preset-handle",
    ]
    assert [release.version for release in bert.releases] == ["3", None]
    assert all(release.revision == _REVISION for release in bert.releases)
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in bert.links
    } == {
        ("https://github.com/keras-team/keras-hub", "source_repository", False, ()),
        (
            "https://github.com/keras-team/keras-hub/blob/"
            f"{_REVISION}/keras_hub/src/models/bert/bert_presets.py",
            "preset_definition",
            False,
            (bert.models[0].local_id,),
        ),
        (
            "https://www.kaggle.com/models/keras/bert/keras/bert_base_en/3",
            "model_artifact",
            False,
            (bert.models[0].local_id,),
        ),
        (
            "https://huggingface.co/keras/bert-base-en",
            "model_card",
            False,
            (bert.models[0].local_id,),
        ),
    }
    assert vit.models[0].name == "vit_base_imagenet"
    assert len(vit.releases) == 1
    assert vit.releases[0].version is None
    assert vit.releases[0].revision == _REVISION
    assert any(
        link.url == "https://www.kaggle.com/models/keras/vit/keras/vit_base_imagenet"
        for link in vit.links
    )


def test_preset_registry_skips_archive_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_catalog_archive()))
    adapter = KerasHubPresetRegistrySourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_preset_registry_fails_closed_for_a_dynamic_preset_dictionary() -> None:
    archive = _archive(
        {
            "keras_hub/src/models/bert/bert_presets.py": (
                "backbone_presets = build_presets()\n"
            ),
        }
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(archive))

    with pytest.raises(ValueError, match="not a literal dictionary"):
        KerasHubPresetRegistrySourceAdapter(client=client).fetch_page({})
