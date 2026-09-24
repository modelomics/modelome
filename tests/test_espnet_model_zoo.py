from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.espnet_model_zoo import EspnetModelZooSourceAdapter

_REVISION = "d" * 40
_HEADER = "corpus,task,name,url,fs,lang,gender,pytorch,espnet,commit,valid"


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
    return HttpResponse(200, {}, body, "https://fixtures.test/espnet")


_TABLE = "\n".join(
    (
        _HEADER,
        ",,test,https://zenodo.org/record/1/files/test.zip?download=1,,,,,,,true",
        "wsj,asr,kamo-naoyuki/wsj,https://zenodo.org/record/2/files/wsj.zip?download=1,16000,en,,1.6.0,0.9.1,e67a1ad,true",
        "aidatatang,asr,example/aidatatang,huggingface.co,16000,zh,,,0.21.0,,false",
    )
) + "\n"


def test_espnet_model_zoo_preserves_complete_rows_and_invalidated_release_evidence() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_TABLE))
    adapter = EspnetModelZooSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 21, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["model_count"] == 2
    assert page.next_state["validated_model_count"] == 1
    assert page.next_state["invalidated_model_count"] == 1
    assert page.next_state["skipped_control_rows"] == 1
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/{_REVISION}/espnet_model_zoo/table.csv")

    zenodo, hub = page.records
    assert zenodo.kind is ArtifactKind.MODEL_CARD
    assert zenodo.source_record_id == "model:kamo-naoyuki/wsj"
    assert zenodo.identifiers == (Identifier("espnet:model", "kamo-naoyuki/wsj"),)
    assert zenodo.releases[0].metadata["task"] == "asr"
    assert zenodo.releases[0].metadata["validated_by_espnet"] is True
    assert ("https://zenodo.org/record/2/files/wsj.zip?download=1", "model_artifact", False) in {
        (link.url, link.relation, link.crawl) for link in zenodo.links
    }

    assert hub.source_record_id == "model:example/aidatatang"
    assert hub.releases[0].metadata["validated_by_espnet"] is False
    assert hub.releases[0].metadata["declared_url"] == "huggingface.co"
    assert ("https://huggingface.co/example/aidatatang", "linked_model_artifact", False) in {
        (link.url, link.relation, link.crawl) for link in hub.links
    }


def test_espnet_model_zoo_skips_table_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(_response({"sha": _REVISION}), _response(_TABLE))
    adapter = EspnetModelZooSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_espnet_model_zoo_rejects_duplicate_model_names() -> None:
    duplicate = "\n".join(
        (
            _HEADER,
            "wsj,asr,kamo-naoyuki/wsj,https://zenodo.org/record/2/files/a.zip,16000,en,,,,,true",
            "wsj,asr,kamo-naoyuki/wsj,https://zenodo.org/record/3/files/b.zip,16000,en,,,,,true",
        )
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(duplicate))

    with pytest.raises(ValueError, match="duplicate Model Zoo model name"):
        EspnetModelZooSourceAdapter(client=client).fetch_page({})
