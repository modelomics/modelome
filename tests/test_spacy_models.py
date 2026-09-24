from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.spacy_models import SpacyModelsSourceAdapter


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


def _response(
    payload: Mapping[str, Any], *, status: int = 200, padding: int = 0
) -> HttpResponse:
    return HttpResponse(
        status,
        {},
        json.dumps(payload).encode() + (b"x" * padding),
        "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json",
    )


_PAYLOAD = {
    "spacy": {
        "3.8": {
            "de_core_news_sm": ["3.8.0"],
            "en_core_web_sm": ["3.8.0"],
        },
        "3.7": {
            "en_core_web_sm": ["3.7.1", "3.7.0"],
        },
    }
}


def test_spacy_models_enumerates_exact_pipeline_and_package_releases() -> None:
    adapter = SpacyModelsSourceAdapter(
        client=_QueuedClient(_response(_PAYLOAD)),
        clock=lambda: datetime(2026, 9, 22, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 4
    assert page.next_state["model_count"] == 2
    assert page.next_state["release_count"] == 4
    record = page.records[0]
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.source_record_id == "spacy-models:compatibility"
    assert record.canonical_url == (
        "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"
    )
    assert [(item.name, item.identifiers, item.status) for item in record.models] == [
        (
            "de_core_news_sm",
            (Identifier("spacy:model", "de_core_news_sm"),),
            ModelStatus.RELEASED,
        ),
        (
            "en_core_web_sm",
            (Identifier("spacy:model", "en_core_web_sm"),),
            ModelStatus.RELEASED,
        ),
    ]
    by_release = {item.identifiers[0].value: item for item in record.releases}
    assert by_release["en_core_web_sm@3.7.1"].metadata == {
        "spacy_compatibility": ["3.7"]
    }
    assert by_release["en_core_web_sm@3.8.0"].model_local_id == (
        "spacy:en_core_web_sm#model"
    )
    assert record.raw["compatibility"] == {
        "3.7": {"en_core_web_sm": ["3.7.0", "3.7.1"]},
        "3.8": {
            "de_core_news_sm": ["3.8.0"],
            "en_core_web_sm": ["3.8.0"],
        },
    }
    assert adapter.client.calls == [
        (
            record.canonical_url,
            {},
            {"Accept": "application/json"},
        )
    ]


def test_spacy_models_skips_rewriting_an_unchanged_manifest() -> None:
    first_client = _QueuedClient(_response(_PAYLOAD))
    adapter = SpacyModelsSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    adapter.client = _QueuedClient(_response(_PAYLOAD))
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.authoritative_snapshot is False
    assert second.records == ()
    assert second.next_state["completed_content_hash"] == first.next_state[
        "completed_content_hash"
    ]


def test_spacy_models_rejects_bad_manifest_and_response_bounds() -> None:
    invalid = {"spacy": {"3.8": {"en core web sm": ["3.8.0"]}}}
    adapter = SpacyModelsSourceAdapter(client=_QueuedClient(_response(invalid)))
    with pytest.raises(ValueError, match="invalid pipeline package"):
        adapter.fetch_page({})

    bounded = SpacyModelsSourceAdapter(
        max_response_bytes=8,
        client=_QueuedClient(_response(_PAYLOAD)),
    )
    with pytest.raises(ValueError, match="exceeds 8 bytes"):
        bounded.fetch_page({})

    failed = SpacyModelsSourceAdapter(
        client=_QueuedClient(_response({}, status=429)),
    )
    with pytest.raises(ValueError, match="HTTP 429"):
        failed.fetch_page({})
