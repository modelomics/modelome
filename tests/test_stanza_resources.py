from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.stanza_resources import StanzaResourcesSourceAdapter

_REVISION = "a" * 40


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


def _response(payload: Any, *, status: int = 200) -> HttpResponse:
    return HttpResponse(
        status,
        {},
        json.dumps(payload).encode(),
        "https://fixtures.test/stanza",
    )


def _tree(*, truncated: bool = False) -> Mapping[str, Any]:
    return {
        "truncated": truncated,
        "tree": [
            {"path": "README.md", "type": "blob"},
            {"path": "resources_1.10.0.json", "type": "blob"},
            {"path": "resources_1.2.0.json", "type": "blob"},
        ],
    }


_RESOURCE_1_2 = {
    "url": "https://models.example.test/stanza/1.2.0",
    "en": {
        "lang_name": "English",
        "tokenize": {"combined": {"md5": "1" * 32}},
        "pretrain": {"conll17": {"md5": "2" * 32}},
        "depparse": {
            "combined": {
                "md5": "3" * 32,
                "dependencies": [{"model": "pretrain", "package": "conll17"}],
            }
        },
        "default_processors": {"tokenize": "combined"},
        "packages": {"default": {"tokenize": "combined"}},
    },
    "english": {"alias": "en"},
}

_RESOURCE_1_10 = {
    "en": {
        "lang_name": "English",
        "tokenize": {"combined": {"md5": "4" * 32}},
        "pretrain": {"conll17": {"md5": "2" * 32}},
    }
}


def test_stanza_resources_enumerates_historical_component_releases_and_dependencies() -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_RESOURCE_1_2),
        _response(_RESOURCE_1_10),
    )
    adapter = StanzaResourcesSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 22, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 6
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["manifest_count"] == 2
    assert page.next_state["model_release_count"] == 6
    assert [record.title for record in page.records] == [
        "Stanza resources 1.2.0",
        "Stanza resources 1.10.0",
    ]
    first = page.records[0]
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.canonical_url == (
        f"https://raw.githubusercontent.com/stanfordnlp/stanza-resources/{_REVISION}/"
        "resources_1.2.0.json"
    )
    assert [(item.name, item.identifiers, item.status) for item in first.models] == [
        (
            "en/depparse/combined",
            (Identifier("stanza:model", "en/depparse/combined"),),
            ModelStatus.RELEASED,
        ),
        (
            "en/pretrain/conll17",
            (Identifier("stanza:model", "en/pretrain/conll17"),),
            ModelStatus.RELEASED,
        ),
        (
            "en/tokenize/combined",
            (Identifier("stanza:model", "en/tokenize/combined"),),
            ModelStatus.RELEASED,
        ),
        (
            "en/package/default",
            (Identifier("stanza:pipeline-package", "en/package/default"),),
            ModelStatus.RELEASED,
        ),
    ]
    release = next(
        item
        for item in first.releases
        if item.model_local_id == "stanza:en/depparse/combined#model"
    )
    assert release.revision == "1.2.0"
    assert release.identifiers == (
        Identifier(
            "stanza:resources-manifest-entry",
            "1.2.0:en/depparse/combined@" + ("3" * 32),
        ),
    )
    assert release.metadata == {
        "md5": "3" * 32,
        "resource_manifest": "resources_1.2.0.json",
    }
    assert first.model_relations[0].predicate == "depends_on"
    assert first.model_relations[0].target.identifiers == (
        Identifier("stanza:model", "en/pretrain/conll17"),
    )
    package_release = next(
        item for item in first.releases if item.model_local_id == "stanza:en/package/default#model"
    )
    assert package_release.identifiers == (
        Identifier("stanza:pipeline-package-version", "en/package/default@1.2.0"),
    )
    assert any(
        relation.subject_local_id == "stanza:en/package/default#model"
        and relation.target.identifiers
        == (Identifier("stanza:model", "en/tokenize/combined"),)
        for relation in first.model_relations
    )
    assert first.raw["manifest"] == _RESOURCE_1_2
    assert client.calls[1][0].endswith(f"/{_REVISION}?recursive=1")


def test_stanza_resources_skips_manifest_fetches_when_revision_is_unchanged() -> None:
    first_client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_tree()),
        _response(_RESOURCE_1_2),
        _response(_RESOURCE_1_10),
    )
    adapter = StanzaResourcesSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.authoritative_snapshot is False
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert len(second_client.calls) == 1


def test_stanza_resources_rejects_truncated_trees_and_invalid_checksums() -> None:
    truncated = StanzaResourcesSourceAdapter(
        client=_QueuedClient(_response({"sha": _REVISION}), _response(_tree(truncated=True)))
    )
    with pytest.raises(ValueError, match="tree is truncated"):
        truncated.fetch_page({})

    invalid = {
        "en": {"tokenize": {"combined": {"md5": "not-a-checksum"}}}
    }
    malformed = StanzaResourcesSourceAdapter(
        client=_QueuedClient(
            _response({"sha": _REVISION}),
            _response({"tree": [{"path": "resources_1.2.0.json", "type": "blob"}]}),
            _response(invalid),
        )
    )
    with pytest.raises(ValueError, match="invalid checksum"):
        malformed.fetch_page({})
