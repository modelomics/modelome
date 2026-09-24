from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.catalog import create_source
from modelome.sources.json_catalog import JsonCatalogSourceAdapter

SECRET = "catalog-secret-value"


class QueuedClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.responses = [
            HttpResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=json.dumps(payload).encode(),
                url="https://api.example.test/v1/models",
            )
            for payload in payloads
        ]
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected catalog request")
        return self.responses.pop(0)


def _config() -> dict[str, Any]:
    return {
        "name": "provider-fixture",
        "adapter": "json_catalog",
        "url": "https://api.example.test/v1/models",
        "provider_namespace": "fixture:model",
        "model_card_url_template": "https://cards.example.test/models/{id}",
        "provider_id_is_release": True,
        "page_size": 2,
        "page_size_param": "limit",
        "cursor_param": "after",
        "auth_env": "FIXTURE_CATALOG_TOKEN",
        "auth_header": "Authorization",
        "auth_scheme": "Bearer",
        "mapping": {
            "items_path": "$.data.models",
            "id_path": "identity.id",
            "name_path": "display.name",
            "created_path": "timestamps.created",
            "updated_path": "timestamps.updated",
            "base_model_path": "lineage.parents",
            "next_cursor_path": "paging.next_cursor",
            "total_path": "paging.total",
        },
    }


def _page_one() -> dict[str, Any]:
    return {
        "data": {
            "models": [
                {
                    "identity": {"id": "team/model-one"},
                    "display": {"name": "First Fixture Model"},
                    "timestamps": {"created": 1700000000, "updated": "2026-08-31"},
                    "lineage": {"parents": ["team/base-model"]},
                    "echoed_request": f"Bearer {SECRET}",
                },
                {
                    "identity": {"id": "team/model-two"},
                    "display": {},
                    "timestamps": {"created": "2026-08-30"},
                    "lineage": {"parents": []},
                },
            ]
        },
        "paging": {"next_cursor": "opaque-page-two", "total": 3},
    }


def _page_two() -> dict[str, Any]:
    return {
        "data": {
            "models": [
                {
                    "identity": {"id": "team/model-three"},
                    "display": {"name": "Third Fixture Model"},
                    "timestamps": {"created": "2026-09-01"},
                }
            ]
        },
        "paging": {"next_cursor": None, "total": 3},
    }


def test_cursor_catalog_is_declarative_resumable_and_secret_safe() -> None:
    client = QueuedClient(_page_one(), _page_two(), _page_one())
    adapter = create_source(
        _config(),
        client=client,
        environ={"FIXTURE_CATALOG_TOKEN": SECRET},
    )
    assert isinstance(adapter, JsonCatalogSourceAdapter)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    replay = adapter.fetch_page({})

    assert first.complete is False
    assert first.next_state == {
        "cursor": "opaque-page-two",
        "raw_items_seen": 2,
        "scan_total": 3,
    }
    assert first.upstream_count == 3
    assert second.complete is True
    assert second.next_state == {}
    assert first.records[0].links[0].crawl is True
    assert [record.source_record_id for record in replay.records] == [
        record.source_record_id for record in first.records
    ]
    assert replay.next_state == first.next_state

    assert client.calls[0][1] == {"limit": 2}
    assert client.calls[1][1] == {"limit": 2, "after": "opaque-page-two"}
    assert all(
        call[2]["Authorization"] == f"Bearer {SECRET}" for call in client.calls
    )

    record = first.records[0]
    assert record.kind is ArtifactKind.PROVIDER_PAGE
    assert record.canonical_url == "https://cards.example.test/models/team%2Fmodel-one"
    assert record.published_at == "1700000000"
    assert record.modified_at == "2026-08-31"
    assert record.identifiers == (Identifier("fixture:model", "team/model-one"),)
    assert record.models[0].status is ModelStatus.RELEASED
    assert record.models[0].identifiers == record.identifiers
    assert record.model_relations[0].predicate == "base_model"
    assert record.model_relations[0].target.identifiers == (
        Identifier("fixture:model", "team/base-model"),
    )
    assert len(record.releases) == 1
    assert record.releases[0].model_local_id == record.models[0].local_id
    assert record.releases[0].version == "team/model-one"
    assert record.releases[0].identifiers == record.identifiers
    assert first.records[1].title == "team/model-two"

    serialized_evidence = json.dumps(
        {
            "state": first.next_state,
            "raw": [item.raw for item in first.records],
        }
    )
    assert SECRET not in serialized_evidence
    assert "[REDACTED]" in serialized_evidence


def test_catalog_status_is_configurable_for_documented_knowledge_graph_entries() -> None:
    config = _config()
    config.pop("provider_id_is_release")
    config["model_status"] = "documented"
    client = QueuedClient(_page_one())

    page = create_source(
        config,
        client=client,
        environ={"FIXTURE_CATALOG_TOKEN": SECRET},
    ).fetch_page({})

    assert page.records[0].models[0].status is ModelStatus.DOCUMENTED
    assert page.records[0].releases == ()


def test_category_catalog_uses_provider_cursor_and_documented_status() -> None:
    config = {
        "name": "category-fixture",
        "adapter": "json_catalog",
        "url": "https://example.test/w/api.php?action=query&list=categorymembers",
        "provider_namespace": "example:page",
        "model_card_url_template": "https://example.test/wiki/{name}",
        "model_status": "documented",
        "page_size": 500,
        "page_size_param": "cmlimit",
        "cursor_param": "cmcontinue",
        "mapping": {
            "items_path": "query.categorymembers",
            "id_path": "pageid",
            "name_path": "title",
            "next_cursor_path": "continue.cmcontinue",
        },
    }
    payload = {
        "continue": {"cmcontinue": "opaque-next-page"},
        "query": {"categorymembers": [{"pageid": 7, "title": "An architecture"}]},
    }
    client = QueuedClient(payload)
    page = create_source(config, client=client).fetch_page({})

    assert page.complete is False
    assert page.next_state == {"cursor": "opaque-next-page", "raw_items_seen": 1}
    assert client.calls[0][1] == {"cmlimit": 500}
    assert page.records[0].models[0].status is ModelStatus.DOCUMENTED
    assert page.records[0].canonical_url == "https://example.test/wiki/An%20architecture"


def test_server_next_url_pagination_is_resolved_and_resumed() -> None:
    first_payload = {
        "results": [{"id": "model-a", "name": "Model A"}],
        "links": {"next": "/v1/models?page=2"},
    }
    second_payload = {
        "results": [{"id": "model-b", "name": "Model B"}],
        "links": {"next": None},
    }
    client = QueuedClient(first_payload, second_payload)
    adapter = JsonCatalogSourceAdapter(
        name="url-pages",
        url="https://api.example.test/v1/models",
        provider_namespace="url-pages:model",
        model_card_url_template="https://cards.example.test/{id}",
        mapping={
            "items_path": "results",
            "id_path": "id",
            "name_path": "name",
            "next_url_path": "links.next",
        },
        client=client,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.next_state == {
        "next_url": "https://api.example.test/v1/models?page=2",
        "raw_items_seen": 1,
    }
    assert first.complete is False
    assert second.complete is True
    assert client.calls[1][0] == "https://api.example.test/v1/models?page=2"
    assert client.calls[1][1] == {}


def test_cursor_condition_and_static_version_header_follow_provider_contract() -> None:
    first_payload = {
        "data": [{"id": "claude-example-1", "display_name": "Claude Example"}],
        "has_more": True,
        "last_id": "claude-example-1",
    }
    second_payload = {
        "data": [{"id": "claude-example-0", "display_name": "Claude Older"}],
        "has_more": False,
        "last_id": "claude-example-0",
    }
    client = QueuedClient(first_payload, second_payload)
    adapter = JsonCatalogSourceAdapter(
        name="conditional-pages",
        url="https://api.example.test/v1/models",
        provider_namespace="conditional-pages:model",
        model_card_url_template="https://api.example.test/v1/models/{id}",
        mapping={
            "items_path": "data",
            "id_path": "id",
            "name_path": "display_name",
            "next_cursor_path": "last_id",
            "next_cursor_condition_path": "has_more",
        },
        cursor_param="after_id",
        page_size=1_000,
        page_size_param="limit",
        auth_header_name="x-api-key",
        auth_token=SECRET,
        auth_scheme="",
        static_headers={"anthropic-version": "2023-06-01"},
        model_page_crawl=False,
        client=client,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.next_state == {"cursor": "claude-example-1", "raw_items_seen": 1}
    assert second.complete is True
    assert second.next_state == {}
    assert first.records[0].links[0].crawl is False
    assert client.calls == [
        (
            "https://api.example.test/v1/models",
            {"limit": 1_000},
            {
                "Accept": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": SECRET,
            },
        ),
        (
            "https://api.example.test/v1/models",
            {"limit": 1_000, "after_id": "claude-example-1"},
            {
                "Accept": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": SECRET,
            },
        ),
    ]


def test_query_key_authentication_and_path_identifier_keep_secrets_out_of_evidence() -> None:
    client = QueuedClient(
        {
            "models": [
                {
                    "name": "models/gemini-example-001",
                    "displayName": "Gemini Example",
                    "baseModelId": "gemini-example",
                }
            ],
            "nextPageToken": "opaque-page-two",
        },
        {
            "models": [{"name": "models/gemini-example-002"}],
        },
    )
    adapter = JsonCatalogSourceAdapter(
        name="query-key-pages",
        url="https://api.example.test/v1beta/models",
        provider_namespace="query-key-pages:model",
        model_card_url_template="https://api.example.test/v1beta/{id_path}",
        mapping={
            "items_path": "models",
            "id_path": "name",
            "name_path": "displayName",
            "base_model_path": "baseModelId",
            "next_cursor_path": "nextPageToken",
        },
        cursor_param="pageToken",
        page_size=1_000,
        page_size_param="pageSize",
        auth_query_param="key",
        auth_token=SECRET,
        model_page_crawl=False,
        client=client,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.records[0].canonical_url == (
        "https://api.example.test/v1beta/models/gemini-example-001"
    )
    assert first.records[0].links[0].crawl is False
    assert first.records[0].model_relations[0].target.identifiers == (
        Identifier("query-key-pages:model", "gemini-example"),
    )
    assert first.next_state == {
        "cursor": "opaque-page-two",
        "raw_items_seen": 1,
    }
    assert second.complete is True
    assert client.calls[0][1] == {"pageSize": 1_000, "key": SECRET}
    assert client.calls[1][1] == {
        "pageSize": 1_000,
        "pageToken": "opaque-page-two",
        "key": SECRET,
    }
    serialized = json.dumps(
        {"state": first.next_state, "records": [record.raw for record in first.records]}
    )
    assert SECRET not in serialized


def test_catalog_response_cap_is_enforced_by_the_default_http_client() -> None:
    adapter = JsonCatalogSourceAdapter(
        name="response-cap",
        url="https://api.example.test/v1/models",
        provider_namespace="response-cap:model",
        model_card_url_template="https://api.example.test/v1/models/{id}",
        mapping={"items_path": "data", "id_path": "id", "name_path": "name"},
        max_response_bytes=1_024,
    )

    assert isinstance(adapter.client, HttpClient)
    assert adapter.client.max_response_bytes == 1_024


def test_cursor_condition_requires_cursor_and_static_headers_cannot_carry_keys() -> None:
    adapter = JsonCatalogSourceAdapter(
        name="conditional-errors",
        url="https://api.example.test/v1/models",
        provider_namespace="conditional-errors:model",
        model_card_url_template="https://api.example.test/v1/models/{id}",
        mapping={
            "items_path": "data",
            "id_path": "id",
            "name_path": "name",
            "next_cursor_path": "last_id",
            "next_cursor_condition_path": "has_more",
        },
        client=QueuedClient(
            {"data": [{"id": "model-a", "name": "Model A"}], "has_more": True}
        ),
    )

    with pytest.raises(ValueError, match="without a cursor"):
        adapter.fetch_page({})
    with pytest.raises(ValueError, match="must not carry credentials"):
        JsonCatalogSourceAdapter(
            name="unsafe-static-header",
            url="https://api.example.test/v1/models",
            provider_namespace="unsafe-static-header:model",
            model_card_url_template="https://api.example.test/v1/models/{id}",
            mapping={"items_path": "data", "id_path": "id", "name_path": "name"},
            static_headers={"x-api-key": SECRET},
        )
    with pytest.raises(ValueError, match="either a header or query parameter"):
        JsonCatalogSourceAdapter(
            name="ambiguous-auth",
            url="https://api.example.test/v1/models",
            provider_namespace="ambiguous-auth:model",
            model_card_url_template="https://api.example.test/v1/models/{id}",
            mapping={"items_path": "data", "id_path": "id", "name_path": "name"},
            auth_header_name="Authorization",
            auth_query_param="key",
            auth_token=SECRET,
        )


def test_catalog_rejects_truncated_scan_before_known_total() -> None:
    client = QueuedClient(
        {
            "items": [{"id": "model-a", "name": "Model A"}],
            "total": 2,
        }
    )
    adapter = JsonCatalogSourceAdapter(
        name="truncated",
        url="https://api.example.test/v1/models",
        provider_namespace="truncated:model",
        model_card_url_template="https://cards.example.test/{id}",
        mapping={
            "items_path": "items",
            "id_path": "id",
            "name_path": "name",
            "total_path": "total",
            "next_cursor_path": "next_cursor",
        },
        client=client,
    )

    with pytest.raises(ValueError, match="before the known total of 2"):
        adapter.fetch_page({})


@pytest.mark.parametrize(
    "next_url",
    [
        "http://api.example.test/v1/models?page=2",
        "https://other.example.test/v1/models?page=2",
        "https://api.example.test:444/v1/models?page=2",
    ],
)
def test_server_next_url_cannot_move_authentication_to_another_origin(
    next_url: str,
) -> None:
    client = QueuedClient(
        {
            "results": [{"id": "safe-id", "name": "Safe Name"}],
            "links": {"next": next_url},
        }
    )
    adapter = JsonCatalogSourceAdapter(
        name="origin-safety",
        url="https://api.example.test/v1/models",
        provider_namespace="origin-safety:model",
        model_card_url_template="https://cards.example.test/{id}",
        mapping={
            "items_path": "results",
            "id_path": "id",
            "name_path": "name",
            "next_url_path": "links.next",
        },
        client=client,
        auth_header_name="Authorization",
        auth_token=SECRET,
    )

    with pytest.raises(ValueError, match="catalog origin") as error:
        adapter.fetch_page({})

    assert SECRET not in str(error.value)
    assert len(client.calls) == 1


def test_catalog_factory_preserves_an_explicit_empty_auth_scheme() -> None:
    config = _config()
    config["auth_scheme"] = ""
    client = QueuedClient(_page_two())
    adapter = create_source(
        config,
        client=client,
        environ={"FIXTURE_CATALOG_TOKEN": SECRET},
    )

    adapter.fetch_page(
        {
            "cursor": "opaque-page-two",
            "raw_items_seen": 2,
            "scan_total": 3,
        }
    )

    assert client.calls[0][2]["Authorization"] == SECRET


def test_sensitive_pagination_url_and_client_errors_do_not_leak_credentials() -> None:
    client = QueuedClient(
        {
            "results": [{"id": "safe-id", "name": "Safe Name"}],
            "next": f"https://api.example.test/v1/models?api_key={SECRET}",
        }
    )
    adapter = JsonCatalogSourceAdapter(
        name="secret-safety",
        url="https://api.example.test/v1/models",
        provider_namespace="secret-safety:model",
        model_card_url_template="https://cards.example.test/{id}",
        mapping={
            "items_path": "results",
            "id_path": "id",
            "name_path": "name",
            "next_url_path": "next",
        },
        client=client,
        auth_header_name="X-API-Key",
        auth_token=SECRET,
        auth_scheme="",
    )

    with pytest.raises(ValueError) as error:
        adapter.fetch_page({})

    assert SECRET not in str(error.value)
    assert "configured credentials" in str(error.value)


def test_configured_auth_environment_variable_must_be_set_without_echoing_values() -> None:
    with pytest.raises(ValueError, match="credential environment variable is unset") as error:
        create_source(_config(), client=QueuedClient(), environ={})

    assert SECRET not in str(error.value)


def test_malformed_item_identifier_cannot_leak_catalog_credential() -> None:
    adapter = JsonCatalogSourceAdapter(
        name="secret-item",
        url="https://api.example.test/v1/models",
        provider_namespace="secret-item:model",
        model_card_url_template="https://cards.example.test/{id}",
        mapping={"items_path": "results", "id_path": "id", "name_path": "name"},
        client=QueuedClient({"results": [{"id": SECRET, "name": "Invalid"}]}),
        auth_header_name="X-API-Key",
        auth_token=SECRET,
        auth_scheme="",
    )

    page = adapter.fetch_page({})

    assert len(page.issues) == 1
    assert SECRET not in page.issues[0].source_record_id
    assert page.issues[0].source_record_id == "[REDACTED]"
