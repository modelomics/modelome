from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.normalize import content_hash
from modelome.sources.datacite import DataCiteSourceAdapter

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
API_URL = "https://api.datacite.org/dois"


class QueuedClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/vnd.api+json"},
            body=json.dumps(self.payloads.pop(0)).encode(),
            url=url,
        )


def resource(doi: str = "10.5438/EXAMPLE") -> dict[str, Any]:
    return {
        "id": doi,
        "type": "dois",
        "attributes": {
            "doi": doi,
            "creators": [
                {
                    "givenName": "Ada",
                    "familyName": "Lovelace",
                    "name": "Lovelace, Ada",
                    "nameIdentifiers": [
                        {
                            "nameIdentifierScheme": "ORCID",
                            "nameIdentifier": "https://orcid.org/0000-0001-0000-0001",
                        }
                    ],
                }
            ],
            "titles": [
                {"title": "A <b>universal</b> scientific neural artifact"},
                {"title": "Alternate title", "titleType": "AlternativeTitle"},
            ],
            "descriptions": [
                {
                    "descriptionType": "Abstract",
                    "description": "We introduce <i>BioMatter-X</i> across fields.",
                }
            ],
            "types": {
                "resourceTypeGeneral": "Software",
                "resourceType": "Research software and trained weights",
                "schemaOrg": "SoftwareSourceCode",
            },
            "dates": [{"date": "2020-01-02", "dateType": "Issued"}],
            "publicationYear": 2020,
            "updated": "2026-09-01T08:09:10Z",
            "url": "https://repository.example/items/one?utm_source=tracker",
            "contentUrl": [
                "https://repository.example/files/model.bin#download",
                "javascript:alert(1)",
            ],
            "relatedIdentifiers": [
                {
                    "relatedIdentifier": "10.5555/RELATED",
                    "relatedIdentifierType": "DOI",
                    "relationType": "IsSupplementTo",
                },
                {
                    "relatedIdentifier": "https://github.com/example/project",
                    "relatedIdentifierType": "URL",
                    "relationType": "IsDocumentedBy",
                },
            ],
            "state": "findable",
            "isActive": True,
        },
        "relationships": {"client": {"data": {"id": "example.repository", "type": "clients"}}},
    }


def next_url(
    token: str,
    *,
    query: str = "updated:[2026-08-26 TO 2026-09-01]",
    page_size: int = 1,
    endpoint: str = API_URL,
    extras: Mapping[str, str] | None = None,
) -> str:
    params: list[tuple[str, str]] = [
        ("affiliation", "true"),
        ("page[cursor]", token),
        ("page[size]", str(page_size)),
        ("publisher", "true"),
        ("query", query),
    ]
    params.extend((extras or {}).items())
    return f"{endpoint}?{urlencode(params)}"


def response(
    resources: list[Any],
    *,
    total: int | Any | None = None,
    next_link: Any = None,
) -> dict[str, Any]:
    return {
        "data": resources,
        "meta": {"total": len(resources) if total is None else total},
        "links": {"self": API_URL, "next": next_link},
    }


def adapter(client: QueuedClient, **kwargs: Any) -> DataCiteSourceAdapter:
    return DataCiteSourceAdapter(client=client, clock=lambda: NOW, **kwargs)


def continuation_state(
    cursor: str = "opaque-token",
    *,
    raw_items_seen: int = 1,
    scan_total: int = 3,
) -> dict[str, Any]:
    return {
        "cursor": cursor,
        "window_start": "2026-08-31",
        "window_end": "2026-09-01",
        "raw_items_seen": raw_items_seen,
        "scan_total": scan_total,
        "seen_cursor_hashes": [content_hash(cursor)],
        "started_at": "2026-09-02T00:00:00Z",
    }


def test_all_findable_dois_resume_with_reconstructed_canonical_requests() -> None:
    first_query = "updated:[2026-08-26 TO 2026-09-01]"
    client = QueuedClient(
        response(
            [resource()],
            total=2,
            next_link=next_url("opaque+/=,token"),
        ),
        response(
            [resource("10.5281/ZENODO.123")],
            total=2,
            # An unused terminal link must not create a false cursor fault.
            next_link="https://attacker.example/unused",
        ),
    )
    source = adapter(client, page_size=1)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    expected_params = {
        "query": first_query,
        "page[cursor]": "1",
        "page[size]": 1,
        "affiliation": "true",
        "publisher": "true",
    }
    assert client.calls[0] == (
        API_URL,
        expected_params,
        {"Accept": "application/vnd.api+json"},
    )
    assert client.calls[1][0] == API_URL
    assert client.calls[1][1] == {
        **expected_params,
        "page[cursor]": "opaque+/=,token",
    }
    assert first.complete is False
    assert first.upstream_count == 2
    assert first.next_state["cursor"] == "opaque+/=,token"
    assert first.next_state["raw_items_seen"] == 1
    assert first.next_state["scan_total"] == 2
    assert first.next_state["seen_cursor_hashes"] == [content_hash("opaque+/=,token")]
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-01"

    record = first.records[0]
    assert record.kind is ArtifactKind.CODE_REPOSITORY
    assert record.source_record_id == "10.5438/example"
    assert record.canonical_url == "https://repository.example/items/one"
    assert record.title == "A universal scientific neural artifact"
    assert record.published_at == "2020-01-02"
    assert record.modified_at == "2026-09-01T08:09:10Z"
    assert record.identifiers == (Identifier("doi", "10.5438/example"),)
    assert "We introduce BioMatter-X across fields." in record.text
    assert "Creators: Lovelace, Ada" in record.text
    assert "DataCite resource type: Software; Research software" in record.text
    assert record.raw["attributes"]["types"]["resourceTypeGeneral"] == "Software"
    relations = {(link.relation, link.url) for link in record.links}
    assert ("doi", "https://doi.org/10.5438/example") in relations
    assert (
        "content",
        "https://repository.example/files/model.bin",
    ) in relations
    assert ("is_supplement_to", "https://doi.org/10.5555/related") in relations
    assert (
        "is_documented_by",
        "https://github.com/example/project",
    ) in relations
    assert Identifier("doi", "10.5555/related") not in record.identifiers


def test_daily_overlap_and_fixed_history_use_only_an_updated_date_query() -> None:
    daily_client = QueuedClient(response([], total=0))
    fixed_client = QueuedClient(response([], total=0))

    daily = adapter(daily_client).fetch_page({"watermark": "2026-08-31"})
    fixed = adapter(fixed_client).fetch_page(
        {"window_start": "1950-01-01", "window_end": "1950-12-31"}
    )

    assert daily.complete is True
    assert daily_client.calls[0][1]["query"] == ("updated:[2026-08-30 TO 2026-09-01]")
    assert fixed.complete is True
    fixed_params = fixed_client.calls[0][1]
    assert fixed_params["query"] == "updated:[1950-01-01 TO 1950-12-31]"
    assert set(fixed_params) == {
        "query",
        "page[cursor]",
        "page[size]",
        "affiliation",
        "publisher",
    }


@pytest.mark.parametrize(
    ("resource_type", "expected"),
    [
        ("JournalArticle", ArtifactKind.PAPER),
        ("Preprint", ArtifactKind.PAPER),
        ("ComputationalNotebook", ArtifactKind.CODE_REPOSITORY),
        ("Model", ArtifactKind.MODEL_CARD),
        ("Dataset", ArtifactKind.OTHER),
    ],
)
def test_controlled_resource_type_drives_artifact_kind(
    resource_type: str,
    expected: ArtifactKind,
) -> None:
    item = resource()
    item["attributes"]["types"]["resourceTypeGeneral"] = resource_type

    record = adapter(QueuedClient(response([item], total=1))).fetch_page({}).records[0]

    assert record.kind is expected


def test_explicit_artifact_kind_override_is_supported() -> None:
    record = adapter(
        QueuedClient(response([resource()], total=1)),
        artifact_kind="provider_page",
    ).fetch_page({}).records[0]

    assert record.kind is ArtifactKind.PROVIDER_PAGE


def test_malformed_resource_is_quarantined_without_advancing() -> None:
    item = resource()
    item["attributes"] = "not-an-object"
    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert page.issues[0].stage == "source_normalize"
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor"] == "1"
    assert page.retry_state["raw_items_seen"] == 0


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda item: item.update(id="not-a-doi"), "valid DOI"),
        (
            lambda item: item["attributes"].update(doi="10.9999/conflict"),
            "conflicts",
        ),
        (
            lambda item: item["attributes"].update(state="registered"),
            "Findable",
        ),
        (
            lambda item: item["attributes"].update(isActive=False),
            "inactive",
        ),
    ],
)
def test_identity_and_findable_state_fail_closed(mutate, error) -> None:
    item = resource()
    mutate(item)

    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    assert page.complete is False
    assert page.records == ()
    assert error in page.issues[0].error


def test_optional_bad_urls_and_related_ids_do_not_discard_primary_record() -> None:
    item = resource()
    item["attributes"]["url"] = "https://user:password@example.test/private"
    item["attributes"]["contentUrl"] = ["file:///private/data", 42]
    item["attributes"]["relatedIdentifiers"] = [
        {
            "relatedIdentifier": "not a doi",
            "relatedIdentifierType": "DOI",
            "relationType": "References",
        },
        {
            "relatedIdentifier": "javascript:alert(1)",
            "relatedIdentifierType": "URL",
            "relationType": "IsDocumentedBy",
        },
    ]

    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    assert page.complete is True
    assert page.issues == ()
    record = page.records[0]
    assert record.canonical_url == "https://doi.org/10.5438/example"
    assert record.links == (next(link for link in record.links if link.relation == "doi"),)


def test_protocol_related_identifier_categories_become_relation_links_only() -> None:
    item = resource()
    item["attributes"]["relatedIdentifiers"] = [
        {
            "relatedIdentifier": "12345678",
            "relatedIdentifierType": "PMID",
            "relationType": "References",
        },
        {
            "relatedIdentifier": "arXiv:1706.03762v5",
            "relatedIdentifierType": "arXiv",
            "relationType": "IsVersionOf",
        },
        {
            "relatedIdentifier": "20.500.123/abc",
            "relatedIdentifierType": "Handle",
            "relationType": "HasPart",
        },
        {
            "relatedIdentifier": "2020ApJ...000..001A",
            "relatedIdentifierType": "bibcode",
            "relationType": "IsCitedBy",
        },
        {
            "relatedIdentifier": "urn:nbn:de:example",
            "relatedIdentifierType": "URN",
            "relationType": "HasMetadata",
        },
    ]

    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    record = page.records[0]
    assert record.identifiers == (Identifier("doi", "10.5438/example"),)
    relations = {(link.relation, link.url) for link in record.links}
    assert ("references", "https://pubmed.ncbi.nlm.nih.gov/12345678") in relations
    assert ("is_version_of", "https://arxiv.org/abs/1706.03762v5") in relations
    assert ("has_part", "https://hdl.handle.net/20.500.123/abc") in relations
    assert (
        "is_cited_by",
        "https://ui.adsabs.harvard.edu/abs/2020ApJ...000..001A/abstract",
    ) in relations
    assert ("has_metadata", "https://n2t.net/urn:nbn:de:example") in relations


def test_related_item_software_url_is_preserved_as_an_exact_relation_link() -> None:
    item = resource()
    item["attributes"]["types"] = {
        "resourceTypeGeneral": "JournalArticle",
        "resourceType": "Research article",
    }
    item["attributes"]["relatedItems"] = [
        {
            "relatedItemType": "Software",
            "relatedItemIdentifier": (
                "https://zenodo.org/records/123/files/model.safetensors?download=1"
            ),
            "relatedItemIdentifierType": "URL",
            "relationType": "IsSupplementedBy",
            "title": "Trained model weights",
        },
        {
            "relatedItemType": "Model",
            "relatedItemIdentifier": "javascript:alert(1)",
            "relatedItemIdentifierType": "URL",
            "relationType": "HasPart",
        },
    ]

    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    assert page.complete is True
    assert len(page.records) == 1
    record = page.records[0]
    artifact_link = next(
        link
        for link in record.links
        if link.url.startswith("https://zenodo.org/records/123/files/")
    )
    assert artifact_link.url == (
        "https://zenodo.org/records/123/files/model.safetensors?download=1"
    )
    assert artifact_link.relation == "is_supplemented_by"
    assert artifact_link.locator == "$.attributes.relatedItems[0].relatedItemIdentifier"
    assert not any(link.url.startswith("javascript:") for link in record.links)
    assert record.models == ()


def test_datacite_bridges_only_related_dois_marked_identical() -> None:
    item = resource()
    item["attributes"]["relatedIdentifiers"] = [
        {
            "relatedIdentifier": "https://doi.org/10.5555/IDENTICAL",
            "relatedIdentifierType": "DOI",
            "relationType": "IsIdenticalTo",
        },
        {
            "relatedIdentifier": "10.5555/supplement",
            "relatedIdentifierType": "DOI",
            "relationType": "IsSupplementTo",
        },
    ]

    page = adapter(QueuedClient(response([item], total=1))).fetch_page({})

    assert page.records[0].identifiers == (
        Identifier("doi", "10.5438/example"),
        Identifier("doi", "10.5555/identical"),
    )


def test_short_or_missing_continuation_restarts_the_frozen_window() -> None:
    client = QueuedClient(response([resource()], total=2, next_link=None))

    page = adapter(client, page_size=2).fetch_page({})

    assert page.complete is False
    assert page.issues[0].stage == "source_pagination"
    assert "before the declared total of 2" in page.issues[0].error
    assert page.next_state == page.retry_state
    assert page.retry_state["cursor"] == "1"
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_total_drift_restarts_the_same_frozen_window() -> None:
    state = continuation_state(scan_total=2)
    page = adapter(QueuedClient(response([resource()], total=3))).fetch_page(state)

    assert page.complete is False
    assert any("meta.total changed" in issue.error for issue in page.issues)
    assert page.retry_state["cursor"] == "1"
    assert page.retry_state["window_start"] == state["window_start"]
    assert page.retry_state["window_end"] == state["window_end"]
    assert page.retry_state["raw_items_seen"] == 0
    assert "scan_total" not in page.retry_state


def test_cursor_cycle_restarts_the_same_frozen_window() -> None:
    state = continuation_state()
    query = "updated:[2026-08-31 TO 2026-09-01]"
    page = adapter(
        QueuedClient(
            response(
                [resource()],
                total=3,
                next_link=next_url("opaque-token", query=query),
            )
        ),
        page_size=1,
    ).fetch_page(state)

    assert any("cursor cycle" in issue.error for issue in page.issues)
    assert page.retry_state["cursor"] == "1"


@pytest.mark.parametrize(
    "link",
    [
        next_url("next", endpoint="https://attacker.example/dois"),
        next_url("next", endpoint="https://api.datacite.org/clients"),
        next_url("next", query="updated:[1900-01-01 TO 1900-01-01]"),
        next_url("next", extras={"resource-type": "Software"}),
        (
            f"{API_URL}?affiliation=true&page%5Bcursor%5D=one&"
            "page%5Bcursor%5D=two&page%5Bsize%5D=1&publisher=true&"
            "query=updated%3A%5B2026-08-26+TO+2026-09-01%5D"
        ),
        (
            f"{API_URL}?page%5Bcursor%5D=next&page%5Bsize%5D=1&"
            "query=updated%3A%5B2026-08-26+TO+2026-09-01%5D"
        ),
        f"{API_URL}?page%5Bcursor%ZZ=next",
        f" {API_URL}?page%5Bcursor%5D=next",
        7,
    ],
)
def test_hostile_or_drifted_next_link_is_quarantined(link) -> None:
    client = QueuedClient(response([resource()], total=2, next_link=link))

    page = adapter(client, page_size=1).fetch_page({})

    assert page.complete is False
    assert page.records[0].source_record_id == "10.5438/example"
    assert page.issues[0].stage == "source_pagination"
    assert "invalid links.next" in page.issues[0].error
    assert page.retry_state["cursor"] == "1"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "payload,error",
    [
        ([], "JSON object"),
        ({"data": {}, "meta": {"total": 0}, "links": {}}, "data must be an array"),
        ({"data": [], "meta": [], "links": {}}, "meta must be a JSON object"),
        ({"data": [], "meta": {"total": "0"}, "links": {}}, "valid meta.total"),
        ({"data": [], "meta": {"total": 0}, "links": []}, "links must be"),
    ],
)
def test_malformed_payload_fails_without_checkpoint_progress(payload, error) -> None:
    client = QueuedClient(payload)

    with pytest.raises(ValueError, match=error):
        adapter(client).fetch_page({})

    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "state,error",
    [
        ({"window_start": "2026-01-01"}, "both boundaries"),
        (
            {"window_start": "2026-09-01", "window_end": "2026-09-02"},
            "closed UTC day",
        ),
        ({"cursor": "orphan"}, "missing its frozen window"),
        (
            {
                "cursor": "page-two",
                "window_start": "2026-08-31",
                "window_end": "2026-09-01",
            },
            "requires raw_items_seen and scan_total",
        ),
        (
            {
                **continuation_state(),
                "seen_cursor_hashes": ["not-a-hash"],
            },
            "seen_cursor_hashes",
        ),
        (
            {
                **continuation_state(),
                "seen_cursor_hashes": [content_hash("some-other-token")],
            },
            "absent from seen_cursor_hashes",
        ),
        ({**continuation_state(), "cursor": " leading-space"}, "malformed"),
        (
            {
                "cursor": "1",
                "window_start": "2026-08-31",
                "window_end": "2026-09-01",
                "raw_items_seen": 0,
                "scan_total": 3,
            },
            "first-page checkpoint has pagination counts",
        ),
        ({**continuation_state(), "started_at": "yesterday"}, "started_at"),
    ],
)
def test_invalid_checkpoint_fails_before_http(state, error) -> None:
    client = QueuedClient()

    with pytest.raises(ValueError, match=error):
        adapter(client).fetch_page(state)

    assert client.calls == []


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"url": "file:///private/datacite.json"}, r"HTTP\(S\)"),
        ({"url": "https://user:secret@api.datacite.org/dois"}, r"HTTP\(S\)"),
        ({"url": "https://api.datacite.org/dois?state=findable"}, "query"),
        ({"url": "https://api.datacite.org/dois#fragment"}, "fragment"),
        ({"page_size": 0}, "between 1 and 1000"),
        ({"page_size": 1001}, "between 1 and 1000"),
        ({"initial_lookback_days": 0}, "must be positive"),
        ({"overlap_days": 0}, "must be positive"),
    ],
)
def test_constructor_rejects_unsafe_or_invalid_configuration(kwargs, error) -> None:
    with pytest.raises(ValueError, match=error):
        DataCiteSourceAdapter(**kwargs)
