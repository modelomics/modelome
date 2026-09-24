from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.extract import IntroductionCueExtractor
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.zenodo import ZenodoModelRecordsSourceAdapter

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
URL = "https://zenodo.org/api/records"
QUERY = "resource_type.type:model"


class QueuedClient:
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


def response(
    items: list[Mapping[str, Any]],
    *,
    total: int,
    next_url: str | None = None,
) -> HttpResponse:
    payload: dict[str, Any] = {
        "hits": {"hits": items, "total": total},
        "links": {},
    }
    if next_url is not None:
        payload["links"]["next"] = next_url
    return HttpResponse(200, {}, json.dumps(payload).encode(), URL)


def item(
    record_id: int,
    *,
    title: str,
    concept_id: int | None = None,
    resource_type: str = "model",
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": record_id,
        "doi_url": f"https://doi.org/10.5281/zenodo.{record_id}",
        "metadata": {
            "title": title,
            "description": "<p>We introduce OceanNet, a neural network model.</p>",
            "publication_date": "2024-01-02",
            "resource_type": {"type": resource_type, "title": "Model"},
            "related_identifiers": [
                {
                    "identifier": "10.48550/arXiv.2401.00001",
                    "resource_type": "publication",
                },
                {
                    "identifier": "https://github.com/example/oceannet",
                    "resource_type": "software",
                },
            ],
        },
        "created": "2024-01-02T03:04:05.000000+00:00",
        "modified": "2024-01-03T03:04:05.000000+00:00",
        "links": {
            "self": f"https://zenodo.org/api/records/{record_id}",
            "self_html": f"https://zenodo.org/records/{record_id}",
            "doi": f"https://doi.org/10.5281/zenodo.{record_id}",
            "parent_html": f"https://zenodo.org/records/{concept_id or record_id - 1}",
            "versions": f"https://zenodo.org/api/records/{concept_id or record_id - 1}/versions",
        },
        "files": [
            {
                "id": "file-1",
                "key": "oceannet.ckpt",
                "checksum": "md5:0123456789abcdef0123456789abcdef",
                "links": {
                    "self": (
                        f"https://zenodo.org/api/records/{record_id}/files/"
                        "oceannet.ckpt/content"
                    )
                },
            }
        ],
    }
    if concept_id is not None:
        result["conceptrecid"] = str(concept_id)
        result["conceptdoi"] = f"10.5281/zenodo.{concept_id}"
    return result


def test_zenodo_model_catalog_preserves_cross_domain_evidence_without_model_claims() -> None:
    next_url = (
        f"{URL}?page=2&q={QUERY}&size=2&sort=oldest"
    )
    client = QueuedClient(
        response(
            [item(101, title="OceanNet", concept_id=100), item(102, title="RiverNet")],
            total=3,
            next_url=next_url,
        ),
        response([item(103, title="CoastNet")], total=3),
    )
    adapter = ZenodoModelRecordsSourceAdapter(
        url=URL,
        page_size=2,
        client=client,
        clock=lambda: NOW,
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count is None
    assert first.next_state == {
        "next_url": f"{URL}?page=2&q=resource_type.type%3Amodel&size=2&sort=oldest",
        "page_number": 2,
        "raw_items_seen": 2,
        "scan_total": 3,
        "started_at": "2026-09-21T12:00:00Z",
    }
    assert second.complete is True
    assert second.authoritative_snapshot is False
    assert second.upstream_count == 3
    assert second.next_state == {
        "completed_at": "2026-09-21T12:00:00Z",
        "observed_record_count": 3,
        "provider_total": 3,
        "query": QUERY,
        "sort": "oldest",
    }
    assert [call[1] for call in client.calls] == [
        {"q": QUERY, "sort": "oldest", "page": 1, "size": 2},
        {},
    ]

    record = first.records[0]
    assert record.source_record_id == "record:101"
    assert record.kind is ArtifactKind.CATALOG_RECORD
    assert record.models == ()
    assert record.identifiers == (
        Identifier("zenodo:record", "101"),
        Identifier("doi", "10.5281/zenodo.101"),
    )
    assert record.text == "OceanNet\n\nWe introduce OceanNet, a neural network model."
    assert [hint.name for hint in IntroductionCueExtractor().extract(record)] == ["OceanNet"]
    links = {(link.url, link.relation, link.crawl) for link in record.links}
    assert (
        "https://zenodo.org/records/101",
        "catalog_page",
        True,
    ) in links
    assert (
        "https://doi.org/10.48550/arxiv.2401.00001",
        "related_publication",
        True,
    ) in links
    assert (
        "https://github.com/example/oceannet",
        "related_code",
        True,
    ) in links
    assert (
        "https://zenodo.org/api/records/101/files/oceannet.ckpt/content",
        "artifact_file",
        False,
    ) in links


def test_zenodo_model_catalog_rejects_cross_scope_next_urls() -> None:
    client = QueuedClient(
        response(
            [item(101, title="OceanNet")],
            total=2,
            next_url=f"{URL}?page=2&q=all&size=1&sort=oldest",
        )
    )
    adapter = ZenodoModelRecordsSourceAdapter(url=URL, page_size=1, client=client)

    with pytest.raises(ValueError, match="changed the configured catalog scope"):
        adapter.fetch_page({})


def test_zenodo_anonymous_search_clamps_configured_page_size_and_paginates() -> None:
    next_url = f"{URL}?page=2&q=resource_type.type%3Amodel&size=25&sort=oldest"
    client = QueuedClient(
        response([item(101, title="OceanNet")], total=2, next_url=next_url),
        response([item(102, title="RiverNet")], total=2),
    )
    adapter = ZenodoModelRecordsSourceAdapter(url=URL, page_size=100, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls[0][1] == {
        "q": QUERY,
        "sort": "oldest",
        "page": 1,
        "size": 25,
    }
    assert first.next_state["next_url"] == next_url
    assert client.calls[1][0] == next_url
    assert second.complete is True


def test_zenodo_model_catalog_rejects_provider_total_drift_mid_scan() -> None:
    next_url = f"{URL}?page=2&q={QUERY}&size=1&sort=oldest"
    drifted_next_url = next_url.replace("page=2", "page=3")
    client = QueuedClient(
        response([item(101, title="OceanNet")], total=2, next_url=next_url),
        response([item(102, title="RiverNet")], total=3, next_url=drifted_next_url),
    )
    adapter = ZenodoModelRecordsSourceAdapter(url=URL, page_size=1, client=client)

    first = adapter.fetch_page({})

    with pytest.raises(ValueError, match="provider total changed"):
        adapter.fetch_page(first.next_state)


def test_zenodo_model_catalog_quarantines_a_non_model_result() -> None:
    client = QueuedClient(
        response([item(101, title="Marble Sculpture", resource_type="image")], total=1)
    )
    adapter = ZenodoModelRecordsSourceAdapter(url=URL, page_size=1, client=client)

    page = adapter.fetch_page({})

    assert page.records == ()
    assert len(page.issues) == 1
    assert "not typed as a Zenodo Model" in page.issues[0].error


@pytest.mark.parametrize(
    ("filename", "expected_relation"),
    [
        ("oceannet_checkpoint.safetensors", "checkpoint"),
        ("oceannet_weights.pth", "checkpoint"),
        ("oceannet.pt", None),
        ("weights.csv", None),
    ],
)
def test_zenodo_marks_only_explicit_model_weight_files_as_candidate_checkpoints(
    filename: str, expected_relation: str | None
) -> None:
    result = item(101, title="OceanNet")
    result["files"][0]["key"] = filename
    result["files"][0]["links"]["self"] = (
        f"https://zenodo.org/api/records/101/files/{filename}/content"
    )
    client = QueuedClient(response([result], total=1))
    adapter = ZenodoModelRecordsSourceAdapter(url=URL, page_size=1, client=client)

    page = adapter.fetch_page({})

    file_links = [link for link in page.records[0].links if link.locator == "$.files[0].key"]
    assert [link.relation for link in file_links] == (
        [expected_relation] if expected_relation else []
    )
    assert page.records[0].models == ()
