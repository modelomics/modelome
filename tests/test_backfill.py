from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.backfill import (
    ArxivBackfill,
    BioRxivBackfill,
    CrossrefBackfill,
    DataCiteBackfill,
    EuropePmcBackfill,
    OpenAlexBackfill,
    run_arxiv_backfill,
    run_biorxiv_backfill,
    run_crossref_backfill,
    run_datacite_backfill,
    run_europe_pmc_backfill,
    run_openalex_backfill,
    run_osf_preprints_backfill,
)
from modelome.http import HttpResponse
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.storage import Database

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


class FakeHttp:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        payload = self.payloads.pop(0)
        return HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload).encode(),
            url=url,
        )


class FakeXmlHttp:
    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.payloads:
            raise AssertionError(f"unexpected GET {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/xml"},
            body=self.payloads.pop(0).encode(),
            url=url,
        )


def openalex_page(
    work_id: str,
    publication_date: str,
    *,
    next_cursor: str | None,
    count: int = 1,
) -> dict[str, Any]:
    return {
        "meta": {"count": count, "next_cursor": next_cursor},
        "results": [
            {
                "id": f"https://openalex.org/{work_id}",
                "title": f"Historical work {work_id}",
                "publication_date": publication_date,
                "updated_date": f"{publication_date}T12:00:00Z",
                "abstract_inverted_index": None,
                "ids": {"openalex": f"https://openalex.org/{work_id}"},
                "primary_location": None,
                "best_oa_location": None,
                "locations": [],
            }
        ],
    }


def arxiv_page(
    arxiv_id: str,
    *,
    cursor: int,
    total: int,
    next_token: str | None,
) -> str:
    token = next_token or ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"
         xmlns:raw="http://arxiv.org/OAI/arXivRaw/">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="ListRecords">https://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:{arxiv_id}</identifier>
        <datestamp>2017-06-12</datestamp>
        <setSpec>cs:cs:LG</setSpec>
      </header>
      <metadata>
        <raw:arXivRaw>
          <raw:id>{arxiv_id}</raw:id>
          <raw:submitter>Example Submitter</raw:submitter>
          <raw:version version="v1">
            <raw:date>Mon, 12 Jun 2017 12:00:00 GMT</raw:date>
          </raw:version>
          <raw:title>Historical arXiv paper {arxiv_id}</raw:title>
          <raw:authors>Example Author</raw:authors>
          <raw:categories>cs.LG</raw:categories>
          <raw:abstract>Historical abstract.</raw:abstract>
        </raw:arXivRaw>
      </metadata>
    </record>
    <resumptionToken completeListSize="{total}" cursor="{cursor}">{token}</resumptionToken>
  </ListRecords>
</OAI-PMH>"""


def crossref_page(
    doi: str,
    *,
    total: int,
    next_cursor: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "message-type": "work-list",
        "message-version": "1.0.0",
        "message": {
            "total-results": total,
            "next-cursor": next_cursor,
            "items": [
                {
                    "DOI": doi,
                    "title": [f"Historical Crossref work {doi}"],
                    "published": {"date-parts": [[2020, 1, 2]]},
                    "indexed": {"date-time": "2026-08-31T12:00:00Z"},
                    "URL": f"https://doi.org/{doi}",
                }
            ],
        },
    }


def europe_pmc_page(
    record_id: str,
    *,
    total: int,
    next_cursor: str,
) -> dict[str, Any]:
    return {
        "version": "6.9",
        "hitCount": total,
        "nextCursorMark": next_cursor,
        "resultList": {
            "result": [
                {
                    "id": record_id,
                    "source": "MED",
                    "pmid": record_id,
                    "doi": f"10.5555/{record_id}",
                    "title": f"Historical Europe PMC work {record_id}",
                    "abstractText": "We introduce a biomedical neural architecture.",
                    "firstPublicationDate": "2020-01-02",
                    "firstIndexDate": "2026-08-31",
                }
            ]
        },
    }


def datacite_page(*resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "data": list(resources),
        "meta": {"total": len(resources)},
        "links": {"next": None},
    }


def datacite_resource(doi: str) -> dict[str, Any]:
    return {
        "id": doi,
        "type": "dois",
        "attributes": {
            "doi": doi,
            "state": "findable",
            "isActive": True,
            "titles": [{"title": "Historical neural software deposit"}],
            "types": {"resourceTypeGeneral": "Software"},
            "updated": "2001-01-02T03:04:05Z",
            "url": "https://repository.example/software/deposit",
        },
    }


def biorxiv_page(
    *records: Mapping[str, Any],
    cursor: int = 0,
    total: int | None = None,
) -> dict[str, Any]:
    return {
        "messages": [
            {
                "status": "ok",
                "cursor": cursor,
                "count": len(records),
                "total": len(records) if total is None else total,
            }
        ],
        "collection": list(records),
    }


def preprint(doi: str, version: int, date_value: str) -> dict[str, Any]:
    return {
        "doi": doi,
        "title": f"Historical preprint {doi} v{version}",
        "version": version,
        "date": date_value,
        "server": "bioRxiv",
    }


def osf_preprint_page(*records: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "data": list(records),
        "links": {"next": None, "meta": {"total": len(records), "per_page": 100}},
    }


def osf_preprint(identifier: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "type": "preprints",
        "attributes": {
            "title": "Historical neural preprint",
            "description": "A reusable neural-network implementation.",
            "date_published": "2020-01-02T03:04:05.000000",
            "date_modified": "2020-01-03T04:05:06.000000",
        },
        "relationships": {
            "provider": {"data": {"id": "psyarxiv", "type": "preprint-providers"}}
        },
        "links": {"html": f"https://osf.io/preprints/psyarxiv/{identifier}/"},
    }


@pytest.fixture
def database(tmp_path) -> Database:
    result = Database(tmp_path / "store")
    result.initialize()
    return result


def adapter(client: FakeHttp, **overrides: Any) -> OpenAlexSourceAdapter:
    settings: dict[str, Any] = {
        "client": client,
        "clock": lambda: NOW,
        "filter": "type:article",
        "corpus": "all",
        "sync_mode": "published",
    }
    settings.update(overrides)
    return OpenAlexSourceAdapter(**settings)


def test_backfill_uses_separate_checkpoint_and_fixed_inclusive_window(database) -> None:
    daily_state = {"watermark": "2026-08-31T00:00:00Z", "daily": True}
    database.ingest_page("openalex", (), daily_state, extractor="fixture")
    client = FakeHttp(openalex_page("W1", "1990-01-01", next_cursor=None))
    source = adapter(client)

    outcome = run_openalex_backfill(
        database,
        source,
        from_date="1990-01-01",
        to_date="1990-12-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "openalex:backfill:published:1990-01-01:1990-12-31"
    assert database.get_source_state("openalex") == daily_state
    backfill_state = database.get_source_state(outcome.source)
    assert backfill_state["backfill_complete"] is True
    assert backfill_state["window_start"] == "1990-01-01T00:00:00Z"
    assert backfill_state["window_end"] == "1990-12-31T23:59:59.999999Z"
    assert backfill_state["backfill"]["filter"] == "type:article"
    params = client.calls[0][1]
    assert params["filter"] == (
        "type:article,from_publication_date:1990-01-01,"
        "to_publication_date:1990-12-31"
    )
    assert params["corpus"] == "all"
    assert params["sort"] == "publication_date:asc"


def test_backfill_resumes_opaque_cursor_after_page_budget(database) -> None:
    client = FakeHttp(
        openalex_page("W1", "1980-01-01", next_cursor="opaque-page-two", count=2),
        openalex_page("W2", "1980-02-01", next_cursor=None, count=2),
    )
    workflow = OpenAlexBackfill(
        database,
        adapter(client),
        from_date="1980-01-01",
        to_date="1980-12-31",
        extractor="fixture",
    )

    partial = workflow.run(max_pages=1)
    partial_state = database.get_source_state(workflow.namespace)
    resumed = workflow.run(max_pages=1)

    assert partial.status == "partial"
    assert partial_state["cursor"] == "opaque-page-two"
    assert partial_state["backfill_complete"] is False
    assert resumed.status == "complete"
    assert client.calls[1][1]["cursor"] == "opaque-page-two"
    assert client.calls[1][1]["filter"] == client.calls[0][1]["filter"]
    assert database.stats()["artifacts"] == 2


def test_openalex_backfill_reuses_daily_artifact_identity(database) -> None:
    daily_source = adapter(
        FakeHttp(openalex_page("W1", "1990-01-01", next_cursor=None))
    )
    daily_page = daily_source.fetch_page({})
    database.ingest_page("openalex", daily_page, extractor="fixture")

    outcome = run_openalex_backfill(
        database,
        adapter(FakeHttp(openalex_page("W1", "1990-01-01", next_cursor=None))),
        from_date="1990-01-01",
        to_date="1990-12-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert database.stats()["artifacts"] == 1
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "openalex"
    assert artifact["source_record_id"] == "W1"


def test_completed_backfill_rerun_is_no_op(database) -> None:
    client = FakeHttp(openalex_page("W1", "1970-01-01", next_cursor=None))
    workflow = OpenAlexBackfill(
        database,
        adapter(client),
        from_date="1970-01-01",
        to_date="1970-12-31",
        extractor="fixture",
    )
    completed = workflow.run(max_pages=1)
    status_before = next(
        item for item in database.source_status() if item["source"] == workflow.namespace
    )

    repeated = workflow.run(max_pages=1)
    status_after = next(
        item for item in database.source_status() if item["source"] == workflow.namespace
    )

    assert completed.status == "complete"
    assert repeated.status == "complete"
    assert repeated.already_complete is True
    assert repeated.run_id is None
    assert len(client.calls) == 1
    assert status_after["last_run_id"] == status_before["last_run_id"]
    assert status_after["pages_ingested"] == status_before["pages_ingested"]


def test_backfill_preserves_paid_updated_sync_mode(database) -> None:
    client = FakeHttp(openalex_page("W1", "2000-01-01", next_cursor=None))
    source = adapter(
        client,
        sync_mode="updated",
        api_key="paid-key",
        corpus="all",
        filter="has_abstract:true",
    )

    outcome = run_openalex_backfill(
        database,
        source,
        from_date="2000-01-01",
        to_date="2000-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.source.startswith("openalex:backfill:updated:")
    params = client.calls[0][1]
    assert params["filter"] == (
        "has_abstract:true,from_updated_date:2000-01-01,to_updated_date:2000-01-31"
    )
    assert params["sort"] == "updated_date:asc"
    assert params["api_key"] == "paid-key"


def test_backfill_refuses_to_resume_with_changed_source_config(database) -> None:
    first_client = FakeHttp(
        openalex_page("W1", "1960-01-01", next_cursor="still-running", count=2)
    )
    first = OpenAlexBackfill(
        database,
        adapter(first_client, filter="type:article"),
        from_date="1960-01-01",
        to_date="1960-12-31",
        extractor="fixture",
    )
    assert first.run(max_pages=1).status == "partial"

    changed_client = FakeHttp()
    changed = OpenAlexBackfill(
        database,
        adapter(changed_client, filter="type:book"),
        from_date="1960-01-01",
        to_date="1960-12-31",
        extractor="fixture",
    )

    with pytest.raises(ValueError, match="does not match"):
        changed.run(max_pages=1)
    assert changed_client.calls == []


@pytest.mark.parametrize(
    ("from_date", "to_date", "message"),
    [
        ("not-a-date", "2000-01-01", "YYYY-MM-DD"),
        ("2000-02-30", "2000-03-01", "invalid from_date"),
        ("2001-01-01", "2000-01-01", "on or before"),
        (datetime(2000, 1, 1), "2000-01-02", "without a time"),
    ],
)
def test_backfill_validates_date_range(
    database,
    from_date: Any,
    to_date: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        OpenAlexBackfill(
            database,
            adapter(FakeHttp()),
            from_date=from_date,
            to_date=to_date,
        )


@pytest.mark.parametrize("namespace", ["openalex", "catalog", "huggingface"])
def test_backfill_rejects_namespace_outside_its_reserved_prefix(
    database, namespace: str
) -> None:
    with pytest.raises(ValueError, match="must begin with 'openalex:backfill:'"):
        OpenAlexBackfill(
            database,
            adapter(FakeHttp()),
            from_date="2000-01-01",
            to_date="2000-01-02",
            namespace=namespace,
        )


def test_backfill_rejects_custom_namespace_owned_by_another_source(database) -> None:
    database.ingest_page(
        "openalex:backfill:occupied",
        (),
        {"checked_at": "2026-09-01T00:00:00Z"},
        extractor="fixture",
    )
    client = FakeHttp()

    with pytest.raises(ValueError, match="existing non-backfill source checkpoint"):
        OpenAlexBackfill(
            database,
            adapter(client),
            from_date="2000-01-01",
            to_date="2000-01-02",
            namespace="openalex:backfill:occupied",
        )

    assert client.calls == []
    assert database.get_source_state("openalex:backfill:occupied") == {
        "checked_at": "2026-09-01T00:00:00Z"
    }


def test_custom_namespace_can_resume_its_own_backfill_checkpoint(database) -> None:
    first_client = FakeHttp(
        openalex_page("W1", "1985-01-01", next_cursor="page-two", count=2)
    )
    first = OpenAlexBackfill(
        database,
        adapter(first_client),
        from_date="1985-01-01",
        to_date="1985-12-31",
        namespace="openalex:backfill:custom",
        extractor="fixture",
    )
    assert first.run(max_pages=1).status == "partial"

    second_client = FakeHttp(
        openalex_page("W2", "1985-02-01", next_cursor=None, count=2)
    )
    resumed = OpenAlexBackfill(
        database,
        adapter(second_client),
        from_date="1985-01-01",
        to_date="1985-12-31",
        namespace="openalex:backfill:custom",
        extractor="fixture",
    ).run(max_pages=1)

    assert resumed.status == "complete"
    assert second_client.calls[0][1]["cursor"] == "page-two"


def test_custom_namespace_can_retry_an_empty_reserved_checkpoint(database) -> None:
    namespace = "openalex:backfill:reserved"
    reservation_run = database.start_run(namespace)
    database.finish_run(reservation_run, "failed", error="initial fetch failed")
    client = FakeHttp(openalex_page("W1", "1986-01-01", next_cursor=None))

    outcome = OpenAlexBackfill(
        database,
        adapter(client),
        from_date="1986-01-01",
        to_date="1986-12-31",
        namespace=namespace,
        extractor="fixture",
    ).run(max_pages=1)

    assert outcome.status == "complete"
    assert len(client.calls) == 1


def test_backfill_requires_finite_positive_page_budget(database) -> None:
    workflow = OpenAlexBackfill(
        database,
        adapter(FakeHttp()),
        from_date="2000-01-01",
        to_date="2000-01-02",
    )

    with pytest.raises(ValueError, match="finite, positive"):
        workflow.run(max_pages=None)
    with pytest.raises(ValueError, match="finite, positive"):
        workflow.run(max_pages=0)


def test_backfill_rejects_conflicting_adapter_date_filter(database) -> None:
    with pytest.raises(ValueError, match="date boundaries"):
        OpenAlexBackfill(
            database,
            adapter(
                FakeHttp(),
                filter="type:article,from_publication_date:1950-01-01",
            ),
            from_date="2000-01-01",
            to_date="2000-01-02",
        )


def test_arxiv_backfill_resumes_token_in_a_fixed_inclusive_window(database) -> None:
    daily_state = {"watermark": "2026-08-31", "daily": True}
    database.ingest_page("arxiv", (), daily_state, extractor="fixture")
    client = FakeXmlHttp(
        arxiv_page("1706.00001", cursor=0, total=2, next_token="page-two-token"),
        arxiv_page("1706.00002", cursor=1, total=2, next_token=None),
    )
    workflow = ArxivBackfill(
        database,
        ArxivSourceAdapter(client=client, clock=lambda: NOW),
        from_date="2017-01-01",
        to_date="2017-12-31",
        extractor="fixture",
    )

    partial = workflow.run(max_pages=1)
    partial_state = database.get_source_state(workflow.namespace)
    completed = workflow.run(max_pages=1)

    assert partial.status == "partial"
    assert partial_state["resumption_token"] == "page-two-token"
    assert partial_state["raw_items_seen"] == 1
    assert partial_state["backfill_complete"] is False
    assert partial_state["window_start"] == "2017-01-01"
    assert partial_state["window_end"] == "2017-12-31"
    assert partial_state["backfill"]["adapter_type"] == "ArxivSourceAdapter"
    assert partial_state["_modelome_source_signature"] == workflow.source.checkpoint_signature
    assert completed.status == "complete"
    assert database.get_source_state("arxiv") == daily_state
    assert client.calls[0][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": "2017-01-01",
        "until": "2017-12-31",
    }
    assert client.calls[1][1] == {
        "verb": "ListRecords",
        "resumptionToken": "page-two-token",
    }
    assert database.stats()["artifacts"] == 2


def test_arxiv_backfill_reuses_daily_artifact_identity(database) -> None:
    paper = arxiv_page("1706.03762", cursor=0, total=1, next_token=None)
    daily_source = ArxivSourceAdapter(
        client=FakeXmlHttp(paper),
        clock=lambda: NOW,
        initial_lookback_days=1,
    )
    database.ingest_page(
        "arxiv",
        daily_source.fetch_page({}),
        extractor="fixture",
    )

    outcome = run_arxiv_backfill(
        database,
        ArxivSourceAdapter(client=FakeXmlHttp(paper), clock=lambda: NOW),
        from_date="2017-06-12",
        to_date="2017-06-12",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "arxiv:backfill:2017-06-12:2017-06-12"
    assert database.stats()["artifacts"] == 1
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "arxiv"
    assert artifact["source_record_id"] == "1706.03762"


def test_crossref_backfill_resumes_cursor_in_a_fixed_index_window(database) -> None:
    daily_state = {"watermark": "2026-08-31", "daily": True}
    database.ingest_page("crossref", (), daily_state, extractor="fixture")
    client = FakeHttp(
        crossref_page("10.5555/first", total=2, next_cursor="page-two"),
        crossref_page("10.5555/second", total=2, next_cursor="unused"),
    )
    workflow = CrossrefBackfill(
        database,
        CrossrefSourceAdapter(
            client=client,
            clock=lambda: NOW,
            page_size=1,
            mailto="registry@example.test",
        ),
        from_date="2020-01-01",
        to_date="2020-12-31",
        extractor="fixture",
    )

    partial = workflow.run(max_pages=1)
    partial_state = database.get_source_state(workflow.namespace)
    completed = workflow.run(max_pages=1)

    assert partial.status == "partial"
    assert partial_state["cursor"] == "page-two"
    assert partial_state["raw_items_seen"] == 1
    assert partial_state["scan_total"] == 2
    assert partial_state["backfill_complete"] is False
    assert partial_state["window_start"] == "2020-01-01"
    assert partial_state["window_end"] == "2020-12-31"
    assert partial_state["backfill"]["adapter_type"] == "CrossrefSourceAdapter"
    assert partial_state["_modelome_source_signature"] == workflow.source.checkpoint_signature
    assert completed.status == "complete"
    assert database.get_source_state("crossref") == daily_state
    assert client.calls[0][1] == {
        "cursor": "*",
        "rows": 1,
        "filter": "from-index-date:2020-01-01,until-index-date:2020-12-31",
        "mailto": "registry@example.test",
    }
    assert client.calls[1][1]["cursor"] == "page-two"
    assert client.calls[1][1]["filter"] == client.calls[0][1]["filter"]
    assert database.stats()["artifacts"] == 2


def test_crossref_backfill_reuses_daily_artifact_identity(database) -> None:
    payload = crossref_page("10.1038/example", total=1, next_cursor="unused")
    daily_source = CrossrefSourceAdapter(
        client=FakeHttp(payload),
        clock=lambda: NOW,
        page_size=1,
        initial_lookback_days=1,
    )
    database.ingest_page(
        "crossref",
        daily_source.fetch_page({}),
        extractor="fixture",
    )

    outcome = run_crossref_backfill(
        database,
        CrossrefSourceAdapter(
            client=FakeHttp(payload),
            clock=lambda: NOW,
            page_size=1,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "crossref:backfill:2020-01-01:2020-01-31"
    assert database.stats()["artifacts"] == 1
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "crossref"
    assert artifact["source_record_id"] == "10.1038/example"


def test_europe_pmc_backfill_resumes_cursor_in_a_fixed_update_window(database) -> None:
    daily_state = {"watermark": "2026-08-31", "daily": True}
    database.ingest_page("europe-pmc", (), daily_state, extractor="fixture")
    client = FakeHttp(
        europe_pmc_page("111", total=2, next_cursor="page-two"),
        europe_pmc_page("222", total=2, next_cursor="unused"),
    )
    workflow = EuropePmcBackfill(
        database,
        EuropePmcSourceAdapter(
            client=client,
            clock=lambda: NOW,
            page_size=1,
            email="registry@example.test",
        ),
        from_date="2020-01-01",
        to_date="2020-12-31",
        extractor="fixture",
    )

    partial = workflow.run(max_pages=1)
    partial_state = database.get_source_state(workflow.namespace)
    completed = workflow.run(max_pages=1)

    assert partial.status == "partial"
    assert partial_state["cursor_mark"] == "page-two"
    assert partial_state["raw_items_seen"] == 1
    assert partial_state["scan_total"] == 2
    assert partial_state["backfill_complete"] is False
    assert partial_state["window_start"] == "2020-01-01"
    assert partial_state["window_end"] == "2020-12-31"
    assert partial_state["backfill"]["adapter_type"] == "EuropePmcSourceAdapter"
    assert partial_state["_modelome_source_signature"] == workflow.source.checkpoint_signature
    assert completed.status == "complete"
    assert database.get_source_state("europe-pmc") == daily_state
    assert client.calls[0][1] == {
        "query": "UPDATE_DATE:[2020-01-01 TO 2020-12-31]",
        "format": "json",
        "resultType": "core",
        "cursorMark": "*",
        "pageSize": 1,
        "synonym": "false",
        "email": "registry@example.test",
    }
    assert client.calls[1][1]["cursorMark"] == "page-two"
    assert client.calls[1][1]["query"] == client.calls[0][1]["query"]
    assert database.stats()["artifacts"] == 2


def test_europe_pmc_backfill_reuses_daily_artifact_identity(database) -> None:
    payload = europe_pmc_page("12345", total=1, next_cursor="unused")
    daily_source = EuropePmcSourceAdapter(
        client=FakeHttp(payload),
        clock=lambda: NOW,
        page_size=1,
        initial_lookback_days=1,
    )
    database.ingest_page(
        "europe-pmc",
        daily_source.fetch_page({}),
        extractor="fixture",
    )

    outcome = run_europe_pmc_backfill(
        database,
        EuropePmcSourceAdapter(
            client=FakeHttp(payload),
            clock=lambda: NOW,
            page_size=1,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "europe-pmc:backfill:2020-01-01:2020-01-31"
    assert database.stats()["artifacts"] == 1
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "europe-pmc"
    assert artifact["source_record_id"] == "MED:12345"


def test_datacite_backfill_uses_fixed_window_and_dynamic_resource_kind(database) -> None:
    client = FakeHttp(datacite_page(datacite_resource("10.5438/example")))
    workflow = DataCiteBackfill(
        database,
        DataCiteSourceAdapter(
            client=client,
            clock=lambda: NOW,
            page_size=1,
        ),
        from_date="2001-01-01",
        to_date="2001-01-31",
        extractor="fixture",
    )

    outcome = workflow.run(max_pages=1)
    repeated = run_datacite_backfill(
        database,
        workflow.source.adapter,
        from_date="2001-01-01",
        to_date="2001-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "datacite:backfill:2001-01-01:2001-01-31"
    assert repeated.already_complete is True
    assert len(client.calls) == 1
    assert client.calls[0][1]["query"] == "updated:[2001-01-01 TO 2001-01-31]"
    state = database.get_source_state(workflow.namespace)
    assert state["backfill_complete"] is True
    assert state["backfill"]["artifact_kind"] == "resource_type"
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "datacite"
    assert artifact["kind"] == "code_repository"


def test_biorxiv_backfill_uses_separate_fixed_inclusive_window(database) -> None:
    daily_state = {"watermark": "2026-08-31", "daily": True}
    database.ingest_page("biorxiv", (), daily_state, extractor="fixture")
    client = FakeHttp(
        biorxiv_page(preprint("10.1101/2020.01.02.123456", 1, "2020-01-02"))
    )
    source = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        client=client,
        clock=lambda: NOW,
    )

    workflow = BioRxivBackfill(
        database,
        source,
        from_date="2020-01-01",
        to_date="2020-01-31",
        extractor="fixture",
    )
    outcome = workflow.run(max_pages=1)

    assert outcome.status == "complete"
    assert outcome.source == "biorxiv:backfill:2020-01-01:2020-01-31"
    assert database.get_source_state("biorxiv") == daily_state
    state = database.get_source_state(outcome.source)
    assert state["backfill_complete"] is True
    assert state["window_start"] == "2020-01-01"
    assert state["window_end"] == "2020-01-31"
    assert state["backfill"]["adapter_type"] == "BioRxivSourceAdapter"
    assert state["_modelome_source_signature"] == workflow.source.checkpoint_signature
    assert client.calls[0][0] == (
        "https://api.biorxiv.org/details/biorxiv/2020-01-01/2020-01-31/0/json"
    )


def test_biorxiv_backfill_resumes_offset_after_page_budget(database) -> None:
    client = FakeHttp(
        biorxiv_page(
            preprint("10.1101/2020.01.01.000001", 1, "2020-01-01"),
            cursor=0,
            total=2,
        ),
        biorxiv_page(
            preprint("10.1101/2020.01.02.000002", 1, "2020-01-02"),
            cursor=1,
            total=2,
        ),
    )
    workflow = BioRxivBackfill(
        database,
        BioRxivSourceAdapter(
            name="biorxiv",
            url="https://api.biorxiv.org/details",
            server="biorxiv",
            client=client,
            clock=lambda: NOW,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
        extractor="fixture",
    )

    partial = workflow.run(max_pages=1)
    state = database.get_source_state(workflow.namespace)
    resumed = workflow.run(max_pages=1)

    assert partial.status == "partial"
    assert state["cursor"] == 1
    assert state["backfill_complete"] is False
    assert resumed.status == "complete"
    assert client.calls[1][0].endswith("/2020-01-01/2020-01-31/1/json")
    assert database.stats()["artifacts"] == 2


def test_biorxiv_publication_link_stream_has_its_own_backfill(database) -> None:
    client = FakeHttp(
        biorxiv_page(
            {
                "biorxiv_doi": "10.1101/2020.01.02.123456",
                "published_doi": "10.1038/example",
                "preprint_platform": "bioRxiv",
                "preprint_date": "2020-01-02",
                "published_date": "2022-06-01",
            }
        )
    )
    source = BioRxivPublicationSourceAdapter(
        name="biorxiv-publications",
        url="https://api.biorxiv.org/pubs",
        server="biorxiv",
        client=client,
        clock=lambda: NOW,
    )

    outcome = run_biorxiv_backfill(
        database,
        source,
        from_date="2022-06-01",
        to_date="2022-06-30",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source.startswith("biorxiv-publications:backfill:")
    assert client.calls[0][0] == (
        "https://api.biorxiv.org/pubs/biorxiv/2022-06-01/2022-06-30/0"
    )


def test_biorxiv_backfill_refuses_changed_adapter_config(database) -> None:
    first_client = FakeHttp(
        biorxiv_page(
            preprint("10.1101/2020.01.01.000001", 1, "2020-01-01"),
            cursor=0,
            total=2,
        )
    )
    first = BioRxivBackfill(
        database,
        BioRxivSourceAdapter(
            name="biorxiv",
            url="https://api.biorxiv.org/details",
            server="biorxiv",
            overlap_days=2,
            client=first_client,
            clock=lambda: NOW,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
        extractor="fixture",
    )
    assert first.run(max_pages=1).status == "partial"

    changed_client = FakeHttp()
    changed = BioRxivBackfill(
        database,
        BioRxivSourceAdapter(
            name="biorxiv",
            url="https://api.biorxiv.org/details",
            server="biorxiv",
            overlap_days=3,
            client=changed_client,
            clock=lambda: NOW,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
        extractor="fixture",
    )

    with pytest.raises(ValueError, match="does not match"):
        changed.run(max_pages=1)
    assert changed_client.calls == []


def test_biorxiv_backfill_reuses_daily_artifact_identity(database) -> None:
    record = preprint("10.1101/2020.01.02.123456", 1, "2020-01-02")
    daily_source = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=1,
        client=FakeHttp(biorxiv_page(record)),
        clock=lambda: NOW,
    )
    daily_page = daily_source.fetch_page({})
    database.ingest_page(
        "biorxiv",
        daily_page,
        extractor="fixture",
    )
    backfill_source = BioRxivSourceAdapter(
        name="biorxiv",
        url="https://api.biorxiv.org/details",
        server="biorxiv",
        initial_lookback_days=1,
        client=FakeHttp(biorxiv_page(record)),
        clock=lambda: NOW,
    )

    outcome = run_biorxiv_backfill(
        database,
        backfill_source,
        from_date="2020-01-01",
        to_date="2020-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert database.stats()["artifacts"] == 1
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "biorxiv"
    assert artifact["source_record_id"] == "biorxiv:10.1101/2020.01.02.123456:v1"


def test_osf_preprints_backfill_uses_the_daily_artifact_namespace(database) -> None:
    daily_state = {"watermark": "2026-08-31", "daily": True}
    database.ingest_page("osf-preprints", (), daily_state, extractor="fixture")
    client = FakeHttp(osf_preprint_page(osf_preprint("a1b2c_v1")))
    source = OsfPreprintSourceAdapter(client=client, clock=lambda: NOW)

    outcome = run_osf_preprints_backfill(
        database,
        source,
        from_date="2020-01-01",
        to_date="2020-01-31",
        max_pages=1,
        extractor="fixture",
    )

    assert outcome.status == "complete"
    assert outcome.source == "osf-preprints:backfill:2020-01-01:2020-01-31"
    assert database.get_source_state("osf-preprints") == daily_state
    state = database.get_source_state(outcome.source)
    assert state["backfill_complete"] is True
    assert state["backfill"]["adapter_type"] == "OsfPreprintSourceAdapter"
    assert client.calls[0][1] == {
        "page": 1,
        "page[size]": 100,
        "sort": "date_modified",
        "filter[date_modified][gte]": "2020-01-01T00:00:00Z",
        "filter[date_modified][lte]": "2020-01-31T23:59:59.999999Z",
    }
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source"] == "osf-preprints"
    assert artifact["source_record_id"] == "osf-preprints:a1b2c_v1"


@pytest.mark.parametrize("invalid", ["false", 1, None])
def test_biorxiv_backfill_rejects_non_boolean_completion_state(
    database, invalid: Any
) -> None:
    workflow = BioRxivBackfill(
        database,
        BioRxivSourceAdapter(
            name="biorxiv",
            url="https://api.biorxiv.org/details",
            server="biorxiv",
            client=FakeHttp(),
            clock=lambda: NOW,
        ),
        from_date="2020-01-01",
        to_date="2020-01-31",
    )
    state = {
        "backfill": workflow.source.signature,
        "backfill_complete": invalid,
        "window_start": "2020-01-01",
        "window_end": "2020-01-31",
    }

    with pytest.raises(ValueError, match="must be boolean"):
        workflow.source.is_complete(state)
