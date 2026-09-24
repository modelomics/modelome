from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.normalize import content_hash
from modelome.pipeline import SyncEngine
from modelome.pmc_bootstrap import PmcBootstrap, PmcBootstrapSource, run_pmc_bootstrap
from modelome.sources.pmc import PmcSourceAdapter
from modelome.storage import Database

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
OAI = "http://www.openarchives.org/OAI/2.0/"
JATS = "https://jats.nlm.nih.gov/ns/archiving/1.4/"


class QueueClient:
    def __init__(self, *bodies: str) -> None:
        self.bodies = list(bodies)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.bodies:
            raise AssertionError(f"unexpected request: {url}")
        return HttpResponse(
            status=200,
            headers={"content-type": "application/xml"},
            body=self.bodies.pop(0).encode(),
            url=url,
        )


def oai(body: str, *, verb: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="{verb}">https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/</request>
  {body}
</OAI-PMH>"""


def identify(*, earliest: str = "1999-01-01") -> str:
    return oai(
        f"""
<Identify>
  <repositoryName>PubMed Central</repositoryName>
  <baseURL>https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/</baseURL>
  <protocolVersion>2.0</protocolVersion>
  <adminEmail>example@example.test</adminEmail>
  <earliestDatestamp>{earliest}</earliestDatestamp>
  <deletedRecord>no</deletedRecord>
  <granularity>YYYY-MM-DD</granularity>
</Identify>""",
        verb="Identify",
    )


def record(pmc_number: str, *, metadata_pmc_number: str | None = None) -> str:
    metadata_number = metadata_pmc_number or pmc_number
    return f"""
<record>
  <header>
    <identifier>oai:pubmedcentral.nih.gov:{pmc_number}</identifier>
    <datestamp>2026-08-31</datestamp>
    <setSpec>pmc-open</setSpec>
  </header>
  <metadata>
    <article xmlns="{JATS}" article-type="research-article">
      <front>
        <article-meta>
          <article-id pub-id-type="pmcid">PMC{metadata_number}</article-id>
          <article-id pub-id-type="pmid">{pmc_number}</article-id>
          <title-group><article-title>Article {pmc_number}</article-title></title-group>
          <pub-date pub-type="epub"><year>2025</year></pub-date>
          <abstract><p>A neural learning system is introduced from ordinary text.</p></abstract>
        </article-meta>
      </front>
      <body><sec><p>Complete reusable body text.</p></sec></body>
    </article>
  </metadata>
</record>"""


def page(
    pmc_number: str,
    *,
    token: str = "",
    total: int = 1,
    cursor: int = 0,
    metadata_pmc_number: str | None = None,
) -> str:
    return oai(
        f"<ListRecords>"
        f"{record(pmc_number, metadata_pmc_number=metadata_pmc_number)}"
        f'<resumptionToken completeListSize="{total}" cursor="{cursor}">'
        f"{token}</resumptionToken>"
        f"</ListRecords>",
        verb="ListRecords",
    )


def no_records() -> str:
    return oai(
        '<error code="noRecordsMatch">No matching records</error>',
        verb="ListRecords",
    )


def adapter(client: QueueClient, **overrides: Any) -> PmcSourceAdapter:
    values: dict[str, Any] = {"client": client, "clock": lambda: NOW}
    values.update(overrides)
    return PmcSourceAdapter(**values)


def database(path: Path) -> Database:
    result = Database(path / "store")
    result.initialize()
    return result


def valid_state(source: PmcBootstrapSource, *, complete: bool = False) -> dict[str, Any]:
    repository = {
        "repository_name": "PubMed Central",
        "base_url": source.adapter.url,
        "protocol_version": "2.0",
        "earliest_datestamp": "1999-01-01",
        "deleted_record": "no",
        "granularity": "YYYY-MM-DD",
    }
    state: dict[str, Any] = {
        "bootstrap": {
            "version": 1,
            "workflow_checkpoint_signature": source.checkpoint_signature,
            "adapter_checkpoint_signature": source.adapter.checkpoint_signature,
            "repository_identity_signature": content_hash(repository),
            "repository": repository,
            "window_start": "1999-01-01",
            "window_end": "2026-08-31",
            "discovered_at": "2026-09-01T12:00:00Z",
        },
        "bootstrap_complete": complete,
        "window_start": "1999-01-01",
        "window_end": "2026-08-31",
        "_modelome_source_signature": source.checkpoint_signature,
    }
    if complete:
        state["completed_at"] = "2026-09-01T12:30:00Z"
    return state


def test_bootstrap_discovers_and_freezes_complete_repository_history(
    tmp_path: Path,
) -> None:
    client = QueueClient(identify(), page("1234567"))
    source = adapter(client)
    store = database(tmp_path)

    outcome = run_pmc_bootstrap(
        store, source, max_pages=1, extractor="fixture"
    )

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    state = store.get_source_state("pmc:bootstrap")
    descriptor = state["bootstrap"]
    assert state["bootstrap_complete"] is True
    assert descriptor["window_start"] == "1999-01-01"
    assert descriptor["window_end"] == "2026-08-31"
    assert descriptor["repository"]["earliest_datestamp"] == "1999-01-01"
    assert descriptor["repository_identity_signature"] == content_hash(
        descriptor["repository"]
    )
    assert descriptor["workflow_checkpoint_signature"] == (
        PmcBootstrapSource(source).checkpoint_signature
    )
    assert client.calls[1][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "pmc",
        "set": "pmc-open",
        "from": "1999-01-01",
        "until": "2026-08-31",
    }
    artifacts = store.table_rows("artifacts")
    assert len(artifacts) == 1
    assert artifacts[0]["source"] == "pmc"
    assert artifacts[0]["source_record_id"] == "PMC1234567"


def test_bootstrap_resumes_token_without_repeating_identify(tmp_path: Path) -> None:
    client = QueueClient(
        identify(),
        page("1111111", token="opaque-next", total=2, cursor=0),
        page("2222222", total=2, cursor=1),
    )
    workflow = PmcBootstrap(
        database(tmp_path), adapter(client), extractor="fixture"
    )

    first = workflow.run(max_pages=1)
    partial_state = workflow.database.get_source_state(workflow.namespace)
    second = workflow.run(max_pages=1)

    assert first.status == "partial"
    assert second.status == "complete"
    assert partial_state["bootstrap_complete"] is False
    assert partial_state["resumption_token"] == "opaque-next"
    assert partial_state["bootstrap"]["window_start"] == "1999-01-01"
    assert [call[1]["verb"] for call in client.calls] == [
        "Identify",
        "ListRecords",
        "ListRecords",
    ]
    assert client.calls[2][1] == {
        "verb": "ListRecords",
        "resumptionToken": "opaque-next",
    }
    artifacts = workflow.database.table_rows("artifacts")
    assert {item["source"] for item in artifacts} == {"pmc"}
    assert len(artifacts) == 2


def test_completed_bootstrap_is_a_network_free_noop(tmp_path: Path) -> None:
    client = QueueClient(identify(), no_records())
    workflow = PmcBootstrap(
        database(tmp_path), adapter(client), extractor="fixture"
    )

    first = workflow.run(max_pages=1)
    second = workflow.run(max_pages=1)

    assert first.status == "complete"
    assert second.status == "complete"
    assert second.already_complete is True
    assert second.run_id is None
    assert second.stats["upstream_count"] == 0
    assert len(client.calls) == 2


def test_malformed_page_retries_frozen_window_without_reidentifying(
    tmp_path: Path,
) -> None:
    client = QueueClient(
        identify(),
        page("1234567", metadata_pmc_number="9999999"),
        page("1234567"),
    )
    workflow = PmcBootstrap(
        database(tmp_path), adapter(client), extractor="fixture"
    )

    failed = workflow.run(max_pages=1)
    held = workflow.database.get_source_state(workflow.namespace)
    recovered = workflow.run(max_pages=1)

    assert failed.status == "failed"
    assert recovered.status == "complete"
    assert held["bootstrap_complete"] is False
    assert held["raw_items_seen"] == 0
    assert held["window_start"] == "1999-01-01"
    assert held["window_end"] == "2026-08-31"
    assert client.calls[2][1] == client.calls[1][1]
    assert [call[1]["verb"] for call in client.calls].count("Identify") == 1


def test_bootstrap_and_daily_source_share_artifact_identity(tmp_path: Path) -> None:
    store = database(tmp_path)
    bootstrap_client = QueueClient(identify(), page("1234567"))
    historical = adapter(bootstrap_client)
    run_pmc_bootstrap(store, historical, max_pages=1, extractor="fixture")

    daily = adapter(QueueClient(page("1234567")))
    outcome = SyncEngine(store, {daily.name: daily}, extractor="fixture").sync(
        max_pages=1
    )[0]

    assert outcome.status == "complete"
    artifacts = store.table_rows("artifacts")
    assert len(artifacts) == 1
    assert artifacts[0]["source"] == historical.name == daily.name
    assert PmcBootstrapSource(historical).artifact_source == daily.name


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.0])
def test_bootstrap_requires_a_finite_positive_integer_page_budget(
    tmp_path: Path, budget: Any
) -> None:
    workflow = PmcBootstrap(database(tmp_path), adapter(QueueClient()))

    with pytest.raises(ValueError, match="finite, positive max_pages"):
        workflow.run(max_pages=budget)


def test_future_repository_boundary_completes_without_list_request(
    tmp_path: Path,
) -> None:
    client = QueueClient(identify(earliest="2026-09-02"))
    workflow = PmcBootstrap(database(tmp_path), adapter(client))

    outcome = workflow.run(max_pages=1)

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 0
    assert len(client.calls) == 1
    state = workflow.database.get_source_state(workflow.namespace)
    assert state["bootstrap_complete"] is True
    assert state["window_start"] == "2026-09-02"
    assert state["window_end"] == "2026-08-31"


def test_partial_state_without_completion_marker_is_not_complete() -> None:
    source = PmcBootstrapSource(adapter(QueueClient()))
    state = valid_state(source)
    del state["bootstrap_complete"]

    assert source.is_complete(state) is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda state: state.update(bootstrap_complete="true"),
            "bootstrap_complete checkpoint value must be boolean",
        ),
        (
            lambda state: state["bootstrap"].update(version=True),
            "checkpoint version must be an integer",
        ),
        (
            lambda state: state["bootstrap"].update(version=2),
            "unsupported bootstrap checkpoint version",
        ),
        (
            lambda state: state["bootstrap"].update(
                workflow_checkpoint_signature="0" * 64
            ),
            "different workflow config",
        ),
        (
            lambda state: state["bootstrap"].update(
                adapter_checkpoint_signature="0" * 64
            ),
            "different PMC adapter",
        ),
        (
            lambda state: state.update(_modelome_source_signature="0" * 64),
            "stored source signature",
        ),
        (
            lambda state: state["bootstrap"].update(
                repository_identity_signature="0" * 64
            ),
            "repository identity signature does not match",
        ),
        (
            lambda state: state["bootstrap"]["repository"].update(
                unexpected="value"
            ),
            "identity fields are incomplete or unknown",
        ),
        (
            lambda state: state["bootstrap"].update(window_start="2000-01-01"),
            "lower boundary does not match",
        ),
        (
            lambda state: state["bootstrap"].update(window_end="2026-08-30"),
            "upper boundary is not the last closed UTC day",
        ),
        (
            lambda state: state.update(window_start="2000-01-01"),
            "outer window_start does not match",
        ),
        (
            lambda state: state.update(upstream_count=True),
            "upstream_count must be a nonnegative integer",
        ),
    ],
)
def test_bootstrap_rejects_corrupt_or_wrongly_typed_checkpoint_state(
    mutate: Any, message: str
) -> None:
    source = PmcBootstrapSource(adapter(QueueClient()))
    state = valid_state(source)
    mutate(state)

    with pytest.raises(ValueError, match=message):
        source.is_complete(state)


def test_completed_state_requires_a_typed_completion_timestamp() -> None:
    source = PmcBootstrapSource(adapter(QueueClient()))
    state = valid_state(source, complete=True)
    del state["completed_at"]

    with pytest.raises(ValueError, match="completed_at is required"):
        source.is_complete(state)


def test_adapter_configuration_change_invalidates_frozen_checkpoint() -> None:
    original = PmcBootstrapSource(adapter(QueueClient()))
    state = valid_state(original)
    changed = PmcBootstrapSource(adapter(QueueClient(), overlap_days=3))

    with pytest.raises(ValueError, match="different workflow config"):
        changed.is_complete(state)


def test_bootstrap_checkpoint_namespace_cannot_overwrite_daily_state() -> None:
    daily = adapter(QueueClient())

    with pytest.raises(ValueError, match="must differ from the daily source name"):
        PmcBootstrapSource(daily, namespace=daily.name)
