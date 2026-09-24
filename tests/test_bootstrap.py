from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.bootstrap import ArxivBootstrap, ArxivBootstrapSource, run_arxiv_bootstrap
from modelome.http import HttpResponse
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.storage import Database

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
OAI = "http://www.openarchives.org/OAI/2.0/"
RAW = "http://arxiv.org/OAI/arXivRaw/"


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


def identify(
    *,
    earliest: str = "1991-08-14",
    base_url: str = "https://oaipmh.arxiv.org/oai",
    granularity: str = "YYYY-MM-DD",
) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="Identify">{base_url}</request>
  <Identify>
    <repositoryName>arXiv</repositoryName>
    <baseURL>{base_url}</baseURL>
    <protocolVersion>2.0</protocolVersion>
    <adminEmail>help@example.test</adminEmail>
    <earliestDatestamp>{earliest}</earliestDatestamp>
    <deletedRecord>persistent</deletedRecord>
    <granularity>{granularity}</granularity>
  </Identify>
</OAI-PMH>"""


def page(*, token: str | None = None, total: int = 1, cursor: int = 0) -> str:
    token_text = token or ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI}" xmlns:raw="{RAW}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="ListRecords">https://oaipmh.arxiv.org/oai</request>
  <ListRecords>
    <record>
      <header>
        <identifier>oai:arXiv.org:1706.03762</identifier>
        <datestamp>2017-06-12</datestamp>
        <setSpec>cs:cs.CL</setSpec>
      </header>
      <metadata>
        <raw:arXivRaw>
          <raw:id>1706.03762</raw:id>
          <raw:version version="v1">
            <raw:date>Mon, 12 Jun 2017 17:57:34 GMT</raw:date>
          </raw:version>
          <raw:title>Attention Is All You Need</raw:title>
          <raw:authors>Ashish Vaswani et al.</raw:authors>
          <raw:categories>cs.CL cs.LG</raw:categories>
          <raw:abstract>We propose a new network architecture, the Transformer.</raw:abstract>
        </raw:arXivRaw>
      </metadata>
    </record>
    <resumptionToken completeListSize="{total}" cursor="{cursor}">{token_text}</resumptionToken>
  </ListRecords>
</OAI-PMH>"""


def no_records() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="{OAI}">
  <responseDate>2026-09-01T12:00:00Z</responseDate>
  <request verb="ListRecords">https://oaipmh.arxiv.org/oai</request>
  <error code="noRecordsMatch">No records</error>
</OAI-PMH>"""


def adapter(client: QueueClient, **overrides: Any) -> ArxivSourceAdapter:
    values: dict[str, Any] = {"client": client, "clock": lambda: NOW}
    values.update(overrides)
    return ArxivSourceAdapter(**values)


def test_identify_discovers_repository_boundary_without_a_seed_date() -> None:
    client = QueueClient(
        identify(earliest="1991-08-14T00:00:00Z", granularity="YYYY-MM-DDThh:mm:ssZ")
    )

    identity = adapter(client).identify()

    assert client.calls[0][1] == {"verb": "Identify"}
    assert identity.repository_name == "arXiv"
    assert identity.earliest_datestamp == "1991-08-14T00:00:00Z"
    assert identity.deleted_record == "persistent"
    assert identity.granularity == "YYYY-MM-DDThh:mm:ssZ"


def test_bootstrap_freezes_identify_bounds_and_acquires_aiayn(tmp_path: Path) -> None:
    client = QueueClient(identify(), page())
    source = adapter(client)
    database = Database(tmp_path / "store")
    database.initialize()

    outcome = run_arxiv_bootstrap(database, source, max_pages=1, extractor="fixture")

    assert outcome.status == "complete"
    assert outcome.stats["records_seen"] == 1
    state = database.get_source_state("arxiv:bootstrap")
    assert state["bootstrap_complete"] is True
    assert state["bootstrap"]["window_start"] == "1991-08-14"
    assert state["bootstrap"]["window_end"] == "2026-08-31"
    assert state["bootstrap"]["repository"]["earliest_datestamp"] == "1991-08-14"
    assert client.calls[1][1] == {
        "verb": "ListRecords",
        "metadataPrefix": "arXivRaw",
        "from": "1991-08-14",
        "until": "2026-08-31",
    }
    artifacts = database.table_rows("artifacts")
    assert len(artifacts) == 1
    assert artifacts[0]["source"] == "arxiv"
    assert artifacts[0]["source_record_id"] == "1706.03762"


def test_bootstrap_resumes_frozen_token_without_repeating_identify(tmp_path: Path) -> None:
    client = QueueClient(
        identify(),
        page(token="opaque-next", total=2),
        page(total=2, cursor=1),
    )
    workflow = ArxivBootstrap(
        Database(tmp_path / "store"),
        adapter(client),
        extractor="fixture",
    )
    workflow.database.initialize()

    first = workflow.run(max_pages=1)
    state = workflow.database.get_source_state(workflow.namespace)
    second = workflow.run(max_pages=1)

    assert first.status == "partial"
    assert second.status == "complete"
    assert state["resumption_token"] == "opaque-next"
    assert [call[1]["verb"] for call in client.calls] == [
        "Identify",
        "ListRecords",
        "ListRecords",
    ]
    assert client.calls[2][1] == {
        "verb": "ListRecords",
        "resumptionToken": "opaque-next",
    }


def test_completed_bootstrap_is_a_network_free_noop(tmp_path: Path) -> None:
    client = QueueClient(identify(), no_records())
    database = Database(tmp_path / "store")
    database.initialize()
    workflow = ArxivBootstrap(database, adapter(client), extractor="fixture")

    first = workflow.run(max_pages=1)
    second = workflow.run(max_pages=1)

    assert first.status == "complete"
    assert second.status == "complete"
    assert second.already_complete is True
    assert second.run_id is None
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (identify(base_url="https://example.test/oai"), "baseURL does not match"),
        (identify(granularity="YYYY"), "unsupported OAI-PMH datestamp granularity"),
        (
            identify().replace("<protocolVersion>2.0", "<protocolVersion>1.1"),
            "unsupported OAI-PMH protocol version",
        ),
        (
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            '<error code="badArgument">bad</error></OAI-PMH>',
            "OAI-PMH Identify error",
        ),
    ],
)
def test_identify_rejects_inconsistent_or_unsupported_repositories(
    body: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adapter(QueueClient(body)).identify()


def test_bootstrap_rejects_corrupt_completion_or_boundary_state() -> None:
    source = ArxivBootstrapSource(adapter(QueueClient()))
    valid = {
        "bootstrap": {
            "version": 1,
            "adapter_checkpoint_signature": source.adapter.checkpoint_signature,
            "repository": {
                "base_url": source.adapter.url,
                "earliest_datestamp": "1991-08-14",
            },
            "window_start": "1991-08-14",
            "window_end": "2026-08-31",
            "discovered_at": "2026-09-01T12:00:00Z",
        },
        "bootstrap_complete": True,
    }

    with pytest.raises(ValueError, match="bootstrap_complete"):
        source.is_complete({**valid, "bootstrap_complete": "true"})
    with pytest.raises(ValueError, match="lower boundary"):
        source.is_complete(
            {
                **valid,
                "bootstrap": {**valid["bootstrap"], "window_start": "1992-01-01"},
            }
        )
