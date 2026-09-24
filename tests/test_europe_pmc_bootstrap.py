from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from modelome.europe_pmc_bootstrap import (
    EuropePmcBootstrap,
    run_europe_pmc_bootstrap,
)
from modelome.http import HttpResponse
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.storage import Database

NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)


class QueueClient:
    def __init__(self, *payloads: Mapping[str, Any]) -> None:
        self.payloads = list(payloads)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        payload = self.payloads.pop(0)
        return HttpResponse(
            200,
            {"content-type": "application/json"},
            json.dumps(payload).encode(),
            url,
        )


def empty_page():
    return {"version": "6.9", "hitCount": 0}


def adapter(client: QueueClient) -> EuropePmcSourceAdapter:
    return EuropePmcSourceAdapter(client=client, clock=lambda: NOW)


def database(tmp_path: Path) -> Database:
    result = Database(tmp_path / "store")
    result.initialize()
    return result


def test_runner_persists_month_progress_and_skips_completed_bootstrap(tmp_path: Path) -> None:
    client = QueueClient(empty_page(), empty_page())
    source = adapter(client)
    store = database(tmp_path)

    first = run_europe_pmc_bootstrap(
        store,
        source,
        earliest_update_date=date(2026, 8, 31),
        max_pages=1,
    )
    saved = store.get_source_state("europe-pmc:bootstrap")
    second = run_europe_pmc_bootstrap(
        store,
        source,
        earliest_update_date=date(2026, 8, 31),
        max_pages=1,
    )
    third = EuropePmcBootstrap(store, source, earliest_update_date=date(2026, 8, 31)).run(
        max_pages=1
    )

    assert first.status == "partial"
    assert saved["bootstrap"]["chunk_start"] == "2026-09-01"
    assert second.status == "complete"
    assert not second.already_complete
    assert third.status == "complete" and third.already_complete
    assert len(client.calls) == 2
    assert [call[1]["query"] for call in client.calls] == [
        "UPDATE_DATE:[2026-08-31 TO 2026-08-31]",
        "UPDATE_DATE:[2026-09-01 TO 2026-09-01]",
    ]


def test_runner_requires_a_finite_positive_page_budget(tmp_path: Path) -> None:
    workflow = EuropePmcBootstrap(database(tmp_path), adapter(QueueClient()))
    try:
        workflow.run(max_pages=0)
    except ValueError as error:
        assert "finite, positive" in str(error)
    else:
        raise AssertionError("zero-page budget was accepted")
