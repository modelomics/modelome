from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.figshare_model_candidates_workflow import (
    FigshareModelCandidatesWorkflow,
)

OAI = "https://api.figshare.com/v2/oai"


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        return self.responses.pop(0)


def response(body: str) -> HttpResponse:
    return HttpResponse(200, {}, body.encode(), OAI)


def page(token: str = "") -> str:
    token_xml = (
        f'<resumptionToken expirationDate="2026-09-25T11:00:00Z">{token}</resumptionToken>'
        if token
        else ""
    )
    return (
        '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
        f"<ListRecords>{token_xml}</ListRecords></OAI-PMH>"
    )


def test_window_workflow_advances_only_after_opaque_token_chain_completes() -> None:
    client = Client(
        response(page("opaque")),
        response(page()),
        response(
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            '<error code="noRecordsMatch">empty</error></OAI-PMH>'
        ),
    )
    workflow = FigshareModelCandidatesWorkflow(
        from_date="2022-02-07",
        until_date="2022-02-09",
        window_days=1,
        client=client,
        clock=lambda: datetime(2026, 9, 25, 10, 0, tzinfo=UTC),
    )

    first = workflow.fetch_page({})
    second = workflow.fetch_page(first.next_state)
    third = workflow.fetch_page(second.next_state)
    fourth = workflow.fetch_page(third.next_state)

    assert [call[1] for call in client.calls] == [
        {
            "verb": "ListRecords",
            "metadataPrefix": "mets",
            "from": "2022-02-07",
            "until": "2022-02-08",
        },
        {"verb": "ListRecords", "resumptionToken": "opaque"},
        {
            "verb": "ListRecords",
            "metadataPrefix": "mets",
            "from": "2022-02-08",
            "until": "2022-02-09",
        },
    ]
    assert first.complete is False
    assert first.next_state["cursor"] == "2022-02-07"
    assert first.next_state["window_state"]["resumption_token"] == "opaque"
    assert second.complete is False
    assert second.next_state["cursor"] == "2022-02-08"
    assert second.next_state["window_state"] == {}
    assert third.complete is True
    assert third.next_state["cursor"] == "2022-02-09"
    assert third.next_state["records_seen"] == 0
    assert fourth.complete is True
    assert fourth.records == ()
    assert fourth.next_state == third.next_state
    assert len(client.calls) == 3


def test_window_workflow_is_bounded_and_checkpoint_bound_to_range() -> None:
    with pytest.raises(ValueError, match="between 1 and 31"):
        FigshareModelCandidatesWorkflow(
            from_date="2020-01-01", until_date="2021-01-01", window_days=32
        )

    workflow = FigshareModelCandidatesWorkflow(from_date="2020-01-01", until_date="2020-01-02")
    other = FigshareModelCandidatesWorkflow(from_date="2020-01-02", until_date="2020-01-03")
    with pytest.raises(ValueError, match="does not match"):
        other.fetch_page(
            {
                "workflow_signature": workflow.checkpoint_signature,
                "cursor": "2020-01-01",
                "window_state": {},
            }
        )


def test_near_expiry_token_restarts_same_date_window_without_skipping() -> None:
    client = Client(
        response(
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            '<ListRecords><resumptionToken expirationDate="2026-09-25T10:04:00Z">'
            "expiring-token</resumptionToken></ListRecords></OAI-PMH>"
        ),
        response(
            '<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            '<error code="noRecordsMatch">empty after restart</error></OAI-PMH>'
        ),
    )
    workflow = FigshareModelCandidatesWorkflow(
        from_date="2022-02-07",
        until_date="2022-02-08",
        client=client,
        clock=lambda: datetime(2026, 9, 25, 10, 0, tzinfo=UTC),
    )

    first = workflow.fetch_page({})
    restarted = workflow.fetch_page(first.next_state)

    assert [call[1] for call in client.calls] == [
        {
            "verb": "ListRecords",
            "metadataPrefix": "mets",
            "from": "2022-02-07",
            "until": "2022-02-08",
        },
        {
            "verb": "ListRecords",
            "metadataPrefix": "mets",
            "from": "2022-02-07",
            "until": "2022-02-08",
        },
    ]
    assert first.next_state["cursor"] == "2022-02-07"
    assert restarted.complete is True
    assert restarted.next_state["cursor"] == "2022-02-08"
    assert restarted.next_state["token_window_restarts"] == 1
