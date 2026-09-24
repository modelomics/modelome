from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.aggregator_registry import (
    OpenMLFlowRegistrySourceAdapter,
    OpenMLRunRegistrySourceAdapter,
)


class _Client:
    def __init__(self, *payloads: Any) -> None:
        self.responses = [
            HttpResponse(200, {}, json.dumps(payload).encode(), "https://www.openml.org")
            for payload in payloads
        ]
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected request: {url}")
        return self.responses.pop(0)


def test_openml_registry_pages_flows_as_versioned_implementation_evidence() -> None:
    client = _Client(
        {
            "flows": {
                "flow": [
                    {
                        "id": 42,
                        "full_name": "sklearn.tree.DecisionTreeClassifier",
                        "name": "DecisionTreeClassifier",
                        "external_version": "1.5",
                        "uploader": "alice",
                    },
                    {"id": 43, "name": "sklearn.svm.SVC", "version": 2},
                ]
            }
        },
        {"flows": {"flow": [{"id": 44, "name": "example.Pipeline", "version": "3"}]}},
    )
    adapter = OpenMLFlowRegistrySourceAdapter(page_size=2, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls == [
        "https://www.openml.org/api/v1/json/flow/list/limit/2/offset/0",
        "https://www.openml.org/api/v1/json/flow/list/limit/2/offset/2",
    ]
    assert first.complete is False
    assert first.next_state["offset"] == 2
    assert second.complete is True
    record = first.records[0]
    assert record.kind is ArtifactKind.MODEL_CARD
    assert record.canonical_url == "https://www.openml.org/f/42"
    assert record.models[0].name == "sklearn.tree.DecisionTreeClassifier"
    assert record.models[0].status is ModelStatus.DOCUMENTED
    assert record.models[0].identifiers == (Identifier("openml:flow", "42"),)
    assert record.releases[0].version == "1.5"
    assert record.releases[0].identifiers == (Identifier("openml:flow-release", "42"),)
    assert all(link.crawl is False for link in record.links)


@pytest.mark.parametrize("payload", [[], {"flows": {"error": "bad"}}, {"flows": [None]}])
def test_openml_registry_fails_closed_on_invalid_api_shape(payload: Any) -> None:
    adapter = OpenMLFlowRegistrySourceAdapter(client=_Client(payload))
    with pytest.raises(ValueError, match="response"):
        adapter.fetch_page({})


def test_openml_registry_rejects_row_without_stable_identity_or_name() -> None:
    adapter = OpenMLFlowRegistrySourceAdapter(
        client=_Client({"flows": {"flow": [{"name": "missing id"}]}})
    )
    with pytest.raises(ValueError, match="lacks an ID or name"):
        adapter.fetch_page({})


@pytest.mark.parametrize(
    "payload",
    [
        {"flows": {"flow": []}},
        {"flows": {"@xmlns:oml": "http://openml.org/openml"}},
        {"flows": {"flow": None}},
    ],
)
def test_openml_registry_treats_an_empty_list_page_as_the_terminal_page(payload: Any) -> None:
    adapter = OpenMLFlowRegistrySourceAdapter(client=_Client(payload))

    page = adapter.fetch_page({"offset": 100})

    assert page.complete is True
    assert page.records == ()
    assert page.next_state == {"offset": 100, "completed": True}


def test_openml_run_registry_pages_exact_experiment_ids_without_model_claims() -> None:
    client = _Client(
        {"runs": {"run": [
            {
                "run_id": 73,
                "flow_id": 42,
                "task_id": 6,
                "dataset_id": 31,
                "flow_name": "sklearn.tree.DecisionTreeClassifier",
            },
            {"id": "74", "flow_id": "43", "task_id": "7"},
        ]}},
        {"runs": {"run": [{"id": 75, "flow_id": 42, "task_id": 8, "did": 32}]}},
    )
    adapter = OpenMLRunRegistrySourceAdapter(page_size=2, client=client)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert client.calls == [
        "https://www.openml.org/api/v1/json/run/list/limit/2/offset/0",
        "https://www.openml.org/api/v1/json/run/list/limit/2/offset/2",
    ]
    assert first.complete is False
    assert second.complete is True
    run = first.records[0]
    assert run.source_record_id == "openml-run:73"
    assert run.kind is ArtifactKind.CATALOG_RECORD
    assert run.canonical_url == "https://www.openml.org/r/73"
    assert run.identifiers == (Identifier("openml:run", "73"),)
    assert run.models == ()
    assert run.releases == ()
    assert {(link.url, link.relation, link.crawl) for link in run.links} == {
        ("https://www.openml.org/r/73", "experiment_run", False),
        ("https://www.openml.org/f/42", "evaluated_implementation", False),
        ("https://www.openml.org/t/6", "evaluation_task", False),
        ("https://www.openml.org/d/31", "evaluation_dataset", False),
    }


def test_openml_run_registry_accepts_empty_terminal_page() -> None:
    adapter = OpenMLRunRegistrySourceAdapter(client=_Client({"runs": {"run": []}}))

    page = adapter.fetch_page({"offset": 20})

    assert page.complete is True
    assert page.records == ()
    assert page.next_state == {"offset": 20, "completed": True}


@pytest.mark.parametrize(
    "payload",
    [[], {"runs": {"error": "bad"}}, {"runs": {"run": [None]}}],
)
def test_openml_run_registry_rejects_malformed_run_lists(payload: Any) -> None:
    adapter = OpenMLRunRegistrySourceAdapter(client=_Client(payload))
    with pytest.raises(ValueError, match="response"):
        adapter.fetch_page({})


def test_openml_run_registry_rejects_rows_without_exact_run_id() -> None:
    adapter = OpenMLRunRegistrySourceAdapter(client=_Client({"runs": {"run": [{"flow_id": 42}]}}))
    with pytest.raises(ValueError, match="valid numeric ID"):
        adapter.fetch_page({})

