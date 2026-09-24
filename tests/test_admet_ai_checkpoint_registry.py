from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.admet_ai_checkpoint_registry import (
    ADMETAICheckpointRegistrySourceAdapter,
)

REVISION = "c" * 40
TREE_SHA = "d" * 40
REPOSITORY = "swansonk14/admet_ai"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Any]] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append((url, params))
        return self.responses.pop(0)


def response(payload: Any, url: str) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(payload).encode(), url)


def fixture_tree() -> list[dict[str, Any]]:
    return [
        {
            "path": f"admet_ai/resources/models/{family}/model_{index}.pt",
            "mode": "100644",
            "type": "blob",
            "sha": str(index) * 40,
            "size": 128,
        }
        for family in ("admet_classification", "admet_regression")
        for index in range(5)
    ]


def test_indexes_the_exact_ten_admet_ai_chemprop_tree_checkpoints() -> None:
    commit_url = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_SHA}"
    client = QueueClient(
        response({"sha": REVISION, "commit": {"tree": {"sha": TREE_SHA}}}, commit_url),
        response({"tree": fixture_tree(), "truncated": False}, tree_url),
    )
    source = ADMETAICheckpointRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 10
    assert len(page.records) == 10
    records = {record.raw["checkpoint_handle"]: record for record in page.records}
    assert set(records) == {
        f"{family}/model_{index}"
        for family in ("admet_classification", "admet_regression")
        for index in range(5)
    }
    model = records["admet_regression/model_3"].models[0]
    assert model.identifiers == (Identifier("admet-ai:checkpoint", "admet_regression/model_3"),)
    rec = records["admet_regression/model_3"]
    expected_url = (
        f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"
        "admet_ai/resources/models/admet_regression/model_3.pt"
    )
    assert rec.raw["weight_url"] == expected_url
    assert rec.releases[0].metadata["ensemble_family"] == "regression"
    assert rec.releases[0].metadata["ensemble_member"] == 3
    assert rec.links[1].url == expected_url
    assert client.calls == [(commit_url, None), (tree_url, {"recursive": "1"})]


def test_skips_tree_request_when_repository_revision_is_unchanged() -> None:
    commit_url = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    client = QueueClient(
        response({"sha": REVISION, "commit": {"tree": {"sha": TREE_SHA}}}, commit_url)
    )
    page = ADMETAICheckpointRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 10}
    )
    assert page.records == ()
    assert page.upstream_count == 10
    assert client.calls == [(commit_url, None)]


def test_rejects_incomplete_expected_checkpoint_inventory() -> None:
    commit_url = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_SHA}"
    client = QueueClient(
        response({"sha": REVISION, "commit": {"tree": {"sha": TREE_SHA}}}, commit_url),
        response({"tree": fixture_tree()[:-1], "truncated": False}, tree_url),
    )
    with pytest.raises(ValueError, match="expected 10 recognized checkpoint files"):
        ADMETAICheckpointRegistrySourceAdapter(client=client).fetch_page({})


def test_ignores_nearby_noncheckpoint_and_unrecognized_tree_paths() -> None:
    commit_url = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    tree_url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{TREE_SHA}"
    tree = fixture_tree() + [
        {
            "path": "admet_ai/resources/models/admet_classification/model_5.pt",
            "type": "blob",
            "sha": "e" * 40,
        },
        {
            "path": "admet_ai/resources/models/admet_classification/README.md",
            "type": "blob",
            "sha": "f" * 40,
        },
    ]
    client = QueueClient(
        response({"sha": REVISION, "commit": {"tree": {"sha": TREE_SHA}}}, commit_url),
        response({"tree": tree, "truncated": False}, tree_url),
    )
    page = ADMETAICheckpointRegistrySourceAdapter(client=client).fetch_page({})
    assert page.upstream_count == 10
