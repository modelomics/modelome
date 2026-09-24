from __future__ import annotations

import json

import pytest

from modelome.http import HttpResponse
from modelome.sources.huggingface_revision_pilot import (
    HuggingFaceRevisionPilotSourceAdapter,
    run_huggingface_revision_pilot,
)
from modelome.storage import Database


class _RouteClient:
    def __init__(self, routes: dict[str, object]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(url)
        payload = self.routes[url]
        return HttpResponse(status=200, headers={}, body=json.dumps(payload).encode(), url=url)


def _routes(repo: str) -> dict[str, object]:
    base = f"https://huggingface.co/api/models/{repo}"
    return {
        f"{base}/refs": {
            "branches": [{"ref": "refs/heads/main", "target_commit": "head-sha"}],
            "tags": [],
            "converts": [],
        },
        f"{base}/commits/refs%2Fheads%2Fmain": [
            {"id": "head-sha", "date": "2025-02-01T00:00:00Z", "title": "Current"},
            {"id": "historic-sha", "date": "2020-01-01T00:00:00Z", "title": "Old weights"},
        ],
        f"{base}/tree/head-sha?recursive=true&expand=false": [],
        f"{base}/tree/historic-sha?recursive=true&expand=false": [
            {"type": "file", "path": "checkpoints/model-v1.safetensors"}
        ],
    }


def test_exact_repo_pilot_resumes_with_a_finite_request_budget_and_finds_old_weights(
    tmp_path,
) -> None:
    repo = "lab/historical-model"
    client = _RouteClient(_routes(repo))
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=[repo], client=client)
    database = Database(tmp_path / "pilot-store")
    database.initialize()

    first = run_huggingface_revision_pilot(database, source, request_budget=2)
    assert first.status == "partial"
    assert len(client.calls) == 2
    state = database.get_source_state(source.name)
    assert state["revision_queue"][0]["tree_queue"]

    resumed = run_huggingface_revision_pilot(database, source, request_budget=3)
    assert resumed.status == "complete"
    assert len(client.calls) == 4
    releases = database.table_rows("model_releases")
    historical = next(row for row in releases if row["revision"] == "historic-sha")
    metadata = json.loads(historical["metadata_json"])
    assert metadata["weight_files"] == ["checkpoints/model-v1.safetensors"]
    assert metadata["weight_files_complete"] is True


@pytest.mark.parametrize("repo_ids", [[], ["owner"], ["https://huggingface.co/a/b"]])
def test_pilot_requires_exact_owner_repository_ids(repo_ids) -> None:
    with pytest.raises(ValueError, match="repo_ids"):
        HuggingFaceRevisionPilotSourceAdapter(repo_ids=repo_ids)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_pilot_requires_a_positive_integer_request_budget(tmp_path, budget) -> None:
    database = Database(tmp_path / "pilot-store")
    database.initialize()
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=["lab/model"])
    with pytest.raises(ValueError, match="request_budget"):
        run_huggingface_revision_pilot(database, source, request_budget=budget)
