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
        route = self.routes[url]
        status, payload = route if isinstance(route, tuple) else (200, route)
        return HttpResponse(
            status=status,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


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


def test_multi_repo_pilot_keeps_later_repositories_queued_across_budget_boundary(
    tmp_path,
) -> None:
    repos = ("lab/first-model", "lab/second-model")
    routes: dict[str, object] = {}
    for repo in repos:
        routes.update(_routes(repo))
    client = _RouteClient(routes)
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=repos, client=client)
    database = Database(tmp_path / "multi-repo-pilot-store")
    database.initialize()

    first = run_huggingface_revision_pilot(database, source, request_budget=4)
    assert first.status == "partial"
    assert len(client.calls) == 4
    state = database.get_source_state(source.name)
    assert [item["model_id"] for item in state["revision_queue"]] == [repos[1]]
    assert all(repos[0] in url for url in client.calls)

    resumed = run_huggingface_revision_pilot(database, source, request_budget=4)
    assert resumed.status == "complete"
    assert len(client.calls) == 8
    assert all(repos[1] in url for url in client.calls[4:])
    revisions = {
        row["revision"] for row in database.table_rows("model_releases")
    }
    assert revisions == {"head-sha", "historic-sha"}
    assert database.stats()["artifacts"] == 4


def test_inaccessible_repo_is_quarantined_without_blocking_later_pilot_repositories(
    tmp_path,
) -> None:
    repos = ("lab/a-gated-model", "lab/b-live-model")
    routes = _routes(repos[1])
    routes[f"https://huggingface.co/api/models/{repos[0]}/refs"] = (404, {})
    client = _RouteClient(routes)
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=repos, client=client)
    database = Database(tmp_path / "inaccessible-repo-pilot-store")
    database.initialize()

    outcome = run_huggingface_revision_pilot(database, source, request_budget=5)

    assert outcome.status == "complete"
    assert len(client.calls) == 5
    assert client.calls[0].endswith("/lab/a-gated-model/refs")
    assert all(repos[1] in url for url in client.calls[1:])
    issue = next(
        row
        for row in database.table_rows("dead_letters")
        if row["source_record_id"] == f"{repos[0]}:revision-history"
    )
    summary = json.loads(issue["summary_json"])
    assert summary["revision_history_status"] == "incomplete"
    assert summary["http_status"] == 404
    assert len(database.table_rows("model_releases")) == 2


def test_inaccessible_ref_history_advances_to_other_refs_and_repositories(tmp_path) -> None:
    repos = ("lab/a-deleted-ref", "lab/b-live-model")
    routes = _routes(repos[0])
    routes.update(_routes(repos[1]))
    routes[f"https://huggingface.co/api/models/{repos[0]}/refs"] = {
        "branches": [
            {"ref": "refs/heads/deleted", "target_commit": "gone"},
            {"ref": "refs/heads/main", "target_commit": "live"},
        ],
        "tags": [],
        "converts": [],
    }
    routes[f"https://huggingface.co/api/models/{repos[0]}/commits/refs%2Fheads%2Fdeleted"] = (
        404,
        {},
    )
    routes[f"https://huggingface.co/api/models/{repos[0]}/commits/refs%2Fheads%2Fmain"] = [
        {"id": "live", "date": "2025-01-01T00:00:00Z", "title": "Live ref"}
    ]
    live_tree = (
        f"https://huggingface.co/api/models/{repos[0]}"
        "/tree/live?recursive=true&expand=false"
    )
    routes[live_tree] = [{"type": "file", "path": "weights.safetensors"}]
    client = _RouteClient(routes)
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=repos, client=client)
    database = Database(tmp_path / "deleted-ref-pilot-store")
    database.initialize()

    outcome = run_huggingface_revision_pilot(database, source, request_budget=8)

    assert outcome.status == "complete"
    assert "refs%2Fheads%2Fmain" in client.calls[2]
    assert any(repos[1] in url for url in client.calls)
    revision_ids = {
        row["revision"] for row in database.table_rows("model_releases")
    }
    assert "live" in revision_ids
    assert "head-sha" in revision_ids
    issues = database.list_dead_letters()
    assert any("commits endpoint is unavailable (HTTP 404)" in issue["error"] for issue in issues)


def test_inaccessible_ref_issue_and_later_repos_resume_across_request_budget(
    tmp_path,
) -> None:
    repos = ("lab/a-mixed-model", "lab/b-live-model")
    first_base = f"https://huggingface.co/api/models/{repos[0]}"
    routes = _routes(repos[1])
    routes[f"{first_base}/refs"] = {
        "branches": [
            {"ref": "refs/heads/deleted", "target_commit": "gone"},
            {"ref": "refs/heads/main", "target_commit": "live"},
        ],
        "tags": [],
        "converts": [],
    }
    routes[f"{first_base}/commits/refs%2Fheads%2Fdeleted"] = (404, {})
    routes[f"{first_base}/commits/refs%2Fheads%2Fmain"] = [
        {"id": "live", "date": "2025-01-01T00:00:00Z", "title": "Live ref"}
    ]
    routes[f"{first_base}/tree/live?recursive=true&expand=false"] = [
        {"type": "file", "path": "weights.safetensors"}
    ]
    client = _RouteClient(routes)
    source = HuggingFaceRevisionPilotSourceAdapter(repo_ids=repos, client=client)
    database = Database(tmp_path / "mixed-ref-resume-store")
    database.initialize()

    first = run_huggingface_revision_pilot(database, source, request_budget=2)
    assert first.status == "partial"
    assert client.calls == [
        f"{first_base}/refs",
        f"{first_base}/commits/refs%2Fheads%2Fdeleted",
    ]
    state = database.get_source_state(source.name)
    assert state["revision_queue"][0]["model_id"] == repos[0]
    assert state["revision_queue"][0]["ref_index"] == 1
    issue = next(
        row
        for row in database.list_dead_letters()
        if row["source_record_id"] == f"{repos[0]}:revision-history"
    )
    assert "commits endpoint is unavailable (HTTP 404)" in issue["error"]

    resumed = run_huggingface_revision_pilot(database, source, request_budget=6)
    assert resumed.status == "complete"
    assert len(client.calls) == 8
    assert client.calls[2] == f"{first_base}/commits/refs%2Fheads%2Fmain"
    assert any(repos[1] in url for url in client.calls[4:])
    release_sources = {
        row["source_record_id"] for row in database.table_rows("artifacts")
    }
    assert f"{repos[0]}@live" in release_sources
    assert f"{repos[1]}@historic-sha" in release_sources


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
