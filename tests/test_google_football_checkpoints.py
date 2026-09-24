from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.google_football_checkpoints import GoogleFootballCheckpointAdapter

_README = "https://raw.githubusercontent.com/google-research/football/master/README.md"
_ROOT = "https://storage.googleapis.com/gfootball-public-bucket/"
_OBJECTS = {
    "11_vs_11_easy_stochastic": "trained_model_11_vs_11_easy_stochastic",
    "academy_run_to_score_with_keeper": (
        "trained_model_academy_run_to_score_with_keeper_v2"
    ),
}


class _Client:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "text/plain"}, self.body, url)


def _readme() -> str:
    links = "\n".join(
        f"- [{scenario}]({_ROOT}{object_name})"
        for scenario, object_name in _OBJECTS.items()
    )
    return f"# Trained checkpoints\n\n{links}\n"


def test_indexes_only_the_two_exact_readme_checkpoint_objects() -> None:
    client = _Client(_readme())
    adapter = GoogleFootballCheckpointAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_README]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert {record.canonical_url for record in page.records} == {
        f"{_ROOT}{object_name}" for object_name in _OBJECTS.values()
    }
    assert {record.releases[0].metadata["algorithm"] for record in page.records} == {
        "PPO"
    }
    assert all(record.kind.value == "weights" for record in page.records)
    assert all(record.links[-1].crawl is False for record in page.records)


def test_fails_closed_if_a_documented_checkpoint_is_missing_or_changed() -> None:
    with pytest.raises(ValueError, match="inventory is incomplete"):
        GoogleFootballCheckpointAdapter(client=_Client("# Trained checkpoints\n")).fetch_page({})

    changed = _readme().replace(
        f"{_ROOT}{_OBJECTS['11_vs_11_easy_stochastic']}",
        "https://example.com/checkpoint",
    )
    with pytest.raises(ValueError, match="unexpected checkpoint URL"):
        GoogleFootballCheckpointAdapter(client=_Client(changed)).fetch_page({})


def test_rejects_duplicate_expected_scenario_and_http_errors() -> None:
    duplicate = _readme() + (
        f"- [11_vs_11_easy_stochastic]({_ROOT}"
        f"{_OBJECTS['11_vs_11_easy_stochastic']})\n"
    )
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        GoogleFootballCheckpointAdapter(client=_Client(duplicate)).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        GoogleFootballCheckpointAdapter(client=_Client("", status=503)).fetch_page({})
