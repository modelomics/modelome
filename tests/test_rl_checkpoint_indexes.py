from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.rl_checkpoint_indexes import RlClarityCheckpointIndexAdapter

_INDEX = "https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/index.html"


class _PageClient:
    def __init__(self, body: str, *, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "text/html"}, self.body, url)


def test_indexes_all_explicit_supported_rl_clarity_model_handles() -> None:
    levels = (100, 300, 1000, 3000, 10000, 30000, 100000)
    procgen = (
        "coinrun", "starpilot", "caveflyer", "dodgeball", "fruitbot", "chaser",
        "miner", "jumper", "leaper", "maze", "bigfish", "heist", "climber",
        "plunder", "ninja", "bossfight",
    )
    handles = ["coinrun"]
    handles.extend(
        f"finite_levels/run_{run}/coinrun_{levels_count}"
        for run in (1, 2)
        for levels_count in levels
    )
    handles.extend(("edit/coinrun_saw_edit", "edit/coinrun_enemy_edit"))
    handles.extend(f"procgen/{name}" for name in procgen)
    handles.extend(f"impala/coinrun_ablation_{index}" for index in range(8))
    # Match the source index's explicit model paths embedded in a list page.
    html = "<ul>" + "".join(f"<li>{handle}</li>" for handle in handles)
    html += "<li>coinrun_procgen</li></ul>"
    client = _PageClient(html)
    adapter = RlClarityCheckpointIndexAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == len(handles) == 41
    assert client.calls == [_INDEX]
    assert {record.identifiers[0].value for record in page.records} == set(handles)
    by_handle = {record.identifiers[0].value: record for record in page.records}
    assert by_handle["coinrun"].canonical_url.endswith("/coinrun.jd")
    assert by_handle["finite_levels/run_1/coinrun_100"].canonical_url.endswith(
        "/finite_levels/run_1/coinrun_100.jd"
    )
    assert by_handle["edit/coinrun_saw_edit"].links[-1].url.endswith(
        "/edit/coinrun_saw_edit.jd"
    )
    assert by_handle["impala/coinrun_ablation_7"].links[-1].url.endswith(
        "/impala/coinrun_ablation_7.jd"
    )
    assert all(record.kind.value == "weights" for record in page.records)


def test_rejects_duplicate_handles_and_empty_supported_inventory() -> None:
    duplicate = _PageClient("<ul><li>procgen/coinrun</li><li>procgen/coinrun</li></ul>")
    with pytest.raises(ValueError, match="duplicate checkpoint handle"):
        RlClarityCheckpointIndexAdapter(client=duplicate).fetch_page({})

    unrelated = _PageClient("<ul><li>coinrun_procgen</li><li>rollouts/1</li></ul>")
    with pytest.raises(ValueError, match="no recognized checkpoint handles"):
        RlClarityCheckpointIndexAdapter(client=unrelated).fetch_page({})


def test_enforces_entry_limit_and_fails_closed_on_http_error() -> None:
    html = "<ul><li>coinrun</li><li>procgen/coinrun</li></ul>"
    with pytest.raises(ValueError, match="inventory exceeds 1"):
        RlClarityCheckpointIndexAdapter(client=_PageClient(html), max_entries=1).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        RlClarityCheckpointIndexAdapter(client=_PageClient(html, status=503)).fetch_page({})
