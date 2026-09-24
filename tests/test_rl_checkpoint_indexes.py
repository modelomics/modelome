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


def test_indexes_only_explicit_supported_rl_clarity_model_handles() -> None:
    html = """<ul>
      <li>coinrun</li>
      <li>finite_levels/run_1/coinrun_100</li>
      <li>procgen/coinrun</li>
      <li>impala/coinrun_ablation_0</li>
      <li>coinrun_procgen (procedural assets)</li>
      <li>unrelated item</li>
    </ul>"""
    client = _PageClient(html)
    adapter = RlClarityCheckpointIndexAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert client.calls == [_INDEX]
    assert [record.identifiers[0].value for record in page.records] == [
        "coinrun",
        "finite_levels/run_1/coinrun_100",
        "impala/coinrun_ablation_0",
        "procgen/coinrun",
    ]
    assert page.records[0].canonical_url.endswith("/coinrun.jd")
    assert page.records[1].canonical_url.endswith("/finite_levels/run_1/coinrun_100.jd")
    assert page.records[2].links[-1].url.endswith("/impala/coinrun_ablation_0.jd")
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
