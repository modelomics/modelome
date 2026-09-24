from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.phyre_dqn_checkpoints import PhyreDqnCheckpointAdapter

_SCRIPT = (
    "https://raw.githubusercontent.com/facebookresearch/phyre/main/"
    "agents/download_dqn_ckps.sh"
)
_ROOT = "https://dl.fbaipublicfiles.com/phyre/"
_TEMPLATES = (
    "ball_cross_template",
    "ball_within_template",
    "two_balls_cross_template",
    "two_balls_within_template",
)


class _Client:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "text/plain"}, self.body, url)


def _script() -> str:
    templates = " ".join(_TEMPLATES)
    return """#!/bin/bash -e
for seed in $(seq 0 9); do
    for tpl in TEMPLATE_LIST; do
        for fname in ckpt.00100000 results.json; do
            path="results/finals/dqn_10k/$tpl/$seed/$fname"
            wget "https://dl.fbaipublicfiles.com/phyre/$path" -O "$DST_ROOT/$path"
        done
    done
done
""".replace("TEMPLATE_LIST", templates)


def test_indexes_the_finite_40_checkpoint_map_only() -> None:
    client = _Client(_script())
    adapter = PhyreDqnCheckpointAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_SCRIPT]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 40
    assert {record.canonical_url for record in page.records} == {
        f"{_ROOT}results/finals/dqn_10k/{template}/{seed}/ckpt.00100000"
        for template in _TEMPLATES
        for seed in range(10)
    }
    assert len({record.identifiers[0].value for record in page.records}) == 40
    assert all(record.kind.value == "weights" for record in page.records)
    assert all(record.links[-1].crawl is False for record in page.records)
    assert all(
        record.releases[0].metadata["algorithm"] == "DQN" for record in page.records
    )
    assert all(
        "results.json" not in record.canonical_url for record in page.records
    )


def test_fails_closed_if_upstream_download_contract_changes() -> None:
    changed = _script().replace("two_balls_within_template", "other_template")
    with pytest.raises(ValueError, match="inventory contract changed"):
        PhyreDqnCheckpointAdapter(client=_Client(changed)).fetch_page({})


def test_rejects_http_errors_and_oversized_script() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        PhyreDqnCheckpointAdapter(client=_Client("", status=503)).fetch_page({})
    with pytest.raises(ValueError, match="exceeds response limit"):
        PhyreDqnCheckpointAdapter(
            client=_Client(_script()), max_response_bytes=10
        ).fetch_page({})
