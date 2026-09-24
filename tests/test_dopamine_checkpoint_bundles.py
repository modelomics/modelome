from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.dopamine_checkpoint_bundles import DopamineCheckpointBundleAdapter

_DOCS = "https://google.github.io/dopamine/docs/"
_ARCHIVES = {
    "dqn": "https://storage.cloud.google.com/download-dopamine-rl/dqn_checkpoints.tar.gz",
    "c51": "https://storage.cloud.google.com/download-dopamine-rl/c51_checkpoints.tar.gz",
    "rainbow": "https://storage.cloud.google.com/download-dopamine-rl/rainbow_checkpoints.tar.gz",
    "iqn": "https://storage.cloud.google.com/download-dopamine-rl/iqn_checkpoints.tar.gz",
}


class _Client:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status

    def get(self, url: str, *, headers=None) -> HttpResponse:
        return HttpResponse(self.status, {"content-type": "text/html"}, self.body, url)


def _html() -> str:
    links = "".join(f'<a href="{url}">{agent}</a>' for agent, url in _ARCHIVES.items())
    return links + '<a href="https://elsewhere.test/dqn_checkpoints.tar.gz">bad</a>'


def test_indexes_exact_documented_checkpoint_bundle_urls() -> None:
    adapter = DopamineCheckpointBundleAdapter(
        client=_Client(_html()), clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert {record.models[0].identifiers[0].value for record in page.records} == set(_ARCHIVES)
    assert {record.canonical_url for record in page.records} == set(_ARCHIVES.values())
    assert all(record.kind.value == "weights" for record in page.records)
    assert all("60 Atari games" in record.text for record in page.records)


def test_fails_closed_if_upstream_documentation_changes_or_is_unavailable() -> None:
    changed_docs = _html().replace("iqn_checkpoints.tar.gz", "other.tar.gz")
    with pytest.raises(ValueError, match="missing checkpoint bundles"):
        DopamineCheckpointBundleAdapter(client=_Client(changed_docs)).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        DopamineCheckpointBundleAdapter(client=_Client(_html(), 503)).fetch_page({})
