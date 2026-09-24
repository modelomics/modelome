from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.alphachip_rl_checkpoint import AlphaChipRlCheckpointAdapter

_README = "https://github.com/google-research/circuit_training/blob/main/README.md"
_CHECKPOINT = (
    "https://storage.googleapis.com/rl-infra-public/circuit-training/"
    "tpu_checkpoint_20240815.tar.gz"
)


class _Client:
    def __init__(self, body: str, status: int = 200) -> None:
        self.body = body.encode()
        self.status = status

    def get(self, url: str, *, headers=None) -> HttpResponse:
        return HttpResponse(self.status, {"content-type": "text/html"}, self.body, url)


def test_indexes_only_the_exact_readme_checkpoint_archive() -> None:
    page_html = f"<html><body>Download checkpoint: <code>{_CHECKPOINT}</code></body></html>"
    adapter = AlphaChipRlCheckpointAdapter(
        client=_Client(page_html), clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.canonical_url == _CHECKPOINT
    assert record.models[0].name == "AlphaChip pretrained placement policy"
    assert record.releases[0].metadata["pretraining_description"] == "pre-trained on 20 TPU blocks"
    assert record.links[-1].url == _CHECKPOINT
    assert record.links[-1].crawl is False


def test_fails_closed_if_readme_no_longer_documents_checkpoint_or_is_unavailable() -> None:
    with pytest.raises(ValueError, match="checkpoint URL is missing"):
        AlphaChipRlCheckpointAdapter(client=_Client("<html>changed</html>")).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        AlphaChipRlCheckpointAdapter(client=_Client("", 503)).fetch_page({})
