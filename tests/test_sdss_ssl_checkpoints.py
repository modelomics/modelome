from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.sdss_ssl_checkpoints import (
    SdssSslCheckpointsSourceAdapter,
    _valid_asset_url,
)

ROOT = (
    "https://portal.nersc.gov/project/dasrepo/"
    "self-supervised-learning-sdss/checkpoints/"
)
FILENAMES = (
    "pretrained_paper_model.pth.tar",
    "photoz_finetuned_model.pth.tar",
    "photoz_supervised_baseline_model.pth.tar",
)


class FakeClient:
    def __init__(self, filenames=FILENAMES) -> None:
        self.filenames = filenames
        self.requested = None

    def get(self, url: str, *, headers=None, params=None) -> HttpResponse:
        self.requested = (url, headers)
        anchors = "".join(f'<a href="{ROOT}{name}">download</a>' for name in self.filenames)
        html = f"<html><title>Pretrained Models</title><body>{anchors}</body></html>"
        return HttpResponse(200, {}, html.encode(), url)


def test_extracts_exact_source_declared_sdss_checkpoint_files():
    client = FakeClient()
    adapter = SdssSslCheckpointsSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert [record.raw["checkpoint_filename"] for record in page.records] == list(FILENAMES)
    assert client.requested[1]["Accept"].startswith("text/html")
    for record in page.records:
        filename = record.raw["checkpoint_filename"]
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.canonical_url == ROOT + filename
        assert record.releases[0].metadata["checkpoint_url"] == ROOT + filename
        assert {link.relation for link in record.links} == {"weights", "source_model_catalog"}
        assert record.releases[0].metadata["catalog_sha256"] == page.next_state["catalog_sha256"]


def test_missing_known_release_fails_closed():
    adapter = SdssSslCheckpointsSourceAdapter(client=FakeClient(FILENAMES[:-1]))

    with pytest.raises(ValueError, match="expected model checkpoint links missing"):
        adapter.fetch_page({})


def test_rejects_noncanonical_asset_hosts_and_paths():
    assert _valid_asset_url("https://evil.example/" + FILENAMES[0]) is None
    assert _valid_asset_url(ROOT + "extra.pth") is None


def test_ignores_checkpoint_files_outside_the_declared_inventory():
    adapter = SdssSslCheckpointsSourceAdapter(
        client=FakeClient((*FILENAMES, "extra.pth.tar"))
    )

    page = adapter.fetch_page({})

    assert len(page.records) == 3
