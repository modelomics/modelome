from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.astronn_gaia_release import (
    AstroNNGaiaReleaseSourceAdapter,
)

REVISION = "a" * 40
DIRECTORIES = (
    "astroNN_no_offset_model",
    "astroNN_constant_model",
    "astroNN_constant_model_reduced",
    "astroNN_multivariate_model",
    "astroNN_multivariate_model_reduced",
)


class FakeClient:
    def __init__(self, *, missing: str | None = None) -> None:
        self.missing = missing
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, *, headers=None, params=None) -> HttpResponse:
        self.calls.append((url, {"headers": headers or {}, "params": params or {}}))
        if "/commits/" in url:
            body = {"sha": REVISION}
        else:
            directories = [name for name in DIRECTORIES if name != self.missing]
            body = {
                "truncated": False,
                "tree": [{"path": name, "type": "tree"} for name in directories]
                + [{"path": "README.md", "type": "blob"}],
            }
        return HttpResponse(200, {}, json.dumps(body).encode(), url)


def test_pinned_tree_emits_five_directory_release_refs_without_file_claims():
    client = FakeClient()
    adapter = AstroNNGaiaReleaseSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert [record.raw["model_directory"] for record in page.records] == list(DIRECTORIES)
    assert all(record.kind is ArtifactKind.PROVIDER_PAGE for record in page.records)
    for record in page.records:
        assert record.raw["revision"] == REVISION
        assert f"/tree/{REVISION}/{record.raw['model_directory']}" in record.canonical_url
        assert record.releases[0].revision == REVISION
        assert record.releases[0].metadata["artifact_locator_kind"] == "repository_directory"
        assert {link.relation for link in record.links} == {"model_release_directory"}
        assert "weight_url" not in record.raw
        assert "checkpoint_url" not in record.raw
    assert client.calls[0][0].endswith("/commits/master")
    assert client.calls[1][0].endswith(f"/git/trees/{REVISION}?recursive=1")


def test_missing_expected_model_directory_fails_closed():
    adapter = AstroNNGaiaReleaseSourceAdapter(client=FakeClient(missing=DIRECTORIES[0]))

    with pytest.raises(ValueError, match="expected model directories absent"):
        adapter.fetch_page({})


def test_completed_revision_only_refreshes_checkpoint_time():
    adapter = AstroNNGaiaReleaseSourceAdapter(client=FakeClient())

    page = adapter.fetch_page({"completed_revision": REVISION, "model_count": 5})

    assert page.records == ()
    assert page.complete
    assert page.next_state["completed_revision"] == REVISION
    assert page.upstream_count == 5
