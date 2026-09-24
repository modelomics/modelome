from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.rl_checkpoints_extra import DiffusionPolicyCheckpointIndexAdapter

_SITE = "https://diffusion-policy.cs.columbia.edu"


class _PageClient:
    def __init__(self, pages: dict[str, str]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        if url not in self.pages:
            return HttpResponse(404, {}, b"", url)
        return HttpResponse(200, {}, self.pages[url].encode(), url)


def _tree(root: str, *, score: str) -> dict[str, str]:
    base = f"{_SITE}/data/experiments/{root}/"
    task = base + "pusht/"
    method = task + "diffusion_policy_cnn/"
    run = method + "train_0/"
    checkpoints = run + "checkpoints/"
    return {
        base: '<a href="pusht/">pusht/</a>',
        task: '<a href="diffusion_policy_cnn/">method</a>',
        method: '<a href="train_0/">train_0/</a><a href="train_latest/">bad run</a>',
        run: '<a href="checkpoints/">checkpoints/</a><a href="metrics/">metrics/</a>',
        checkpoints: (
            '<a href="epoch=0550-test_mean_score=' + score + '.ckpt">best</a>'
            '<a href="latest.ckpt">latest</a>'
            '<a href="https://elsewhere.test/epoch=1-test_mean_score=9.ckpt">external</a>'
            '<a href="notes.txt">notes</a>'
        ),
    }


def test_indexes_only_named_best_checkpoints_from_declared_roots() -> None:
    client = _PageClient({**_tree("image", score="0.969"), **_tree("low_dim", score="0.812")})
    adapter = DiffusionPolicyCheckpointIndexAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert len(client.calls) == 10
    image, low_dim = page.records
    assert image.identifiers[0].value == (
        "image/pusht/diffusion_policy_cnn/train_0/checkpoints/"
        "epoch=0550-test_mean_score=0.969.ckpt"
    )
    assert low_dim.identifiers[0].value.endswith("epoch=0550-test_mean_score=0.812.ckpt")
    assert image.releases[0].metadata["test_mean_score"] == "0.969"
    assert image.links[-1].url.endswith("epoch=0550-test_mean_score=0.969.ckpt")
    assert all("latest.ckpt" not in link.url for link in image.links)


def test_fails_closed_when_an_index_is_unavailable() -> None:
    client = _PageClient({})
    with pytest.raises(ValueError, match="HTTP 404"):
        DiffusionPolicyCheckpointIndexAdapter(client=client).fetch_page({})


def test_directory_bound_is_enforced() -> None:
    client = _PageClient(_tree("image", score="0.969"))
    with pytest.raises(ValueError, match="directory index exceeds"):
        DiffusionPolicyCheckpointIndexAdapter(client=client, max_directories=1).fetch_page({})
