from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class _Client:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(dict(params or {}))
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.payload).encode(),
            url=url,
        )


class _RouteClient:
    def __init__(self, routes: dict[str, tuple[object, dict[str, str]]]) -> None:
        self.routes = routes
        self.calls: list[str] = []
        self.params_calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.params_calls.append(dict(params or {}))
        if params:
            url = f"{url}?catalog=1"
        self.calls.append(url)
        payload, response_headers = self.routes[url]
        return HttpResponse(
            status=200,
            headers=response_headers,
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_huggingface_records_additional_checkpoint_formats_and_shard_indexes() -> None:
    filenames = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "model.msgpack",
        "model.flax",
        "model.ggml",
        "model.tflite",
        "model.mlpackage",
        "bert_model.ckpt.index",
        "bert_model.ckpt.data-00000-of-00001",
        "variables.index",
        "variables.data-00000-of-00001",
        "unpaired.index",
        "unpaired-shard.data-00000-of-00001",
        "metadata.json",
    ]
    client = _Client(
        [
            {
                "id": "lab/multiformat-model",
                "sha": "abc123",
                "gated": True,
                "config": {"source": "https://github.com/lab/architecture"},
                "siblings": [{"rfilename": name} for name in filenames],
            }
        ]
    )
    page = HuggingFaceSourceAdapter(client=client).fetch_page({})

    record = page.records[0]
    weights = {
        link.url.split("/resolve/abc123/", maxsplit=1)[-1]
        for link in record.links
        if link.relation == "weights"
    }
    assert weights == set(filenames) - {
        "metadata.json",
        "unpaired.index",
        "unpaired-shard.data-00000-of-00001",
    }
    assert record.releases[0].metadata["weight_files"] == sorted(weights)
    assert all(link.crawl is False for link in record.links if link.relation == "weights")
    assert record.raw["gated"] is True
    assert (
        "https://github.com/lab/architecture",
        "code_reference",
        "$.config",
    ) in {(link.url, link.relation, link.locator) for link in record.links}
    assert client.calls[0] == {
        "limit": 100,
        "full": "true",
        "cardData": "true",
        "config": "true",
        "sort": "lastModified",
        "direction": -1,
    }
    assert not {"gated", "private", "filter", "author", "search"} & set(client.calls[0])


def test_huggingface_links_exact_dataset_identifiers_from_model_card_metadata() -> None:
    client = _Client(
        [
            {
                "id": "lab/model-trained-on-data",
                "cardData": {
                    "datasets": [
                        "stanfordnlp/imdb",
                        "dataset-without-namespace",
                        "owner/../not-a-repo",
                    ]
                },
            }
        ]
    )

    record = HuggingFaceSourceAdapter(client=client).fetch_page({}).records[0]

    dataset_links = [link for link in record.links if link.relation == "dataset"]
    assert [(link.url, link.locator) for link in dataset_links] == [
        (
            "https://huggingface.co/datasets/stanfordnlp/imdb",
            "$.cardData.datasets[0]",
        )
    ]
    assert Identifier("huggingface:dataset", "stanfordnlp/imdb") in record.identifiers


def test_huggingface_revision_enrichment_is_checkpointed_and_does_not_fetch_blobs() -> None:
    catalog_url = "https://huggingface.co/api/models?catalog=1"
    repo = "lab/historical-model"
    refs_url = f"https://huggingface.co/api/models/{repo}/refs"
    branch_url = f"https://huggingface.co/api/models/{repo}/commits/refs%2Fheads%2Fmain"
    branch_next = f"{branch_url}?cursor=more"
    tag_url = f"https://huggingface.co/api/models/{repo}/commits/refs%2Ftags%2Fv1"
    routes = {
        catalog_url: ([{"id": repo, "sha": "head-sha"}], {}),
        refs_url: (
            {
                "branches": [{"name": "main", "ref": "refs/heads/main"}],
                "tags": [{"name": "v1", "ref": "refs/tags/v1"}],
                "converts": [],
            },
            {},
        ),
        branch_url: (
            [{"id": "commit-1", "date": "2025-01-01T00:00:00Z", "title": "first"}],
            {"Link": f'<{branch_next}>; rel="next"'},
        ),
        branch_next: ([{"id": "commit-2", "date": "2025-01-02T00:00:00Z"}], {}),
        tag_url: ([{"id": "commit-1", "date": "2025-01-01T00:00:00Z"}], {}),
    }
    client = _RouteClient(routes)
    adapter = HuggingFaceSourceAdapter(client=client, include_revisions=True)

    catalog_page = adapter.fetch_page({})
    assert [record.source_record_id for record in catalog_page.records] == [repo]
    assert catalog_page.complete is False
    assert len(client.calls) == 1

    state = catalog_page.next_state
    refs_page = adapter.fetch_page(state)
    assert refs_page.records == ()
    assert len(client.calls) == 2

    first_commit_page = adapter.fetch_page(refs_page.next_state)
    assert [record.source_record_id for record in first_commit_page.records] == [f"{repo}@commit-1"]
    assert len(client.calls) == 3
    continuation_page = adapter.fetch_page(first_commit_page.next_state)
    assert [record.source_record_id for record in continuation_page.records] == [f"{repo}@commit-2"]
    assert len(client.calls) == 4
    tag_commit_page = adapter.fetch_page(continuation_page.next_state)
    assert [record.source_record_id for record in tag_commit_page.records] == [f"{repo}@commit-1"]
    assert tag_commit_page.complete is True
    assert len(client.calls) == 5

    historical = tag_commit_page.records[0]
    assert historical.releases[0].revision == "commit-1"
    assert historical.releases[0].identifiers[0].value == f"{repo}@commit-1"
    assert "resolve" not in " ".join(link.url for link in historical.links)


def test_huggingface_revision_queue_returns_to_unfinished_model_listing() -> None:
    repo = "lab/one-model"
    catalog_url = "https://huggingface.co/api/models?catalog=1"
    catalog_next = "https://huggingface.co/api/models?cursor=page-2"
    refs_url = f"https://huggingface.co/api/models/{repo}/refs"
    client = _RouteClient(
        {
            catalog_url: (
                [{"id": repo}],
                {"Link": f'<{catalog_next}>; rel="next"'},
            ),
            refs_url: ({"branches": [], "tags": [], "converts": []}, {}),
            catalog_next: ([{"id": "lab/second-model"}], {}),
        }
    )
    adapter = HuggingFaceSourceAdapter(client=client, include_revisions=True)

    first_catalog_page = adapter.fetch_page({})
    revisions_finished = adapter.fetch_page(first_catalog_page.next_state)
    assert revisions_finished.records == ()
    assert revisions_finished.complete is False
    assert revisions_finished.next_state["next_url"] == catalog_next
    assert "revision_queue" not in revisions_finished.next_state

    second_catalog_page = adapter.fetch_page(revisions_finished.next_state)
    assert [record.source_record_id for record in second_catalog_page.records] == [
        "lab/second-model"
    ]
    assert second_catalog_page.complete is False


def test_huggingface_created_at_sweep_recovers_old_modified_public_models() -> None:
    repo = "lab/long-lived-model"
    catalog_url = "https://huggingface.co/api/models?catalog=1"
    next_url = "https://huggingface.co/api/models?cursor=created-2"
    client = _RouteClient(
        {
            catalog_url: (
                [
                    {
                        "id": repo,
                        "sha": "new-head",
                        "createdAt": "2019-01-01T00:00:00Z",
                        "lastModified": "2019-01-02T00:00:00Z",
                        "disabled": True,
                    }
                ],
                {"Link": f'<{next_url}>; rel="next"'},
            ),
            next_url: ([{"id": "lab/recreated-model", "sha": "new-instance"}], {}),
        }
    )
    now = [datetime(2026, 9, 23, tzinfo=UTC)]
    adapter = HuggingFaceSourceAdapter(
        client=client,
        overlap_days=2,
        created_at_sweep_interval_days=30,
        clock=lambda: now[0],
    )

    first = adapter.fetch_page({"watermark": "2026-09-20T00:00:00Z"})
    assert [record.source_record_id for record in first.records] == [repo]
    assert first.records[0].raw["disabled"] is True
    assert first.complete is False
    assert first.next_state["coverage_mode"] == "created_at_sweep"
    assert client.params_calls[0]["sort"] == "createdAt"
    assert client.params_calls[0]["direction"] == 1

    second = adapter.fetch_page(first.next_state)
    assert [record.source_record_id for record in second.records] == ["lab/recreated-model"]
    assert second.complete is True
    assert second.next_state["created_at_sweep_completed_at"] == "2026-09-23T00:00:00Z"
    # Preserve the lastModified cursor watermark; createdAt has a distinct role.
    assert second.next_state["watermark"] == "2026-09-20T00:00:00Z"

    # The next run is back on the normal incremental listing, not another census.
    adapter.fetch_page(second.next_state)
    assert client.params_calls[-1]["sort"] == "lastModified"
    assert client.params_calls[-1]["direction"] == -1

    now[0] += timedelta(days=31)
    adapter.fetch_page(second.next_state)
    assert client.params_calls[-1]["sort"] == "createdAt"
    assert client.params_calls[-1]["direction"] == 1
