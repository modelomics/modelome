from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.molmobot_policy_collection import MolmoBotPolicyCollectionAdapter

_COLLECTION = {
    "title": "MolmoBot-Models",
    "private": False,
    "owner": {"name": "allenai"},
    "items": [
        {
            "id": "allenai/MolmoBot-DROID",
            "author": "allenai",
            "type": "model",
            "repoType": "model",
            "private": False,
            "note": {"text": "Flagship VLA using two observations"},
            "lastModified": "2026-03-21T00:35:25Z",
        },
        {
            "id": "allenai/MolmoBot-SPOC-DROID",
            "author": "allenai",
            "type": "model",
            "repoType": "model",
            "private": False,
            "note": {"text": "Lightweight Franka policy"},
            "lastModified": "2026-03-24T23:50:00Z",
        },
    ],
}
_REVISIONS = {
    "allenai/MolmoBot-DROID": "a" * 40,
    "allenai/MolmoBot-SPOC-DROID": "b" * 40,
}


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        if url.endswith("/api/collections/allenai/molmobot-models"):
            body: Any = _COLLECTION
        elif "/api/models/" in url and "/tree/" not in url:
            repo = url.removeprefix("https://huggingface.co/api/models/")
            body = {"sha": _REVISIONS[repo]}
        elif "/tree/" in url:
            repo_path = url.removeprefix("https://huggingface.co/api/models/")
            repo = "/".join(repo_path.split("/")[:2])
            filename = (
                "model.pt" if repo.endswith("DROID") and "SPOC" not in repo else "model.safetensors"
            )
            body = [
                {
                    "type": "file",
                    "path": filename,
                    "oid": f"oid-{repo.rsplit('-', 1)[-1]}",
                    "size": 2048,
                    "lfs": {"oid": f"lfs-{repo.rsplit('-', 1)[-1]}"},
                }
            ]
        else:
            raise AssertionError(f"unexpected URL: {url}")
        encoded = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        return HttpResponse(200, {}, encoded, url)


def test_adapter_indexes_collection_models_and_exact_pinned_weights() -> None:
    client = _Client()
    adapter = MolmoBotPolicyCollectionAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert [record.source_record_id for record in page.records] == [
        "molmobot:molmobot-droid",
        "molmobot:molmobot-spoc-droid",
    ]
    for record in page.records:
        repo_id = record.identifiers[0].value
        revision = _REVISIONS[repo_id]
        path = record.releases[0].metadata["weight_path"]
        expected = f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.canonical_url == expected
        assert record.identifiers == (Identifier("molmobot:model", repo_id),)
        assert (
            next(link for link in record.links if link.relation == "model_artifact").url == expected
        )
        assert record.releases[0].metadata["weight_size_bytes"] == 2048
        assert record.releases[0].revision == revision


def test_adapter_rejects_non_a2_or_private_collection_items() -> None:
    class PrivateModelClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            response = super().get(url, params=params, headers=headers)
            if url.endswith("/api/collections/allenai/molmobot-models"):
                payload = json.loads(response.body)
                payload["items"][0]["private"] = True
                return HttpResponse(200, {}, json.dumps(payload).encode(), url)
            return response

    with pytest.raises(ValueError, match="non-public or unexpected item"):
        MolmoBotPolicyCollectionAdapter(client=PrivateModelClient()).fetch_page({})


def test_adapter_rejects_ambiguous_weight_inventory() -> None:
    class MultipleWeightsClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            response = super().get(url, params=params, headers=headers)
            if "/tree/" in url:
                payload = json.loads(response.body)
                payload.append(
                    {"type": "file", "path": "model.safetensors", "oid": "extra", "size": 1}
                )
                return HttpResponse(200, {}, json.dumps(payload).encode(), url)
            return response

    with pytest.raises(ValueError, match="expected one root weight file"):
        MolmoBotPolicyCollectionAdapter(client=MultipleWeightsClient()).fetch_page({})
