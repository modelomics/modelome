from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.flower_vla_collection import FlowerVLACollectionAdapter

_COLLECTION = "https://huggingface.co/api/collections/mbreuss/flower-vla"
_REPO = "mbreuss/flower_libero_10"
_REVISION = "0123456789abcdef0123456789abcdef01234567"


class _Client:
    def __init__(self) -> None:
        self.payload: dict[str, Any] = {
            "title": "FLOWER VLA",
            "private": False,
            "owner": {"name": "mbreuss"},
            "items": [
                {
                    "id": _REPO,
                    "type": "model",
                    "repoType": "model",
                    "private": False,
                    "author": "mbreuss",
                },
                {"id": "2509.04996", "type": "paper"},
            ],
        }

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        del params, headers
        if url == _COLLECTION:
            payload: Any = self.payload
        elif url == f"https://huggingface.co/api/models/{_REPO}":
            payload = {"sha": _REVISION}
        elif "/tree/" in url:
            payload = [
                {
                    "type": "file",
                    "path": "model.safetensors",
                    "oid": "a" * 40,
                    "size": 4096,
                }
            ]
        else:
            raise AssertionError(f"unexpected URL: {url}")
        body = json.dumps(payload).encode()
        return HttpResponse(200, {}, body, url)


def test_indexes_collection_model_and_exact_pinned_weight() -> None:
    adapter = FlowerVLACollectionAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    expected = f"https://huggingface.co/{_REPO}/resolve/{_REVISION}/model.safetensors"
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == expected
    assert record.identifiers[0].value == _REPO
    assert record.releases[0].revision == _REVISION
    assert record.releases[0].metadata["weight_oid"] == "a" * 40
    assert record.releases[0].metadata["weight_size_bytes"] == 4096
    assert FlowerVLACollectionAdapter(client=_Client()).checkpoint_signature == (
        adapter.checkpoint_signature
    )


def test_rejects_private_models_and_ambiguous_weight_files() -> None:
    client = _Client()
    client.payload["items"][0]["private"] = True
    with pytest.raises(ValueError, match="unexpected or non-public"):
        FlowerVLACollectionAdapter(client=client).fetch_page({})

    class AmbiguousClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            response = super().get(url, params=params, headers=headers)
            if "/tree/" in url:
                payload = json.loads(response.body)
                payload.append(
                    {
                        "type": "file",
                        "path": "360000_model_weights.pt",
                        "oid": "b" * 40,
                        "size": 8192,
                    }
                )
                return HttpResponse(200, {}, json.dumps(payload).encode(), url)
            return response

    with pytest.raises(ValueError, match="expected one recognized root weight"):
        FlowerVLACollectionAdapter(client=AmbiguousClient()).fetch_page({})
