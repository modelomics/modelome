from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.starvla_vlact_collection import StarVLAVLActCollectionAdapter

_COLLECTION_URL = (
    "https://huggingface.co/api/collections/StarVLA/vlact-6a903c2e0c176179da425c96"
)
_REPO = "StarVLA/VLAct_Qwen3PI_Robotwin_Finetune"
_REV = "0123456789abcdef0123456789abcdef01234567"
_ITEM = {
    "id": _REPO,
    "type": "model",
    "repoType": "model",
    "private": False,
    "author": "StarVLA",
    "note": {"text": "PI policy fine-tuned on RoboTwin"},
}


class _Client:
    def __init__(self) -> None:
        self.collection: dict[str, Any] = {
            "title": "VLAct",
            "private": False,
            "owner": {"name": "StarVLA"},
            "items": [{"id": "2608.27550", "type": "paper"}, _ITEM],
        }

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        del params, headers
        if url == _COLLECTION_URL:
            payload: Any = self.collection
        elif url == f"https://huggingface.co/api/models/{_REPO}":
            payload = {"sha": _REV}
        elif "/tree/" in url:
            payload = [{
                "type": "file",
                "path": "checkpoints/steps_50000_pytorch_model.pt",
                "oid": "a" * 40,
                "size": 11276440520,
                "lfs": {"oid": "sha256:" + "b" * 64},
            }]
        else:
            raise AssertionError(f"unexpected URL {url}")
        body = json.dumps(payload).encode()
        return HttpResponse(200, {}, body, url)


def test_indexes_exact_public_collection_checkpoint() -> None:
    adapter = StarVLAVLActCollectionAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    expected = f"https://huggingface.co/{_REPO}/resolve/{_REV}/checkpoints/steps_50000_pytorch_model.pt"
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == expected
    assert record.identifiers[0].value == _REPO
    assert record.releases[0].revision == _REV
    assert record.releases[0].metadata["weight_size_bytes"] == 11276440520
    assert next(link for link in record.links if link.relation == "model_artifact").url == expected
    assert adapter.checkpoint_signature == StarVLAVLActCollectionAdapter(
        client=_Client()
    ).checkpoint_signature


def test_rejects_non_public_models_and_ambiguous_weights() -> None:
    client = _Client()
    client.collection["items"][1]["private"] = True
    with pytest.raises(ValueError, match="unexpected or non-public"):
        StarVLAVLActCollectionAdapter(client=client).fetch_page({})
    client.collection["items"][1]["private"] = False

    class AmbiguousClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            response = super().get(url, params=params, headers=headers)
            if "/tree/" in url:
                payload = json.loads(response.body)
                payload.append(
                    {
                        "type": "file",
                        "path": "checkpoints/other.pt",
                        "oid": "b",
                        "size": 1,
                    }
                )
                return HttpResponse(200, {}, json.dumps(payload).encode(), url)
            return response

    with pytest.raises(ValueError, match="expected one checkpoint .pt file"):
        StarVLAVLActCollectionAdapter(client=AmbiguousClient()).fetch_page({})
