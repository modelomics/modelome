from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.audio_extra import CoquiTtsRegistrySourceAdapter

_REVISION = "a" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/coqui")


def test_coqui_registry_emits_only_checkpoint_bearing_model_entries() -> None:
    registry = {
        "tts_models": {
            "en": {
                "ljspeech": {
                    "glow-tts": {
                        "description": "English single speaker",
                        "license": "Apache-2.0",
                        "github_rls_url": "https://coqui.gateway.scarf.sh/v0.1/tts.zip",
                    },
                    "unreleased": {"description": "No published checkpoint"},
                }
            }
        },
        "vocoder_models": {
            "en": {
                "ljspeech": {
                    "hifigan": {
                        "hf_url": [
                            "https://coqui.gateway.scarf.sh/hf/vocoder/model.pth",
                            "https://coqui.gateway.scarf.sh/hf/vocoder/config.json",
                        ],
                        "commit": "deadbeef",
                    }
                }
            }
        },
    }
    client = _QueuedClient(_response({"sha": _REVISION}), _response(registry))
    adapter = CoquiTtsRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert [record.source_record_id for record in page.records] == [
        "model:tts_models/en/ljspeech/glow-tts",
        "model:vocoder_models/en/ljspeech/hifigan",
    ]
    tts, vocoder = page.records
    assert tts.kind is ArtifactKind.MODEL_CARD
    assert tts.identifiers == (
        Identifier("coqui:tts-model", "tts_models/en/ljspeech/glow-tts"),
    )
    assert tts.releases[0].metadata["license"] == "Apache-2.0"
    assert [link.url for link in tts.links if link.relation == "weights"] == [
        "https://coqui.gateway.scarf.sh/v0.1/tts.zip"
    ]
    assert vocoder.releases[0].metadata["artifact_urls"] == [
        "https://coqui.gateway.scarf.sh/hf/vocoder/model.pth"
    ]
    assert len(client.calls) == 2


def test_coqui_registry_skips_document_fetch_when_commit_is_unchanged() -> None:
    registry = {"tts_models": {"en": {"single": {"vits": {
        "github_rls_url": "https://coqui.gateway.scarf.sh/v0/models.zip"
    }}}}}
    adapter = CoquiTtsRegistrySourceAdapter(
        client=_QueuedClient(_response({"sha": _REVISION}), _response(registry))
    )
    first = adapter.fetch_page({})
    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.complete is True
    assert len(second_client.calls) == 1


@pytest.mark.parametrize(
    "artifact_url",
    (
        "http://coqui.gateway.scarf.sh/v0/model.zip",
        "https://example.com/model.zip",
        "https://coqui.gateway.scarf.sh/v0/config.json",
    ),
)
def test_coqui_registry_does_not_admit_non_checkpoint_urls(artifact_url: str) -> None:
    registry = {"tts_models": {"en": {"single": {"vits": {
        "github_rls_url": artifact_url
    }}}}}
    adapter = CoquiTtsRegistrySourceAdapter(
        client=_QueuedClient(_response({"sha": _REVISION}), _response(registry))
    )

    if artifact_url.startswith("http:") or "example.com" in artifact_url:
        with pytest.raises(ValueError, match="untrusted artifact URL"):
            adapter.fetch_page({})
    else:
        with pytest.raises(ValueError, match="no checkpoint entries"):
            adapter.fetch_page({})
