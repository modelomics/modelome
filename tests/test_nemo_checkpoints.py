from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.nemo_checkpoints import NemoCheckpointCatalogSourceAdapter


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        assert params is None
        self.calls.append((url, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(
    body: str = "",
    *,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        status,
        dict(headers or {}),
        body.encode(),
        "https://docs.nvidia.com/nemo-framework/user-guide/latest/checkpoints.html",
    )


_CATALOG = """
<html><body>
  <a href="https://huggingface.co/models?author=nvidia">Hub search</a>
  <table>
    <tr><th>Model Name</th><th>Model Class</th><th>Model Card</th></tr>
    <tr>
      <td>nvidia/hf_audio</td><td>AudioModel</td>
      <td><a href="https://huggingface.co/nvidia/hf_audio">hf_audio</a></td>
    </tr>
    <tr>
      <td><a href="https://ngc.nvidia.com/catalog/models/nvidia:nemo:legacy_audio">legacy_audio</a></td>
      <td>AudioModel</td><td>Legacy card</td>
    </tr>
    <tr>
      <td>QuartzNet</td><td>AudioModel</td>
      <td><a href="https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels">collection</a></td>
    </tr>
  </table>
  <table>
    <tr><th>Model</th><th>Model Card</th></tr>
    <tr>
      <td><a href="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/team_audio">team_audio</a></td>
      <td>NGC</td>
    </tr>
  </table>
</body></html>
"""


def test_checkpoint_catalog_keeps_model_scoped_card_evidence_and_exact_bridges() -> None:
    adapter = NemoCheckpointCatalogSourceAdapter(
        name="nemo-audio-checkpoints",
        url="https://docs.nvidia.com/nemo-framework/user-guide/latest/checkpoints.html",
        client=_QueuedClient(
            _response(
                _CATALOG,
                headers={
                    "ETag": '"nemo-v1"',
                    "Last-Modified": "Tue, 23 Sep 2026 10:00:00 GMT",
                },
            )
        ),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 4
    assert page.next_state["entry_count"] == 4
    assert page.next_state["etag"] == '"nemo-v1"'
    assert [record.title for record in page.records] == [
        "nvidia/hf_audio",
        "legacy_audio",
        "QuartzNet",
        "team_audio",
    ]
    hf, legacy, collection, team = page.records
    assert hf.kind is ArtifactKind.MODEL_CARD
    assert hf.models[0].status is ModelStatus.RELEASED
    assert hf.models[0].identifiers == (
        Identifier("nemo:checkpoint", "nvidia/hf_audio\nhttps://huggingface.co/nvidia/hf_audio"),
        Identifier("huggingface:model", "nvidia/hf_audio"),
    )
    assert legacy.models[0].identifiers[-1] == Identifier(
        "ngc:model", "nvidia/nemo/legacy_audio"
    )
    assert collection.models[0].identifiers == (
        Identifier(
            "nemo:checkpoint",
            "QuartzNet\nhttps://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels",
        ),
    )
    assert team.models[0].identifiers[-1] == Identifier(
        "ngc:model", "nvidia/nemo/team_audio"
    )
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in hf.links
    } == {
        (
            "https://docs.nvidia.com/nemo-framework/user-guide/latest/checkpoints.html",
            "documentation",
            False,
            (hf.models[0].local_id,),
        ),
        (
            "https://huggingface.co/nvidia/hf_audio",
            "model_card",
            False,
            (hf.models[0].local_id,),
        ),
    }
    assert hf.releases[0].identifiers[0].namespace == "nemo:checkpoint-card"


def test_checkpoint_catalog_reuses_conditional_checkpoint() -> None:
    first_client = _QueuedClient(_response(_CATALOG, headers={"ETag": '"nemo-v1"'}))
    adapter = NemoCheckpointCatalogSourceAdapter(
        url="https://docs.nvidia.com/nemo-framework/user-guide/latest/checkpoints.html",
        client=first_client,
    )
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response(status=304))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.records == ()
    assert second.next_state["checked_at"]
    assert second_client.calls[0][1]["If-None-Match"] == '"nemo-v1"'


def test_checkpoint_catalog_fails_closed_when_no_direct_card_rows_exist() -> None:
    adapter = NemoCheckpointCatalogSourceAdapter(
        url="https://docs.nvidia.com/nemo-framework/user-guide/latest/checkpoints.html",
        client=_QueuedClient(
            _response(
                "<table><tr><th>Model Name</th></tr><tr><td>not-a-card</td></tr></table>"
            )
        ),
    )

    with pytest.raises(ValueError, match="no direct checkpoint-card rows"):
        adapter.fetch_page({})


def test_checkpoint_catalog_parses_official_speech_classification_rows() -> None:
    # The first-party Speech Classification checkpoint page uses the
    # Model Name / Model Base Class / Model Card table shape and the current
    # catalog.nvidia.com host (rather than the older ngc.nvidia.com host).
    body = """
    <table>
      <tr><th>Model Name</th><th>Model Base Class</th><th>Model Card</th></tr>
      <tr>
        <td>langid_ambernet</td><td>EncDecSpeakerLabelModel</td>
        <td><a href="https://catalog.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet">NGC</a></td>
      </tr>
    </table>
    """
    adapter = NemoCheckpointCatalogSourceAdapter(
        name="nemo-speech-classification-checkpoints",
        url=(
            "https://docs.nvidia.com/nemo-framework/user-guide/latest/"
            "nemotoolkit/asr/speech_classification/results.html"
        ),
        client=_QueuedClient(_response(body)),
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "langid_ambernet"
    assert record.canonical_url == (
        "https://catalog.nvidia.com/orgs/nvidia/teams/nemo/models/langid_ambernet"
    )
    assert record.models[0].identifiers[-1] == Identifier(
        "ngc:model", "nvidia/nemo/langid_ambernet"
    )
