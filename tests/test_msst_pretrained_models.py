from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.msst_pretrained_models import MsstPretrainedModelsSourceAdapter

_COMMIT = "050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679"
_DOC = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{_COMMIT}/docs/pretrained_models.md"
_REPO = "https://github.com/ZFTurbo/Music-Source-Separation-Training"

_RELEASES = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1/"
_DOCUMENT = "\n".join(
    [
        "## Pre-trained models",
        "### Vocal models",
        "| Model Type | Instruments | Metrics | Config | Checkpoint |",
        "|---|---|---|---|---|",
        "| MDX23C | vocals / other | SDR 10.17 | [Config](" + _RELEASES + "config.yaml) | "
        "[Weights](" + _RELEASES + "model.ckpt) |",
        "| DTTNet | bass / drums / vocals / other | SDR 7.48 | [Config](" + _RELEASES
        + "config.yaml) | Weights ([vocals](" + _RELEASES + "dtt_vocals.ckpt) [bass]("
        + _RELEASES + "dtt_bass.ckpt)) |",
        "| HTDemucs | all stems | SDR 9.16 | [Config](" + _RELEASES + "config.yaml) | "
        "[Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/abc123.th) |",
        "| Model Type | Metrics | Config | Checkpoint |",
        "|---|---|---|---|",
        "| SCNet | SDR 9.3 | [Config](" + _RELEASES + "config.yaml) | "
        "[Weights](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/def456.th) |",
    ]
)


class _Client:
    def __init__(self, document: str = _DOCUMENT, commit: str = _COMMIT, status: int = 200):
        self.document = document
        self.commit = commit
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        body = (
            json.dumps({"sha": self.commit}).encode()
            if url.endswith("/commits/main")
            else self.document.encode()
        )
        return HttpResponse(self.status, {}, body, url)


def test_enumerates_checkpoint_links_with_manifest_row_context() -> None:
    client = _Client()
    page = MsstPretrainedModelsSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    ).fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/ZFTurbo/Music-Source-Separation-Training/commits/main",
        _DOC,
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert [record.title for record in page.records] == [
        "model.ckpt",
        "dtt_vocals.ckpt",
        "dtt_bass.ckpt",
        "abc123.th",
        "def456.th",
    ]
    assert page.records[0].releases[0].metadata["model_type"] == "MDX23C"
    assert page.records[1].releases[0].metadata["instruments"] == "bass / drums / vocals / other"
    assert page.records[-1].releases[0].metadata["published_metrics"] == "SDR 9.3"
    assert page.records[0].links[0].url == f"{_REPO}/blob/{_COMMIT}/docs/pretrained_models.md"
    assert all(record.links[-1].relation == "weights" for record in page.records)


def test_unchanged_revision_skips_document_refetch() -> None:
    client = _Client()
    page = MsstPretrainedModelsSourceAdapter(client=client).fetch_page(
        {"completed_revision": _COMMIT, "model_count": 44}
    )

    assert page.records == ()
    assert page.upstream_count == 44
    assert len(client.calls) == 1


def test_rejects_untrusted_links_duplicate_file_names_and_limits() -> None:
    invalid = _DOCUMENT.replace(
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1/model.ckpt",
        "https://example.com/model.ckpt",
    )
    with pytest.raises(ValueError, match="unsupported checkpoint URL"):
        MsstPretrainedModelsSourceAdapter(client=_Client(invalid)).fetch_page({})

    duplicate = _DOCUMENT.replace("dtt_bass.ckpt", "model.ckpt")
    with pytest.raises(ValueError, match="duplicate checkpoint filename"):
        MsstPretrainedModelsSourceAdapter(client=_Client(duplicate)).fetch_page({})

    with pytest.raises(ValueError, match="exceeds 2 entries"):
        MsstPretrainedModelsSourceAdapter(client=_Client(), max_models=2).fetch_page({})


def test_rejects_bad_commit_revision_or_failed_response() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        MsstPretrainedModelsSourceAdapter(client=_Client(status=503)).fetch_page({})

    with pytest.raises(ValueError, match="SHA-1 revision"):
        MsstPretrainedModelsSourceAdapter(client=_Client(commit="not-a-sha")).fetch_page({})
