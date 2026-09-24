from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.msst_mel_roformer_experiments import (
    MsstMelRoformerExperimentsSourceAdapter,
)

_COMMIT = "050cae7345f4ac1e1e27e066c2c5cdc0a2cdb679"
_DOC = f"https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/{_COMMIT}/docs/mel_roformer_experiments.md"
_RELEASE = "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.11/"

_DOCUMENT = "\n".join(
    [
        "# Mel Roformer experiments",
        "Average SDR Score | Chunk size | Depth | Dim | MLP expansion | Skip connection | "
        "Hop size | FFT Size | Dropout | Batch Size | DL Checkpoint | Comment",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
        "8.21* | 352800 | 4 | 256 | 1 | No | 441 | 2048 | 0/0 | 5 | [Config]("
        + _RELEASE
        + "config.yaml) / [Weights]("
        + _RELEASE
        + "model_ep_1.ckpt) | longer training",
        "8.94 | 485100 | 8 | 384 | 4 | Yes | 882 | 4096 | 0/0 | 2 | [Config]("
        + _RELEASE
        + "config.yaml) / Weights ([part 1]("
        + _RELEASE
        + "large.zip.001), [part2]("
        + _RELEASE
        + "large.zip.002)) | split model",
        "1.05 | 352800 | 4 | 128 | 1 | No | 882 | 2048 | 0/0 | 6 | --- | no checkpoint",
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


def test_enumerates_experiments_and_groups_multipart_checkpoint() -> None:
    client = _Client()
    page = MsstMelRoformerExperimentsSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    ).fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/ZFTurbo/Music-Source-Separation-Training/commits/main",
        _DOC,
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert page.next_state["checkpoint_file_count"] == 3
    assert [record.title for record in page.records] == ["model_ep_1.ckpt", "large.zip"]
    assert page.records[0].releases[0].metadata["sdr_score"] == "8.21*"
    assert page.records[0].releases[0].metadata["experiment_parameters"]["chunk size"] == "352800"
    assert page.records[0].releases[0].metadata["comment"] == "longer training"
    assert page.records[1].releases[0].metadata["multipart"] is True
    assert len(page.records[1].links) == 4
    assert page.records[1].links[-1].url.endswith("large.zip.002")


def test_unchanged_revision_skips_document_refetch() -> None:
    client = _Client()
    page = MsstMelRoformerExperimentsSourceAdapter(client=client).fetch_page(
        {"completed_revision": _COMMIT, "model_count": 10}
    )

    assert page.records == ()
    assert page.upstream_count == 10
    assert len(client.calls) == 1


def test_rejects_untrusted_or_malformed_multipart_urls_and_limits() -> None:
    invalid = _DOCUMENT.replace(_RELEASE, "https://example.com/releases/")
    with pytest.raises(ValueError, match="unsupported experiment checkpoint URL"):
        MsstMelRoformerExperimentsSourceAdapter(client=_Client(invalid)).fetch_page({})

    mismatched = _DOCUMENT.replace("large.zip.002", "different.zip.002")
    with pytest.raises(ValueError, match="multipart checkpoint parts do not match"):
        MsstMelRoformerExperimentsSourceAdapter(client=_Client(mismatched)).fetch_page({})

    with pytest.raises(ValueError, match="exceeds 1 checkpoints"):
        MsstMelRoformerExperimentsSourceAdapter(client=_Client(), max_models=1).fetch_page({})


def test_rejects_bad_commit_revision_or_failed_response() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        MsstMelRoformerExperimentsSourceAdapter(client=_Client(status=503)).fetch_page({})

    with pytest.raises(ValueError, match="SHA-1 revision"):
        MsstMelRoformerExperimentsSourceAdapter(client=_Client(commit="not-a-sha")).fetch_page({})
