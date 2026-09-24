from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.demucs_pretrained_registry import DemucsPretrainedRegistrySourceAdapter

_COMMIT = "e976d93ecc3865e5757426930257e200846a520a"
_DOC = f"https://raw.githubusercontent.com/facebookresearch/demucs/{_COMMIT}/demucs/remote/files.txt"
_STORAGE = "https://dl.fbaipublicfiles.com/demucs/"
_REMOTE = """# MDX Models
root: mdx_final/
0d19c1c6-0f06f20e.th
5d2d6c55-db83574e.th
# Hybrid Transformer models
root: hybrid_transformer/
955717e8-8726e21a.th
f7e0c4bc-ba3fe64a.th
# Experimental 6 sources model
5c90dfd2-34c22ccb.th
"""


class _Client:
    def __init__(self, document: str = _REMOTE, commit: str = _COMMIT, status: int = 200):
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


def test_enumerates_official_file_registry_with_folder_and_family_context() -> None:
    client = _Client()
    page = DemucsPretrainedRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    ).fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/facebookresearch/demucs/commits/main",
        _DOC,
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert [record.title for record in page.records] == [
        "0d19c1c6",
        "5d2d6c55",
        "955717e8",
        "f7e0c4bc",
        "5c90dfd2",
    ]
    assert page.records[0].releases[0].metadata["family"] == "MDX Models"
    assert page.records[0].releases[0].metadata["weight_url"] == (
        f"{_STORAGE}mdx_final/0d19c1c6-0f06f20e.th"
    )
    assert page.records[-1].releases[0].metadata["family"] == "Experimental 6 sources model"


def test_unchanged_revision_skips_manifest_fetch() -> None:
    client = _Client()
    page = DemucsPretrainedRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": _COMMIT, "model_count": 27}
    )
    assert page.records == ()
    assert page.upstream_count == 27
    assert len(client.calls) == 1


def test_rejects_bad_entries_duplicates_and_limits() -> None:
    with pytest.raises(ValueError, match="invalid remote checkpoint entry"):
        DemucsPretrainedRegistrySourceAdapter(
            client=_Client(_REMOTE + "not-a-checkpoint.bin\n")
        ).fetch_page({})

    with pytest.raises(ValueError, match="duplicate model signature"):
        DemucsPretrainedRegistrySourceAdapter(
            client=_Client(_REMOTE + "0d19c1c6-aaaaaaaa.th\n")
        ).fetch_page({})

    with pytest.raises(ValueError, match="exceeds 2 models"):
        DemucsPretrainedRegistrySourceAdapter(client=_Client(), max_models=2).fetch_page({})


def test_rejects_bad_commit_revision_and_failed_response() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        DemucsPretrainedRegistrySourceAdapter(client=_Client(status=503)).fetch_page({})

    with pytest.raises(ValueError, match="SHA-1 revision"):
        DemucsPretrainedRegistrySourceAdapter(client=_Client(commit="not-a-sha")).fetch_page({})
