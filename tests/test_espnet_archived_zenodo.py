from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.espnet_archived_zenodo import (
    EspnetArchivedZenodoCheckpointAdapter,
)

_FILENAME = "asr_train_asr_conformer_raw_char_sp_valid.acc.ave.zip"
_RECORD = "https://zenodo.org/records/4065140"
_API = "https://zenodo.org/api/records/4065140"


class _Client:
    def __init__(self, payload: dict, status: int = 200) -> None:
        import json

        self.body = json.dumps(payload).encode()
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {"content-type": "application/json"}, self.body, url)


def _payload() -> dict:
    return {
        "metadata": {
            "doi": "10.5281/zenodo.4065140",
            "title": (
                "ESPnet2 pretrained model, "
                "kan-bayashi/csj_asr_train_asr_conformer_raw_char_sp_valid.acc.ave, "
                "fs=16k, lang=jp"
            ),
        },
        "files": [
            {
                "key": _FILENAME,
                "size": 407_700_000,
                "checksum": "md5:10e57ef05358561ed64f64a210ddb187",
                "links": {"download": f"{_RECORD}/files/{_FILENAME}?download=1"},
            },
            {"key": "README.txt", "size": 10, "checksum": "md5:abc", "links": {}},
        ],
    }


def test_indexes_the_single_archived_checkpoint_without_fetching_archive_bytes() -> None:
    client = _Client(_payload())
    adapter = EspnetArchivedZenodoCheckpointAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_API]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.kind.value == "weights"
    assert record.title == "kan-bayashi/csj_asr_train_asr_conformer_raw_char_sp_valid.acc.ave"
    assert record.releases[0].metadata["task"] == "asr"
    assert record.releases[0].metadata["language"] == "ja"
    assert record.releases[0].metadata["checksum"] == "md5:10e57ef05358561ed64f64a210ddb187"
    assert record.links[-1].url == f"{_RECORD}/files/{_FILENAME}?download=1"
    assert record.links[-1].crawl is False


def test_accepts_current_zenodo_file_content_self_link() -> None:
    payload = _payload()
    payload["files"][0]["links"] = {
        "self": f"https://zenodo.org/api/records/4065140/files/{_FILENAME}/content"
    }

    record = (
        EspnetArchivedZenodoCheckpointAdapter(client=_Client(payload)).fetch_page({}).records[0]
    )

    assert record.releases[0].metadata["weight_url"] == (
        f"https://zenodo.org/api/records/4065140/files/{_FILENAME}/content"
    )
    assert record.links[-1].relation == "weights"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["metadata"].update(doi="10.5281/zenodo.1"), "DOI does not match"),
        (lambda p: p["metadata"].update(title="unrelated"), "unexpected Zenodo record title"),
        (lambda p: p["files"].clear(), "expected exactly one ESPnet checkpoint file"),
        (
            lambda p: p["files"][0]["links"].update(download="https://example.com/file.zip"),
            "invalid Zenodo checkpoint URL",
        ),
    ],
)
def test_fails_closed_on_mismatched_archive_metadata(mutate, message: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        EspnetArchivedZenodoCheckpointAdapter(client=_Client(payload)).fetch_page({})


def test_rejects_http_errors_and_wrong_file_size_types() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        EspnetArchivedZenodoCheckpointAdapter(client=_Client(_payload(), status=503)).fetch_page({})

    payload = _payload()
    payload["files"][0]["size"] = True
    with pytest.raises(ValueError, match="size is invalid"):
        EspnetArchivedZenodoCheckpointAdapter(client=_Client(payload)).fetch_page({})
