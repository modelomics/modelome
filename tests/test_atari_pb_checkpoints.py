from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.sources.atari_pb_checkpoints import AtariPbCheckpointAdapter

_RECORD = "https://zenodo.org/records/16981616"
_API = "https://zenodo.org/api/records/16981616"
_ALGORITHMS = (
    "atc",
    "bc",
    "cql_dist",
    "cql_mse",
    "curl",
    "dt",
    "idm",
    "mae",
    "r3m",
    "siammae",
    "spr",
    "spr_idm",
)


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
    files = [
        {
            "key": f"{algorithm}.pth",
            "size": 1234,
            "checksum": f"md5:{index:032x}",
            "links": {
                "download": f"{_RECORD}/files/{algorithm}.pth?download=1"
            },
        }
        for index, algorithm in enumerate(_ALGORITHMS, start=1)
    ]
    files.append(
        {
            "key": "Far-OOD.zip",
            "size": 456,
            "checksum": "md5:" + "a" * 32,
            "links": {"download": f"{_RECORD}/files/Far-OOD.zip?download=1"},
        }
    )
    return {"metadata": {"doi": "10.5281/zenodo.16981616"}, "files": files}


def test_indexes_only_the_twelve_exact_algorithm_checkpoint_files() -> None:
    client = _Client(_payload())
    adapter = AtariPbCheckpointAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert client.calls == [_API]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == len(_ALGORITHMS)
    assert [record.models[0].name for record in page.records] == [
        f"Atari-PB {algorithm} pretrained model" for algorithm in sorted(_ALGORITHMS)
    ]
    assert all(record.kind.value == "weights" for record in page.records)
    assert all(record.links[-1].url.endswith("?download=1") for record in page.records)
    assert all(record.links[-1].crawl is False for record in page.records)
    assert all("Far-OOD" not in record.title for record in page.records)


def test_fails_closed_on_incomplete_or_mismatched_release_inventory() -> None:
    payload = _payload()
    payload["files"] = payload["files"][:-2]
    with pytest.raises(ValueError, match="inventory is incomplete"):
        AtariPbCheckpointAdapter(client=_Client(payload)).fetch_page({})

    payload = _payload()
    payload["metadata"]["doi"] = "10.5281/zenodo.1"
    with pytest.raises(ValueError, match="DOI does not match"):
        AtariPbCheckpointAdapter(client=_Client(payload)).fetch_page({})


def test_rejects_out_of_record_download_urls_and_api_failures() -> None:
    payload = _payload()
    payload["files"][0]["links"]["download"] = "https://example.com/atc.pth"
    with pytest.raises(ValueError, match="invalid record-scoped"):
        AtariPbCheckpointAdapter(client=_Client(payload)).fetch_page({})
    with pytest.raises(ValueError, match="HTTP 503"):
        AtariPbCheckpointAdapter(client=_Client(_payload(), status=503)).fetch_page({})
