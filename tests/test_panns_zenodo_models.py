from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.panns_zenodo_models import PannsZenodoModelsAdapter

_TITLE = (
    "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern "
    "Recognition (Pretrained Models)"
)
_RECORDS = ("3576403", "3987831")


def _file(record_id: str, filename: str) -> dict[str, Any]:
    return {
        "key": filename,
        "size": 12345,
        "checksum": "md5:0123456789abcdef0123456789abcdef",
        "links": {"self": f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"},
    }


def _payload(record_id: str, files: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "id": int(record_id),
        "metadata": {
            "doi": f"10.5281/zenodo.{record_id}",
            "title": _TITLE,
        },
        "files": files or [_file(record_id, "Cnn14_mAP=0.431.pth")],
    }


class _Client:
    def __init__(self, payloads: dict[str, dict[str, Any]], status: dict[str, int] | None = None):
        self.payloads = payloads
        self.status = status or {}
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        del headers
        self.calls.append(url)
        record_id = url.rsplit("/", 1)[-1]
        body = json.dumps(self.payloads[record_id]).encode()
        return HttpResponse(self.status.get(record_id, 200), {}, body, url)


def _client() -> _Client:
    return _Client(
        {
            "3576403": _payload(
                "3576403",
                [
                    _file("3576403", "Cnn14_mAP=0.431.pth"),
                    _file("3576403", "Cnn10_mAP=0.380.pth"),
                    _file("3576403", "paper_statistics.zip"),
                ],
            ),
            "3987831": _payload(
                "3987831",
                [
                    _file("3987831", "Cnn14_mAP=0.431.pth"),
                    _file("3987831", "Cnn14_16k_mAP=0.438.pth"),
                ],
            ),
        }
    )


def test_enumerates_checkpoint_files_from_both_published_versions() -> None:
    client = _client()
    page = PannsZenodoModelsAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    ).fetch_page({})

    assert client.calls == [
        "https://zenodo.org/api/records/3576403",
        "https://zenodo.org/api/records/3987831",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    assert page.next_state["record_file_counts"] == {"3576403": 2, "3987831": 2}
    old_release, new_release = [
        record.releases[0] for record in page.records if record.title == "Cnn14_mAP=0.431"
    ]
    assert {old_release.version, new_release.version} == {"v1", "v3"}
    assert {old_release.metadata["zenodo_record_id"], new_release.metadata["zenodo_record_id"]} == {
        "3576403",
        "3987831",
    }
    assert all(record.links[-1].crawl is False for record in page.records)


def test_unchanged_records_return_empty_page_with_saved_model_count() -> None:
    adapter = PannsZenodoModelsAdapter(client=_client())
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert second.records == ()
    assert second.upstream_count == first.upstream_count
    assert second.next_state["records_sha256"] == first.next_state["records_sha256"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["metadata"].update(doi="10.5281/zenodo.1"), "identity mismatch"),
        (
            lambda payload: payload["files"][0]["links"].update(
                self="https://example.com/model.pth"
            ),
            "invalid Zenodo checkpoint URL",
        ),
        (lambda payload: payload["files"][0].update(size=True), "size is invalid"),
    ],
)
def test_fails_closed_on_invalid_record_or_checkpoint_metadata(mutate, message: str) -> None:
    client = _client()
    mutate(client.payloads["3576403"])
    with pytest.raises(ValueError, match=message):
        PannsZenodoModelsAdapter(client=client).fetch_page({})


def test_enforces_file_limit_and_http_status() -> None:
    with pytest.raises(ValueError, match="exceeds file-count limit"):
        PannsZenodoModelsAdapter(client=_client(), max_files_per_record=2).fetch_page({})
    client = _client()
    client.status["3987831"] = 503
    with pytest.raises(ValueError, match="HTTP 503"):
        PannsZenodoModelsAdapter(client=client).fetch_page({})
