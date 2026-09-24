from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.mars_zenodo_checkpoints import (
    _API_URL,
    _EXPECTED_FILES,
    _RECORD_ID,
    _TITLE,
    MarsZenodoCheckpointsSourceAdapter,
    _parse_record,
)

SIZES = {
    "mars_base_rgb_encoder_only.pth": 347_703_439,
    "mars_base_sar_encoder_only.pth": 347_687_055,
    "mars_large_rgb_encoder_only.pth": 634_126_336,
    "mars_large_sar_encoder_only.pth": 639_631_360,
}


def payload() -> dict[str, Any]:
    files = [
        {
            "key": filename,
            "size": SIZES[filename],
            "checksum": checksum,
            "links": {
                "self": (
                    f"https://zenodo.org/api/records/{_RECORD_ID}/files/{filename}/content"
                )
            },
        }
        for filename, (_, _, checksum) in _EXPECTED_FILES.items()
    ]
    files.append({"key": "MaRS16M-Demo.zip", "size": 3_400_000_000})
    return {
        "id": int(_RECORD_ID),
        "metadata": {"doi": f"10.5281/zenodo.{_RECORD_ID}", "title": _TITLE},
        "files": files,
    }


class Client:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.response


def response(body: Any, status: int = 200) -> HttpResponse:
    import json

    return HttpResponse(
        status,
        {"content-type": "application/json"},
        json.dumps(body).encode(),
        "https://fixture.test",
    )


def test_adapter_emits_four_exact_checkpoint_urls_and_metadata() -> None:
    client = Client(response(payload()))
    page = MarsZenodoCheckpointsSourceAdapter(client=client).fetch_page({})
    assert page.authoritative_snapshot and page.complete and page.upstream_count == 4
    assert client.urls == [_API_URL]
    assert {record.raw["filename"] for record in page.records} == set(_EXPECTED_FILES)
    assert {record.raw["size_bytes"] for record in page.records} == set(SIZES.values())
    assert {record.raw["checksum"] for record in page.records} == {
        checksum for _, _, checksum in _EXPECTED_FILES.values()
    }
    assert {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    } == {
        f"https://zenodo.org/records/{_RECORD_ID}/files/{filename}?download=1"
        for filename in _EXPECTED_FILES
    }


def test_adapter_uses_record_body_hash_for_freshness() -> None:
    client = Client(response(payload()))
    adapter = MarsZenodoCheckpointsSourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 4
    assert second.records == ()
    assert second.next_state == first.next_state


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update({"id": 42}),
        lambda data: data["metadata"].update({"doi": "10.5281/zenodo.42"}),
        lambda data: data["files"].pop(0),
        lambda data: data["files"][0].update({"checksum": "md5:wrong"}),
        lambda data: data["files"][0]["links"].update(
            {"self": "https://example.invalid/weights.pth"}
        ),
        lambda data: data["files"].append(dict(data["files"][0])),
        lambda data: data["files"][0].update({"size": True}),
    ],
)
def test_parser_rejects_wrong_or_ambiguous_metadata(mutate: Any) -> None:
    data = payload()
    mutate(data)
    with pytest.raises(ValueError):
        _parse_record(data, "test", 20)


def test_parser_rejects_unexpected_file_and_missing_dataset_scope() -> None:
    data = payload()
    data["files"].append({"key": "unrelated.bin", "size": 12})
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        _parse_record(data, "test", 20)
    data = payload()
    data["files"] = data["files"][:-1]
    with pytest.raises(ValueError, match="expected four checkpoints"):
        _parse_record(data, "test", 20)


def test_adapter_rejects_failed_or_oversized_response() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        MarsZenodoCheckpointsSourceAdapter(client=Client(response({}, 503))).fetch_page({})
    with pytest.raises(ValueError, match="byte limit"):
        MarsZenodoCheckpointsSourceAdapter(
            client=Client(response(payload())), max_response_bytes=10
        ).fetch_page({})
