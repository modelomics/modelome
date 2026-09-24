from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.medigan_registry import MediganRegistrySourceAdapter, _parse_index

COMMIT = "c" * 40


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(value: Any, url: str) -> HttpResponse:
    body = json.dumps(value).encode()
    return HttpResponse(200, {}, body, url)


def entry(
    package_url: str = "https://zenodo.org/records/123/files/demo.zip?download=1",
) -> dict[str, Any]:
    return {
        "execution": {
            "package_name": "demo", "package_link": package_url,
            "model_name": "DCGAN", "extension": ".pt",
        },
        "selection": {"modality": ["MRI"]},
        "description": {
            "title": "Medical image generator", "version": "1.2",
            "doi": ["10.5281/zenodo.123"], "comment": "test model",
        },
    }


def test_medigan_reads_versioned_first_party_index_and_emits_exact_package_url() -> None:
    raw = "https://raw.githubusercontent.com/RichardObi/medigan/main/config/global.json"
    body = json.dumps({"00001_DEMO": entry()}).encode()
    client = QueueClient(HttpResponse(200, {}, body, raw))
    source = MediganRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "model:00001_DEMO"
    assert record.models[0].identifiers[0].value == "00001_DEMO"
    assert record.releases[0].version == "1.2"
    exact_package = "https://zenodo.org/records/123/files/demo.zip?download=1"
    assert record.releases[0].metadata["package_url"] == exact_package
    assert record.releases[0].metadata["package_format"] == "zip"
    assert {link.url for link in record.links} == {
        exact_package,
        "https://github.com/RichardObi/medigan/blob/main/config/global.json",
    }
    assert client.calls == [raw]
    assert page.next_state["completed_revision"] == hashlib.sha256(body).hexdigest()


def test_medigan_unchanged_revision_is_a_noop_with_persisted_count() -> None:
    raw = "https://raw.githubusercontent.com/RichardObi/medigan/main/config/global.json"
    body = json.dumps({"00001_DEMO": entry()}).encode()
    source = MediganRegistrySourceAdapter(
        client=QueueClient(HttpResponse(200, {}, body, raw)),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )
    fingerprint = hashlib.sha256(body).hexdigest()
    page = source.fetch_page({"completed_revision": fingerprint, "checkpoint_count": 1})
    assert page.complete and not page.records
    assert page.upstream_count == 1


@pytest.mark.parametrize("url", [
    "https://example.org/records/123/files/model.zip",
    "https://zenodo.org/records/123/files/model.pt",
    "http://zenodo.org/records/123/files/model.zip",
])
def test_medigan_rejects_untrusted_or_non_package_links(url: str) -> None:
    with pytest.raises(ValueError, match="invalid or duplicate package link"):
        _parse_index({"demo": entry(url)}, 10, "https://github.com/RichardObi/medigan", COMMIT)


def test_medigan_rejects_duplicate_package_urls_and_unbounded_index() -> None:
    duplicate = {"one": entry(), "two": entry()}
    with pytest.raises(ValueError, match="invalid or duplicate package link"):
        _parse_index(duplicate, 10, "https://github.com/RichardObi/medigan", COMMIT)
    with pytest.raises(ValueError, match="bounded object"):
        _parse_index({"a": entry(), "b": entry("https://zenodo.org/records/456/files/b.zip")}, 1,
                     "https://github.com/RichardObi/medigan", COMMIT)
