from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.nnunet_registry import NnUNetV1PretrainedRegistryAdapter

RECORD_ID = "3734294"
DOI = "10.5281/zenodo.3734294"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(value: Any, *, headers: dict[str, str] | None = None) -> HttpResponse:
    return HttpResponse(
        200,
        headers or {},
        json.dumps(value).encode(),
        f"https://zenodo.org/api/records/{RECORD_ID}",
    )


def file_entry(
    name: str, md5: str, *, size: int = 1234, link_kind: str = "download"
) -> dict[str, Any]:
    return {
        "key": name,
        "checksum": f"md5:{md5}",
        "size": size,
        "links": {link_kind: (
            f"https://zenodo.org/api/records/{RECORD_ID}/files/{name}/content"
        )},
    }


def test_nnunet_v1_release_enumerates_exact_task_bundles_without_downloading() -> None:
    payload = {
        "metadata": {
            "doi": DOI,
            "title": "pretrained models for 3D semantic image segmentation with nnU-Net",
            "version": "1",
        },
        "files": [
            file_entry(
                "Task001_BrainTumour.zip", "a" * 32, size=1_900_000_000,
                link_kind="self",
            ),
            file_entry("Task029_LITS.zip", "b" * 32, size=5_100_000_000),
            file_entry("README.txt", "c" * 32),
        ],
    }
    source = NnUNetV1PretrainedRegistryAdapter(
        client=QueuedClient(response(payload, headers={"ETag": '"record-v1"'})),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    record = page.records[0]
    assert record.identifiers == (Identifier("doi", DOI),)
    models = {item.name: item for item in record.models}
    assert set(models) == {"Task001_BrainTumour", "Task029_LITS"}
    assert models["Task001_BrainTumour"].identifiers == (
        Identifier("nnunet:model", "Task001_BrainTumour"),
    )
    releases = {item.model_local_id: item for item in record.releases}
    brain = releases["nnunet:Task001_BrainTumour#model"]
    assert brain.version == "1"
    assert brain.revision == "a" * 32
    assert brain.identifiers == (
        Identifier(
            "nnunet:task-archive",
            DOI + "#Task001_BrainTumour.zip@" + "a" * 32,
        ),
    )
    assert brain.metadata["archive_url"].endswith(
        "/Task001_BrainTumour.zip/content"
    )
    assert brain.metadata["archive_size"] == 1_900_000_000
    assert len(record.links) == 2
    assert all(link.relation == "weights" and not link.crawl for link in record.links)
    assert page.next_state["etag"] == '"record-v1"'
    assert source.client.calls == [f"https://zenodo.org/api/records/{RECORD_ID}"]


def test_nnunet_v1_adapter_keeps_legacy_download_links() -> None:
    payload = {
        "metadata": {"doi": DOI, "title": "nnU-Net", "version": "1"},
        "files": [file_entry("Task001_BrainTumour.zip", "a" * 32)],
    }
    source = NnUNetV1PretrainedRegistryAdapter(client=QueuedClient(response(payload)))

    _, files = source._record(payload)

    assert files[0]["url"] == (
        f"https://zenodo.org/api/records/{RECORD_ID}/files/"
        "Task001_BrainTumour.zip/content"
    )


def test_nnunet_registry_rejects_unexpected_record_and_archive_urls() -> None:
    source = NnUNetV1PretrainedRegistryAdapter(client=QueuedClient())
    with pytest.raises(ValueError, match="only official nnU-Net v1 record"):
        NnUNetV1PretrainedRegistryAdapter(record_id="4003545", client=QueuedClient())

    payload = {
        "metadata": {"doi": DOI, "title": "nnU-Net", "version": "1"},
        "files": [
            {
                **file_entry("Task001_BrainTumour.zip", "a" * 32),
                "links": {
                    "download": "https://example.test/Task001_BrainTumour.zip"
                },
            }
        ],
    }
    with pytest.raises(ValueError, match="not in its record"):
        source._record(payload)

    payload["files"] = [
        {
            **file_entry("Task001_BrainTumour.zip", "a" * 32),
            "links": {
                "self": (
                    f"https://zenodo.org/api/records/{RECORD_ID}/files/"
                    "Task001_BrainTumour.zip"
                )
            },
        }
    ]
    with pytest.raises(ValueError, match="not in its record"):
        source._record(payload)
