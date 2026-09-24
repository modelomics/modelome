from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.radiologynet_registry import (
    RadiologyNETCheckpointSourceAdapter,
    _parse_download_table,
)

COMMIT_HASH = hashlib.sha256(b"test").hexdigest()


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(body: str, url: str) -> HttpResponse:
    return HttpResponse(200, {}, body.encode(), url)


def test_radiologynet_emits_model_rows_with_exact_google_drive_file_ids() -> None:
    raw_url = "https://raw.githubusercontent.com/AIlab-RITEH/RadiologyNET-TL-models/master/README.md"
    readme = """## Download
The models are packaged into `.tar.gz` archives.
Model | File Size |
--- | --- |
| DenseNet121 | 74 MiB | [Download](https://drive.google.com/uc?export=download&id=AAA) |
| ResNet50 | 247 MiB | [Download](https://drive.google.com/uc?export=download&id=BBB) |
## Challenges
"""
    client = QueueClient(response(readme, raw_url))
    source = RadiologyNETCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 2
    assert [record.source_record_id for record in page.records] == [
        "radiologynet-first-party-checkpoints:archive:DenseNet121",
        "radiologynet-first-party-checkpoints:archive:ResNet50",
    ]
    densenet = page.records[0]
    assert densenet.models[0].identifiers[0].value == "DenseNet121"
    assert densenet.releases[0].metadata == {
        "architecture": "DenseNet121",
        "archive_format": "tar.gz",
        "declared_size": "74 MiB",
        "archive_url": "https://drive.google.com/uc?export=download&id=AAA",
        "google_drive_file_id": "AAA",
        "internal_checkpoint_path": None,
    }
    assert densenet.links[-1].url == densenet.releases[0].metadata["archive_url"]
    assert all(not link.crawl for record in page.records for link in record.links)
    assert client.calls == [raw_url]
    assert page.next_state["completed_revision"] == hashlib.sha256(readme.encode()).hexdigest()


def test_radiologynet_unchanged_readme_is_noop() -> None:
    raw_url = "https://raw.githubusercontent.com/AIlab-RITEH/RadiologyNET-TL-models/master/README.md"
    readme = (
        "## Download\n"
        "| VGG16 | 1.21 GiB | [Download](https://drive.google.com/uc?export=download&id=CCC) |\n"
    )
    digest = hashlib.sha256(readme.encode()).hexdigest()
    source = RadiologyNETCheckpointSourceAdapter(client=QueueClient(response(readme, raw_url)))

    page = source.fetch_page({"completed_revision": digest, "checkpoint_count": 10})

    assert page.complete and not page.records and page.upstream_count == 10


def test_radiologynet_table_requires_exact_unique_google_drive_rows() -> None:
    unsafe = "## Download\n| Fake | 1 MiB | [Download](https://example.org/model.tar.gz) |"
    duplicate_id = (
        "## Download\n"
        "| A | 1 MiB | [Download](https://drive.google.com/uc?export=download&id=SAME) |\n"
        "| B | 2 MiB | [Download](https://drive.google.com/uc?export=download&id=SAME) |"
    )
    with pytest.raises(ValueError, match="no admitted rows"):
        _parse_download_table(unsafe, 10)
    with pytest.raises(ValueError, match="invalid or duplicate"):
        _parse_download_table(duplicate_id, 10)
    with pytest.raises(ValueError, match="exceeds entry limit"):
        _parse_download_table(
            "## Download\n"
            "| A | 1 MiB | [Download](https://drive.google.com/uc?export=download&id=A) |\n"
            "| B | 2 MiB | [Download](https://drive.google.com/uc?export=download&id=B) |",
            1,
        )
