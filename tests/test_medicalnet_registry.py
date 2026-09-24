from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.medicalnet_registry import (
    MedicalNetRegistrySourceAdapter,
    _parse_checkpoints,
)

README_URL = "https://raw.githubusercontent.com/Tencent/MedicalNet/master/README.md"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(body: str, url: str = README_URL) -> HttpResponse:
    return HttpResponse(200, {}, body.encode(), url)


def test_medicalnet_emits_exact_documented_checkpoint_members_and_archive_handle() -> None:
    readme = """resnet_10_23dataset.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18_23dataset.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34_23dataset.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50_23dataset.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_101.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet_152.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet_200.pth: --model resnet --model_depth 200 --resnet_shortcut B
"""
    client = QueueClient(response(readme))
    source = MedicalNetRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 11
    record = page.records[0]
    assert record.source_record_id == "medicalnet:pretrained-checkpoint-inventory"
    assert len(record.models) == len(record.releases) == 11
    by_name = {model.aliases[0]: model for model in record.models}
    assert set(by_name) == {
        "resnet_10.pth", "resnet_18.pth", "resnet_34.pth", "resnet_50.pth",
        "resnet_101.pth", "resnet_152.pth", "resnet_200.pth",
        "resnet_10_23dataset.pth", "resnet_18_23dataset.pth",
        "resnet_34_23dataset.pth", "resnet_50_23dataset.pth",
    }
    ten = by_name["resnet_10_23dataset.pth"]
    assert ten.identifiers == (Identifier("medicalnet:checkpoint", "resnet_10_23dataset.pth"),)
    assert ten.status is ModelStatus.RELEASED
    ten_release = next(r for r in record.releases if r.model_local_id == ten.local_id)
    assert ten_release.metadata["checkpoint_variant"] == "23-dataset"
    assert ten_release.metadata["archive_filename"] == "MedicalNet_pytorch_files2.zip"
    assert ten_release.metadata["archive_file_id"] == "13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG"
    assert ten_release.metadata["archive_url"] == (
        "https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing"
    )
    assert ten_release.metadata["archive_member_url"] is None
    assert all(not link.crawl for link in record.links)
    assert client.calls == [README_URL]
    assert page.next_state["completed_revision"] == hashlib.sha256(readme.encode()).hexdigest()


def test_medicalnet_unchanged_readme_returns_no_records() -> None:
    readme = "resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B\n"
    digest = hashlib.sha256(readme.encode()).hexdigest()
    source = MedicalNetRegistrySourceAdapter(client=QueueClient(response(readme)))

    page = source.fetch_page({"completed_revision": digest, "checkpoint_count": 11})

    assert page.complete and page.records == ()
    assert page.upstream_count == 11


def test_medicalnet_parser_rejects_bad_or_unbounded_checkpoint_rows() -> None:
    with pytest.raises(ValueError, match="mismatched depth"):
        _parse_checkpoints(
            "resnet_10.pth: --model resnet --model_depth 18 --resnet_shortcut B", 10
        )
    duplicate = (
        "resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B\n"
        "resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B"
    )
    with pytest.raises(ValueError, match="repeats checkpoint"):
        _parse_checkpoints(duplicate, 10)
    two_rows = (
        "resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B\n"
        "resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A"
    )
    with pytest.raises(ValueError, match="exceeds entry limit"):
        _parse_checkpoints(two_rows, 1)
    with pytest.raises(ValueError, match="no admitted checkpoint"):
        _parse_checkpoints("resnet_999.pth", 10)
