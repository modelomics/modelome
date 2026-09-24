from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.biomedical_registry import StarDistPretrainedRegistrySourceAdapter

REVISION = "e" * 40
SOURCE_URL = (
    "https://raw.githubusercontent.com/stardist/stardist/"
    f"{REVISION}/stardist/models/__init__.py"
)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(
    value: Any, url: str, *, headers: dict[str, str] | None = None
) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(200, headers or {}, body, url)


def test_stardist_registry_keeps_exact_models_archives_and_checksums() -> None:
    source_file = "\n".join(
        [
            "register_model(StarDist2D, '2D_versatile_fluo',",
            "    'https://github.com/stardist/stardist-models/releases/download/"
            "v0.1/python_2D_versatile_fluo.zip',",
            f"    '{'a' * 64}')",
            "register_model(StarDist3D, '3D_demo',",
            "    'https://github.com/stardist/stardist-models/releases/download/"
            "v0.1/python_3D_demo.zip',",
            f"    '{'b' * 64}')",
            "register_aliases(StarDist2D, '2D_versatile_fluo', "
            "'Versatile (fluorescent nuclei)')",
        ]
    )
    client = QueuedClient(
        response({"sha": REVISION}, "https://api.github.com/commit"),
        response(source_file, SOURCE_URL),
    )
    adapter = StarDistPretrainedRegistrySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    record = page.records[0]
    models = {model.name: model for model in record.models}
    assert set(models) == {"2D_versatile_fluo", "3D_demo"}
    assert models["2D_versatile_fluo"].identifiers == (
        Identifier("stardist:model", "2D_versatile_fluo"),
    )
    releases = {release.model_local_id: release for release in record.releases}
    fluo = releases["stardist:2D_versatile_fluo#model"]
    assert fluo.version == "v0.1"
    assert fluo.revision == "a" * 64
    assert fluo.identifiers == (
        Identifier("stardist:model-release", "2D_versatile_fluo@" + "a" * 64),
    )
    assert fluo.metadata["archive_url"].endswith("python_2D_versatile_fluo.zip")
    assert [(link.url, link.model_local_ids) for link in record.links] == [
        (
            "https://github.com/stardist/stardist-models/releases/download/"
            "v0.1/python_2D_versatile_fluo.zip",
            ("stardist:2D_versatile_fluo#model",),
        ),
        (
            "https://github.com/stardist/stardist-models/releases/download/"
            "v0.1/python_3D_demo.zip",
            ("stardist:3D_demo#model",),
        ),
    ]
    assert page.next_state["completed_revision"] == REVISION


def test_stardist_registry_rejects_non_first_party_archives_and_invalid_hashes() -> None:
    adapter = StarDistPretrainedRegistrySourceAdapter(client=QueuedClient())
    source = (
        "register_model(StarDist2D, '2D_demo', "
        "'https://example.test/model.zip', '" + "a" * 64 + "')"
    )
    with pytest.raises(ValueError, match="outside the StarDist release tree"):
        adapter._registrations(source)

    source = (
        "register_model(StarDist2D, '2D_demo', "
        "'https://github.com/stardist/stardist-models/releases/download/v0.1/model.zip', "
        "'bad-hash')"
    )
    with pytest.raises(ValueError, match="invalid SHA-256"):
        adapter._registrations(source)
