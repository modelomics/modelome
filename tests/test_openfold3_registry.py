from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.openfold3_registry import OpenFold3ParameterRegistryAdapter

COMMIT = "a" * 40


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def json_response(value: Any, url: str) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(value).encode(), url)


def text_response(value: str, url: str) -> HttpResponse:
    return HttpResponse(200, {}, value.encode(), url)


def test_openfold3_registry_preserves_checkpoint_names_and_s3_identities() -> None:
    api = "https://api.github.com/repos/aqlaboratory/openfold-3/commits/main"
    raw = f"https://raw.githubusercontent.com/aqlaboratory/openfold-3/{COMMIT}/openfold3/entry_points/parameters.py"
    source_text = '''
OPENFOLD_BUCKET = "openfold3-data"
OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {
    "openbind-2025-06-30-174k": CheckpointEntry(
        file_name="of3-ob-2025-06-30-174k.pt", version_compatibility=">=0.5.0",
    ),
    "openfold3-p2-155k": CheckpointEntry("of3-p2-155k.pt", ">=0.4,<0.4.4dev0"),
    "openfold3-p2-145k": CheckpointEntry(
        file_name="of3-p2-145k.pt", version_compatibility=">=0.4,<0.4.4dev0"
    ),
    "openfold3-p1": CheckpointEntry(file_name="of3_ft3_v1.pt", version_compatibility="<0.4"),
}
DEFAULT_CHECKPOINT_NAME = "openbind-2025-06-30-174k"
LEGACY_CHECKPOINTS = ["openfold3-p1", "openfold3-p2-145k", "openfold3-p2-155k"]

def download_model_parameters():
    checkpoint_s3_key = f"openfold3-parameters/{checkpoint_file_name}"
'''
    client = QueueClient(
        json_response({"sha": COMMIT}, api),
        text_response(source_text, raw),
    )
    source = OpenFold3ParameterRegistryAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 4
    record = page.records[0]
    models = {model.name: model for model in record.models}
    assert set(models) == {
        "OpenFold3 openbind-2025-06-30-174k",
        "OpenFold3 openfold3-p1",
        "OpenFold3 openfold3-p2-145k",
        "OpenFold3 openfold3-p2-155k",
    }
    assert models["OpenFold3 openbind-2025-06-30-174k"].identifiers == (
        Identifier("openfold3:parameter", "openbind-2025-06-30-174k"),
    )
    releases = {release.model_local_id: release for release in record.releases}
    default = releases["model:openbind-2025-06-30-174k"]
    assert default.metadata["filename"] == "of3-ob-2025-06-30-174k.pt"
    assert default.metadata["s3_uri"] == (
        "s3://openfold3-data/openfold3-parameters/of3-ob-2025-06-30-174k.pt"
    )
    assert default.metadata["version_compatibility"] == ">=0.5.0"
    assert default.metadata["download_supported_by_current_registry"] is True
    legacy = releases["model:openfold3-p2-155k"]
    assert legacy.metadata["download_supported_by_current_registry"] is False
    assert releases["model:openfold3-p2-145k"].metadata[
        "version_compatibility"
    ] == ">=0.4,<0.4.4dev0"
    assert releases["model:openfold3-p1"].metadata["filename"] == "of3_ft3_v1.pt"
    assert len(record.links) == 5
    assert all(not link.crawl for link in record.links)
    assert record.links[1].url.startswith("s3://openfold3-data/")
    assert client.calls == [api, raw]


def test_openfold3_rejects_other_repository_and_nonliteral_registry_fields() -> None:
    with pytest.raises(ValueError, match="repository must"):
        OpenFold3ParameterRegistryAdapter(repository="other/openfold-3", client=QueueClient())
    with pytest.raises(ValueError, match="literal filename"):
        from modelome.sources.openfold3_registry import _parse_registry

        _parse_registry(
            '''OPENFOLD_BUCKET = "openfold3-data"
OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {"bad": CheckpointEntry(file_name=NAME)}
LEGACY_CHECKPOINTS = []
def download():
    checkpoint_s3_key = f"openfold3-parameters/{checkpoint_file_name}"
''',
            10,
        )


def test_openfold3_rejects_registry_when_storage_or_legacy_inventory_is_ambiguous() -> None:
    from modelome.sources.openfold3_registry import _parse_registry

    registry = '''OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {
    "release": CheckpointEntry("release.pt", ">=1")
}
LEGACY_CHECKPOINTS = []
OPENFOLD_BUCKET = "openfold3-data"
def download():
    checkpoint_s3_key = f"openfold3-parameters/{checkpoint_file_name}"
'''
    entries, bucket, prefix = _parse_registry(registry, 10)
    assert entries[0]["version_compatibility"] == ">=1"
    assert bucket == "openfold3-data"
    assert prefix == "openfold3-parameters/"

    without_legacy = registry.replace("LEGACY_CHECKPOINTS = []\n", "")
    with pytest.raises(ValueError, match="legacy checkpoint list"):
        _parse_registry(without_legacy, 10)
    bad_storage = registry.replace(
        '"openfold3-parameters/{checkpoint_file_name}"',
        '"other-prefix/{checkpoint_file_name}"',
    )
    assert _parse_registry(bad_storage, 10)[2] == "other-prefix/"
