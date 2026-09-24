from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modelome.models import ArtifactKind, Identifier
from modelome.sources.catalog import create_source, load_source_configs
from modelome.sources.ngc_cli_versions import (
    NgcCliModelVersionsSourceAdapter,
    ngc_model_versions_command,
    parse_ngc_model_versions_json,
)

_VERSION_ROWS = [
    {
        "createdByUser": "ngc-123456789",
        "createdDate": "2025-11-14T23:59:00.730Z",
        "releaseType": "GA",
        "status": "UPLOAD_COMPLETE",
        "totalFileCount": 1,
        "totalSizeInBytes": 1413212300,
        "versionId": "2.0.0",
    },
    {
        "createdDate": "2025-01-02T03:04:05Z",
        "status": "UPLOAD_COMPLETE",
        "totalFileCount": 3,
        "totalSizeInBytes": 4000,
        "versionId": "1.0.0",
    },
]


def test_opt_in_proposal_constructs_without_running_the_cli() -> None:
    config = load_source_configs("config/proposals/ngc_cli_model_versions.toml")[0]
    adapter = create_source(config, environ={})

    assert isinstance(adapter, NgcCliModelVersionsSourceAdapter)
    assert adapter.targets == ("nvidia/tao/actionrecognitionnet",)


def test_cli_command_uses_quoted_wildcard_json_contract() -> None:
    assert ngc_model_versions_command("nvidia/tao/actionrecognitionnet") == (
        "ngc",
        "registry",
        "model",
        "list",
        "nvidia/tao/actionrecognitionnet:*",
        "--format_type",
        "json",
    )
    assert ngc_model_versions_command("org/model", executable="/opt/ngc")[0] == "/opt/ngc"


@pytest.mark.parametrize("target", ["nvidia/model:v1", "model", "org//model", "org/../model"])
def test_cli_command_rejects_non_model_targets(target: str) -> None:
    with pytest.raises(ValueError):
        ngc_model_versions_command(target)


def test_parser_creates_exact_version_identities_and_retains_cli_metadata() -> None:
    records = parse_ngc_model_versions_json(
        json.dumps(_VERSION_ROWS),
        target="nvidia/tao/actionrecognitionnet",
    )

    assert len(records) == 2
    assert records[0].kind is ArtifactKind.CATALOG_RECORD
    assert records[0].source_record_id == "nvidia/tao/actionrecognitionnet:2.0.0"
    assert records[0].identifiers == (
        Identifier("ngc:model-version", "nvidia/tao/actionrecognitionnet:2.0.0"),
    )
    assert records[0].models[0].identifiers == (
        Identifier("ngc:model", "nvidia/tao/actionrecognitionnet"),
    )
    assert records[0].releases[0].metadata == {
        "createdDate": "2025-11-14T23:59:00.730Z",
        "status": "UPLOAD_COMPLETE",
        "totalFileCount": 1,
        "totalSizeInBytes": 1413212300,
        "releaseType": "GA",
    }
    assert records[0].canonical_url == (
        "https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/actionrecognitionnet"
    )


def test_parser_accepts_empty_result_and_rejects_unverifiable_identity_rows() -> None:
    assert parse_ngc_model_versions_json("[]", target="org/model") == ()
    for payload in ("{}", "not json", '[{"status":"UPLOAD_COMPLETE"}]', '["v1"]'):
        with pytest.raises(ValueError):
            parse_ngc_model_versions_json(payload, target="org/model")


def test_adapter_construction_is_lazy_and_signature_covers_target_scope() -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_runner(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(_VERSION_ROWS))

    adapter = NgcCliModelVersionsSourceAdapter(
        ["org/model", "nvidia/tao/actionrecognitionnet"],
        runner=fake_runner,
    )

    assert adapter.checkpoint_signature
    assert adapter.targets == ("org/model", "nvidia/tao/actionrecognitionnet")
    assert calls == []

    first = adapter.fetch_page({})

    assert calls[0][0] == ngc_model_versions_command("org/model")
    assert calls[0][1] == {"check": False, "capture_output": True, "text": True}
    assert first.complete is False
    assert first.next_state == {"target_index": 1}
    assert len(first.records) == 2

    second = adapter.fetch_page(first.next_state)

    assert second.complete is True
    assert second.next_state == {"target_index": 2}
    assert len(second.records) == 2
