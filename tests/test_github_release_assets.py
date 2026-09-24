from __future__ import annotations

from typing import Any

import pytest

from modelome.models import ArtifactKind, ModelStatus, SourceRecord
from modelome.sources.github_release_assets import project_github_release_assets


def release_event_record(
    assets: list[dict[str, Any]],
    *,
    release_name: str = "v1.2.0",
    release_body: str = "",
) -> SourceRecord:
    release = {
        "id": 77,
        "tag_name": "v1.2.0",
        "name": release_name,
        "body": release_body,
        "html_url": "https://github.com/lab/model/releases/tag/v1.2.0",
        "published_at": "2026-09-20T12:00:00Z",
        "assets": assets,
    }
    return SourceRecord(
        source_record_id="gharchive:event:12345",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url="https://data.gharchive.org/2026-09-20-12.json.gz",
        title="GitHub public event fixture",
        raw={
            "record_type": "gharchive_public_event",
            "event_id": "12345",
            "repository": {"id": 42, "name": "lab/model"},
            "event": {
                "id": "12345",
                "type": "ReleaseEvent",
                "created_at": "2026-09-20T12:00:00Z",
                "payload": {"action": "published", "release": release},
            },
        },
    )


def asset(asset_id: int, name: str, url: str | None = None) -> dict[str, Any]:
    return {
        "id": asset_id,
        "name": name,
        "content_type": "application/octet-stream",
        "size": 1234,
        "digest": "sha256:" + "a" * 64,
        "browser_download_url": url
        or f"https://github.com/lab/model/releases/download/v1.2.0/{name}",
    }


def test_projects_checkpoint_assets_from_a_release_event_without_api_calls() -> None:
    event = release_event_record(
        [
            asset(1, "weights.safetensors"),
            asset(2, "README.txt"),
            asset(3, "foreign.pt", "https://example.test/lab/model/releases/download/v1/foreign.pt"),
        ]
    )

    candidates = project_github_release_assets(event)

    assert len(candidates) == 1
    record = candidates[0]
    assert record.source_record_id == "github-release-asset:42:77:1"
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url.endswith("/weights.safetensors")
    assert record.identifiers[-1].value == "1"
    assert record.links[0].relation == "source_release"
    assert record.raw["is_verified_model_checkpoint"] is False


def test_projects_only_the_bounded_prefix_of_embedded_assets() -> None:
    event = release_event_record([asset(i, f"{i}.pt") for i in range(1, 5)])

    candidates = project_github_release_assets(event, max_assets=2)

    assert [item.title for item in candidates] == ["1.pt", "2.pt"]


def test_adds_a_low_confidence_model_hint_for_descriptive_model_asset_evidence() -> None:
    event = release_event_record(
        [asset(1, "resnet50.bin")],
        release_body="Pretrained neural model weights for ResNet50 image classification.",
    )

    [candidate] = project_github_release_assets(event)

    assert len(candidate.models) == 1
    hint = candidate.models[0]
    assert hint.name == "resnet50"
    assert hint.status is ModelStatus.CANDIDATE
    assert hint.confidence == 0.2
    assert hint.identifiers == ()


def test_projects_model_named_zip_when_release_context_confirms_weights() -> None:
    event = release_event_record(
        [asset(1, "qwen2.5-7b-instruct.zip")],
        release_body="Pretrained model weights for Qwen2.5 7B Instruct.",
    )

    [candidate] = project_github_release_assets(event)

    assert candidate.title == "qwen2.5-7b-instruct.zip"
    assert candidate.raw["asset_container_type"] == "archive"
    assert len(candidate.models) == 1
    assert candidate.models[0].name == "qwen2.5 7b instruct"
    assert candidate.models[0].confidence == 0.2


def test_does_not_infer_model_identity_for_generic_named_archive() -> None:
    event = release_event_record(
        [asset(1, "source.zip")],
        release_body="Pretrained model weights for Qwen2.5 7B Instruct.",
    )

    assert project_github_release_assets(event) == ()


def test_projects_model_named_extensionless_asset_only_with_release_context() -> None:
    event = release_event_record(
        [asset(1, "qwen2.5-7b-instruct")],
        release_body="Pretrained model weights for Qwen2.5 7B Instruct.",
    )

    [candidate] = project_github_release_assets(event)

    assert candidate.raw["asset_container_type"] == "single_file"
    assert candidate.models[0].name == "qwen2.5 7b instruct"


def test_does_not_infer_model_from_generic_extensionless_asset_name() -> None:
    event = release_event_record(
        [asset(1, "download")],
        release_body="Pretrained model weights for Qwen2.5 7B Instruct.",
    )

    assert project_github_release_assets(event) == ()


def test_recovers_model_identity_from_release_title_for_generic_checkpoint_filename() -> None:
    event = release_event_record(
        [asset(1, "model.safetensors")],
        release_name="Qwen2.5 7B Instruct",
    )

    [candidate] = project_github_release_assets(event)

    assert len(candidate.models) == 1
    assert candidate.models[0].name == "Qwen2.5 7B Instruct"
    assert candidate.models[0].locator == "$.payload.release.name"
    assert candidate.models[0].status is ModelStatus.CANDIDATE
    assert candidate.models[0].confidence == 0.2


def test_generic_checkpoint_filename_and_generic_release_title_do_not_create_hint() -> None:
    event = release_event_record(
        [asset(1, "model.safetensors")],
        release_name="v1.2.0",
        release_body="Published model weights for the server update.",
    )

    [candidate] = project_github_release_assets(event)

    assert candidate.models == ()


def test_size_only_release_title_is_not_treated_as_a_model_identity() -> None:
    event = release_event_record(
        [asset(1, "model.safetensors")],
        release_name="7B",
    )

    [candidate] = project_github_release_assets(event)

    assert candidate.models == ()


def test_recovers_model_identity_from_a_descriptive_release_tag() -> None:
    event = release_event_record([asset(1, "model.safetensors")])
    raw = dict(event.raw)
    event_payload = dict(raw["event"])
    payload = dict(event_payload["payload"])
    release = dict(payload["release"])
    release["tag_name"] = "llama-3.1-8b-instruct"
    payload["release"] = release
    event_payload["payload"] = payload
    raw["event"] = event_payload
    event = SourceRecord(
        source_record_id=event.source_record_id,
        kind=event.kind,
        canonical_url=event.canonical_url,
        title=event.title,
        raw=raw,
    )

    [candidate] = project_github_release_assets(event)

    assert candidate.models[0].name == "llama 3.1 8b instruct"
    assert candidate.models[0].locator == "$.payload.release.tag_name"


@pytest.mark.parametrize(
    ("filename", "body"),
    [
        ("download.bin", "Pretrained model weights are available."),
        ("model.pt", "Published updated neural model weights."),
        ("firmware.bin", "Pretrained model weights for ResNet50 release."),
        ("tokenizer.bin", "Pretrained model weights for Llama 3 release."),
    ],
)
def test_does_not_create_model_hints_from_generic_or_non_model_asset_names(
    filename: str,
    body: str,
) -> None:
    event = release_event_record([asset(1, filename)], release_body=body)

    [candidate] = project_github_release_assets(event)

    assert candidate.models == ()


def test_ignores_non_release_events_and_rejects_invalid_asset_limit() -> None:
    event = release_event_record([asset(1, "weights.pt")])
    raw = dict(event.raw)
    raw["event"] = {**raw["event"], "type": "PushEvent"}
    non_release = SourceRecord(
        source_record_id=event.source_record_id,
        kind=event.kind,
        canonical_url=event.canonical_url,
        title=event.title,
        raw=raw,
    )

    assert project_github_release_assets(non_release) == ()
    with pytest.raises(ValueError, match="max_assets must be a positive integer"):
        project_github_release_assets(event, max_assets=0)
