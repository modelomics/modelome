from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.openpi_checkpoint_manifest import (
    OpenPiCheckpointManifestAdapter,
    _parse_checkpoints,
)

_SHA = "e" * 40
_CHECKPOINTS = (
    ("Base Models", "π0", "General-purpose base policy", "pi0_base"),
    ("Base Models", "π0-FAST", "Autoregressive action-token base", "pi0_fast_base"),
    ("Base Models", "π0.5", "Generalist policy", "pi05_base"),
    ("Fine-Tuned Models", "π0-FAST-DROID", "DROID fine-tune", "pi0_fast_droid"),
    ("Fine-Tuned Models", "π0-DROID", "DROID fine-tune", "pi0_droid"),
    ("Fine-Tuned Models", "π0-ALOHA-towel", "Towel folding", "pi0_aloha_towel"),
    (
        "Fine-Tuned Models",
        "π0-ALOHA-tupperware",
        "Tupperware manipulation",
        "pi0_aloha_tupperware",
    ),
    (
        "Fine-Tuned Models",
        "π0-ALOHA-pen-uncap",
        "Pen uncapping",
        "pi0_aloha_pen_uncap",
    ),
    ("Fine-Tuned Models", "π0.5-LIBERO", "LIBERO benchmark", "pi05_libero"),
    ("Fine-Tuned Models", "π0.5-DROID", "DROID fine-tune", "pi05_droid"),
)
_README = "\n".join(
    [
        "# OpenPI",
        "## Model Checkpoints",
        "### Base Models",
        "| Model | Use Case | Checkpoint |",
        "| --- | --- | --- |",
        *[
            f"| {label} | {description} | `gs://openpi-assets/checkpoints/{slug}` |"
            for category, label, description, slug in _CHECKPOINTS
            if category == "Base Models"
        ],
        "### Fine-Tuned Models",
        "| Model | Use Case | Checkpoint |",
        "| --- | --- | --- |",
        *[
            f"| {label} | {description} | `gs://openpi-assets/checkpoints/{slug}` |"
            for category, label, description, slug in _CHECKPOINTS
            if category == "Fine-Tuned Models"
        ],
        "## Installation",
        "gs://openpi-assets/checkpoints/not_a_model",
    ]
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        body = (
            json.dumps({"sha": _SHA}).encode()
            if url.endswith("/commits/main")
            else _README.encode()
        )
        return HttpResponse(200, {}, body, url)


def test_parser_extracts_only_the_ten_explicit_checkpoint_paths() -> None:
    entries = _parse_checkpoints(_README, source="test", maximum=12)

    assert len(entries) == 10
    assert [(entry[0], entry[1]) for entry in entries] == [
        (category, label) for category, label, _, _ in _CHECKPOINTS
    ]
    assert [entry[3] for entry in entries] == [
        f"gs://openpi-assets/checkpoints/{slug}" for _, _, _, slug in _CHECKPOINTS
    ]


def test_adapter_preserves_exact_gcs_paths_as_checkpoint_identities() -> None:
    client = _Client()
    adapter = OpenPiCheckpointManifestAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert client.calls == [
        "https://api.github.com/repos/Physical-Intelligence/openpi/commits/main",
        f"https://raw.githubusercontent.com/Physical-Intelligence/openpi/{_SHA}/README.md",
    ]
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 10
    record = next(item for item in page.records if item.title == "OpenPI π0.5-LIBERO")
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.identifiers == (Identifier("openpi:model", "pi05_libero"),)
    assert record.releases[0].identifiers == (
        Identifier(
            "openpi:checkpoint",
            "gs://openpi-assets/checkpoints/pi05_libero",
        ),
    )
    assert record.releases[0].metadata["checkpoint_category"] == "Fine-Tuned Models"
    assert record.releases[0].metadata["revision"] == _SHA
    assert record.releases[0].metadata["checkpoint_uri"] == (
        "gs://openpi-assets/checkpoints/pi05_libero"
    )


def test_adapter_signature_covers_fixed_scope_and_limits() -> None:
    first = OpenPiCheckpointManifestAdapter(max_entries=12)
    same = OpenPiCheckpointManifestAdapter(max_entries=12)
    smaller = OpenPiCheckpointManifestAdapter(max_entries=10)

    assert first.checkpoint_signature == same.checkpoint_signature
    assert first.checkpoint_signature != smaller.checkpoint_signature


def test_build_entries_keeps_all_checkpoint_subjects_distinct() -> None:
    page = OpenPiCheckpointManifestAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    ).fetch_page({})
    result = build_entries(
        source_record_to_entry_seed(record, source="openpi-checkpoint-manifest")
        for record in page.records
    )

    assert len(result.entries) == 10
    assert {entry.canonical_name for entry in result.entries} == {
        record.models[0].name for record in page.records
    }
    assert all(len(entry.members) == 1 for entry in result.entries)
    assert {entry.members[0].artifact_url for entry in result.entries} == {
        page.records[0].canonical_url
    }
    assert {
        release.identifiers[0].value for entry in result.entries for release in entry.releases
    } == {f"gs://openpi-assets/checkpoints/{slug}" for _, _, _, slug in _CHECKPOINTS}


@pytest.mark.parametrize(
    "document,maximum",
    [
        (_README, 9),
        (
            "## Model Checkpoints\n### Base Models\n"
            "| model | use | path |\n| --- | --- | --- |\n"
            "| Unknown | Unknown | `gs://other-bucket/checkpoints/x` |",
            10,
        ),
    ],
)
def test_parser_enforces_bounds_and_first_party_bucket(document: str, maximum: int) -> None:
    if maximum == 9:
        with pytest.raises(ValueError, match="exceeds 9"):
            _parse_checkpoints(document, source="test", maximum=maximum)
    else:
        assert _parse_checkpoints(document, source="test", maximum=maximum) == ()
