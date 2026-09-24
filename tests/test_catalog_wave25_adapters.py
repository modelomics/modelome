from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.sources.catalog import create_source

_PROPOSAL_ADAPTERS = {
    "rfdiffusion2_registry.toml": "RFDiffusion2CheckpointRegistryAdapter",
    "openai_improved_diffusion_checkpoints.toml": (
        "OpenAIImprovedDiffusionCheckpointSourceAdapter"
    ),
    "utile_checkpoint_registry.toml": "UTilizeCheckpointRegistrySourceAdapter",
    "nvidia_nim_lifecycle.toml": "NvidiaNimLifecycle",
    "sherpa_tts_model_release.toml": "SherpaTtsModelReleaseSourceAdapter",
    "wsa_model_zoo.toml": "WSAModelZooSourceAdapter",
    "ultra_checkpoint_files.toml": "UltraCheckpointFilesSourceAdapter",
    "timm_legacy_resnetv2.toml": "TimmLegacyResNetV2SourceAdapter",
    "zatom_checkpoint_registry.toml": "ZatomCheckpointRegistrySourceAdapter",
    "zatom_zenodo_checkpoint_record.toml": "ZatomZenodoCheckpointRecordSourceAdapter",
}


class _UnusedClient:
    pass


@pytest.mark.parametrize(("proposal", "adapter_class"), _PROPOSAL_ADAPTERS.items())
def test_disabled_wave25_proposals_load_through_factory(
    proposal: str,
    adapter_class: str,
) -> None:
    path = Path(__file__).parents[1] / "config/proposals" / proposal
    source: dict[str, Any] = tomllib.loads(path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = create_source(
        source,
        client=_UnusedClient(),
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC),
        environ={},
    )

    assert type(adapter).__name__ == adapter_class
    assert adapter.name == source["name"]
    assert adapter.checkpoint_signature
