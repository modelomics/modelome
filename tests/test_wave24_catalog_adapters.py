from pathlib import Path

import pytest

from modelome.sources.alignn_atomwise_registry import AlignnAtomwiseRegistrySourceAdapter
from modelome.sources.catalog import create_source, load_source_configs
from modelome.sources.esm1v_variants import ESM1vVariantRegistryAdapter
from modelome.sources.google_vertex_open_model_lifecycle import GoogleVertexOpenModelLifecycle
from modelome.sources.lerobot_vlajepa_checkpoints import LeRobotVLAJEPACheckpointSourceAdapter
from modelome.sources.openai_guided_diffusion_checkpoints import (
    OpenAIGuidedDiffusionCheckpointSourceAdapter,
)
from modelome.sources.openrouter import OpenRouterVideoModelsSourceAdapter
from modelome.sources.paddleocr_ppstructure_model_list import (
    PaddleOcrPPStructureModelListSourceAdapter,
)
from modelome.sources.piper_voice_catalog import PiperVoiceCatalogSourceAdapter
from modelome.sources.timm_legacy_poolformer import TimmLegacyPoolFormerSourceAdapter
from modelome.sources.transformer_m_checkpoints import TransformerMCheckpointSourceAdapter


@pytest.mark.parametrize(
    ("proposal", "name", "adapter_type"),
    [
        ("esm1v_variants.toml", "esm1v-ensemble-variants", ESM1vVariantRegistryAdapter),
        (
            "transformer_m_checkpoints.toml",
            "transformer-m-official-checkpoints",
            TransformerMCheckpointSourceAdapter,
        ),
        (
            "lerobot_vlajepa_checkpoints.toml",
            "lerobot-vlajepa-checkpoints",
            LeRobotVLAJEPACheckpointSourceAdapter,
        ),
        (
            "paddleocr_ppstructure_model_list.toml",
            "paddleocr-ppstructure-model-list",
            PaddleOcrPPStructureModelListSourceAdapter,
        ),
        (
            "openai_guided_diffusion_checkpoints.toml",
            "openai-guided-diffusion-checkpoints",
            OpenAIGuidedDiffusionCheckpointSourceAdapter,
        ),
        (
            "alignn_atomwise_registry.toml",
            "alignn-atomwise-checkpoints",
            AlignnAtomwiseRegistrySourceAdapter,
        ),
        (
            "piper_voice_catalog.toml",
            "piper-voice-catalog",
            PiperVoiceCatalogSourceAdapter,
        ),
        (
            "timm_legacy_poolformer.toml",
            "timm-legacy-poolformer-v0613",
            TimmLegacyPoolFormerSourceAdapter,
        ),
    ],
)
def test_wave24_proposals_load_through_catalog(proposal, name, adapter_type):
    path = Path("config/proposals") / proposal
    config = next(item for item in load_source_configs(path) if item["name"] == name)

    assert isinstance(create_source(config), adapter_type)


def test_google_vertex_lifecycle_loads_through_catalog():
    source = create_source(
        {
            "name": "google-vertex-managed-open-model-lifecycle",
            "adapter": "google_vertex_open_model_lifecycle",
        }
    )

    assert isinstance(source, GoogleVertexOpenModelLifecycle)


def test_openrouter_video_loads_with_environment_credential_and_bounded_limits():
    source = create_source(
        {
            "name": "openrouter-video-models",
            "adapter": "openrouter_video",
            "max_response_bytes": 1024,
            "max_models": 25,
        },
        environ={"OPENROUTER_API_KEY": "test-secret"},
    )

    assert isinstance(source, OpenRouterVideoModelsSourceAdapter)
    assert source._token == "test-secret"
    assert source.max_response_bytes == 1024
    assert source.max_models == 25


def test_openrouter_video_requires_environment_credential():
    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(
            {"name": "openrouter-video-models", "adapter": "openrouter_video"},
            environ={},
        )
