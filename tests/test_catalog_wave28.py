from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.catalog import create_source
from modelome.sources.deep_wetlands_checkpoint_registry import (
    DeepWetlandsCheckpointRegistrySourceAdapter,
)
from modelome.sources.msst_mel_roformer_experiments import (
    MsstMelRoformerExperimentsSourceAdapter,
)
from modelome.sources.openai_consistency_cifar10_checkpoints import (
    OpenAIConsistencyCIFAR10CheckpointSourceAdapter,
)
from modelome.sources.paddlenlp_xlm_registry import PaddleNlpXlmRegistrySourceAdapter
from modelome.sources.timm_legacy_byobnet import TimmLegacyByobNetSourceAdapter


def test_paddlenlp_xlm_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "paddlenlp-xlm-pretrained-registry",
            "adapter": "paddlenlp_xlm_registry",
            "repository": "PaddlePaddle/PaddleNLP",
            "branch": "develop",
            "source_path": "paddlenlp/transformers/xlm/configuration.py",
            "provider_namespace": "paddlenlp:transformer-model",
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, PaddleNlpXlmRegistrySourceAdapter)
    assert source.source_path == "paddlenlp/transformers/xlm/configuration.py"
    assert source.provider_namespace == "paddlenlp:transformer-model"
    assert source.client is client


def test_openai_cifar10_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "openai-consistency-cifar10-checkpoints",
            "adapter": "openai_consistency_cifar10_checkpoints",
            "repository": "openai/consistency_models_cifar10",
            "branch": "main",
            "document_path": "README.md",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_checkpoints": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, OpenAIConsistencyCIFAR10CheckpointSourceAdapter)
    assert source.repository == "openai/consistency_models_cifar10"
    assert source.max_checkpoints == 100
    assert source.client is client


def test_deep_wetlands_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "deep-wetlands-checkpoint-registry",
            "adapter": "deep_wetlands_checkpoint_registry",
            "max_response_bytes": 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, DeepWetlandsCheckpointRegistrySourceAdapter)
    assert source.max_response_bytes == 1024 * 1024
    assert source.client is client


def test_timm_byobnet_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "timm-legacy-byobnet-v0613",
            "adapter": "timm_legacy_byobnet",
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, TimmLegacyByobNetSourceAdapter)
    assert source.client is client


def test_msst_mel_roformer_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "msst-mel-roformer-experiments",
            "adapter": "msst_mel_roformer_experiments",
            "max_response_bytes": 2 * 1024 * 1024,
            "max_models": 50,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, MsstMelRoformerExperimentsSourceAdapter)
    assert source.max_models == 50
    assert source.client is client
