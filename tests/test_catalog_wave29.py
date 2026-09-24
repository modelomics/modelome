from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.asteroid_zenodo_models import AsteroidZenodoModelsAdapter
from modelome.sources.biolm_registry import BioLMRegistrySourceAdapter
from modelome.sources.catalog import create_source
from modelome.sources.clay_legacy_checkpoint import ClayLegacyCheckpointSourceAdapter
from modelome.sources.cogact_checkpoint_registry import CogACTCheckpointRegistryAdapter
from modelome.sources.deepinfra_model_catalog import DeepInfraModelCatalogAdapter
from modelome.sources.dryad_model_candidates import DryadModelCandidatesSourceAdapter
from modelome.sources.keras_convnext_weights import KerasConvNeXtWeightsSourceAdapter
from modelome.sources.mlx_registry import MlxRegistrySourceAdapter
from modelome.sources.paddlenlp_roformer_registry import (
    PaddleNlpRoformerRegistrySourceAdapter,
)
from modelome.sources.sevennet_pretrained_registry import SevenNetPretrainedRegistryAdapter
from modelome.sources.timm_legacy_regnet import TimmLegacyRegNetSourceAdapter
from modelome.sources.vision_registry_extra import OnnxModelZooHubSourceAdapter
from modelome.sources.vitae_rsp_checkpoint_registry import VitaeRSPCheckpointRegistrySourceAdapter
from modelome.sources.yandex_ddpm_ffhq_checkpoint import YandexDDPMFFHQCheckpointSourceAdapter


def test_mlx_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {"name": "mlx-lm-model-type-remapping", "adapter": "mlx_registry"},
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, MlxRegistrySourceAdapter)
    assert source.client is client


def test_clay_legacy_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "clay-legacy-checkpoint",
            "adapter": "clay_legacy_checkpoint",
            "url": "https://clay-foundation.github.io/model/clay-v0/model_embeddings.html",
            "max_response_bytes": 2 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, ClayLegacyCheckpointSourceAdapter)
    assert source.url == "https://clay-foundation.github.io/model/clay-v0/model_embeddings.html"
    assert source.client is client


def test_onnx_model_zoo_hub_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "onnx-model-zoo-hub",
            "adapter": "onnx_model_zoo_hub",
            "url": "https://huggingface.co/api/models?author=onnxmodelzoo",
            "page_size": 100,
            "max_response_bytes": 16 * 1024 * 1024,
            "max_entries": 5000,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, OnnxModelZooHubSourceAdapter)
    assert source.url == "https://huggingface.co/api/models?author=onnxmodelzoo"
    assert source.page_size == 100
    assert source.max_entries == 5000
    assert source.client is client


def test_paddlenlp_roformer_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "paddlenlp-roformer-pretrained-registry",
            "adapter": "paddlenlp_roformer_registry",
            "repository": "PaddlePaddle/PaddleNLP",
            "branch": "develop",
            "source_path": "paddlenlp/transformers/roformer/configuration.py",
            "provider_namespace": "paddlenlp:transformer-model",
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, PaddleNlpRoformerRegistrySourceAdapter)
    assert source.source_path == "paddlenlp/transformers/roformer/configuration.py"
    assert source.client is client


def test_yandex_ddpm_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "yandex-ddpm-ffhq-checkpoint",
            "adapter": "yandex_ddpm_ffhq_checkpoint",
            "repository": "yandex-research/ddpm-segmentation",
            "branch": "master",
            "document_path": "README.md",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_checkpoints": 10,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, YandexDDPMFFHQCheckpointSourceAdapter)
    assert source.repository == "yandex-research/ddpm-segmentation"
    assert source.max_checkpoints == 10
    assert source.client is client


def test_vitae_rsp_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "vitae-rsp-checkpoint-registry",
            "adapter": "vitae_rsp_checkpoint_registry",
            "max_response_bytes": 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, VitaeRSPCheckpointRegistrySourceAdapter)
    assert source.max_response_bytes == 1024 * 1024
    assert source.client is client


def test_sevennet_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "sevennet-pretrained-registry",
            "adapter": "sevennet_pretrained_registry",
            "max_response_bytes": 8 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, SevenNetPretrainedRegistryAdapter)
    assert source.max_response_bytes == 8 * 1024 * 1024
    assert source.client is client


def test_cogact_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "cogact-checkpoint-registry",
            "adapter": "cogact_checkpoint_registry",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_entries": 8,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, CogACTCheckpointRegistryAdapter)
    assert source.max_entries == 8
    assert source.client is client


def test_biolm_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "facebook-bio-lm-first-party-archives",
            "adapter": "biolm_registry",
            "repository": "facebookresearch/bio-lm",
            "branch": "main",
            "max_response_bytes": 2 * 1024 * 1024,
            "max_entries": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, BioLMRegistrySourceAdapter)
    assert source.repository == "facebookresearch/bio-lm"
    assert source.max_entries == 100
    assert source.client is client


def test_timm_regnet_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {"name": "timm-legacy-regnet-v0613", "adapter": "timm_legacy_regnet"},
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, TimmLegacyRegNetSourceAdapter)
    assert source.client is client


def test_asteroid_zenodo_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "asteroid-zenodo-models",
            "adapter": "asteroid_zenodo_models",
            "page_size": 25,
            "max_records": 500,
            "max_response_bytes": 8 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, AsteroidZenodoModelsAdapter)
    assert source.page_size == 25
    assert source.max_records == 500
    assert source.client is client


def test_deepinfra_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "deepinfra-model-catalog",
            "adapter": "deepinfra_model_catalog",
            "url": "https://deepinfra.com/models",
            "max_response_bytes": 8 * 1024 * 1024,
            "max_script_chars": 8 * 1024 * 1024,
            "max_entries": 10_000,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, DeepInfraModelCatalogAdapter)
    assert source.url == "https://deepinfra.com/models"
    assert source.max_entries == 10_000
    assert source.client is client


def test_dryad_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "dryad-model-weight-candidates",
            "adapter": "dryad_model_candidates",
            "base_url": "https://datadryad.org/api/v2",
            "page_size": 10,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, DryadModelCandidatesSourceAdapter)
    assert source.base_url == "https://datadryad.org/api/v2"
    assert source.page_size == 10
    assert source.client is client


def test_keras_convnext_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "keras-convnext-imagenet-weights",
            "adapter": "keras_convnext_weights",
            "repository": "keras-team/keras",
            "branch": "master",
            "source_path": "keras/src/applications/convnext.py",
            "max_source_bytes": 4 * 1024 * 1024,
            "max_records": 20,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, KerasConvNeXtWeightsSourceAdapter)
    assert source.repository == "keras-team/keras"
    assert source.branch == "master"
    assert source.max_source_bytes == 4 * 1024 * 1024
    assert source.max_records == 20
    assert source.client is client
