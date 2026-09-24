from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.catalog import create_source
from modelome.sources.keras_resnet_weights import KerasResNetWeightsSourceAdapter
from modelome.sources.molmobot_policy_collection import MolmoBotPolicyCollectionAdapter
from modelome.sources.nvlabs_edm2_checkpoints import NVlabsEDM2CheckpointSourceAdapter
from modelome.sources.paddlematerials_registry import PaddleMaterialsRegistryAdapter
from modelome.sources.pfrl_pretrained_model_zoo import PfrlPretrainedModelZooAdapter
from modelome.sources.roboflow_universe_candidates import RoboflowUniverseCandidatesAdapter
from modelome.sources.vosk_models import VoskModelsSourceAdapter


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_vosk_factory_passes_listing_bounds_and_injections() -> None:
    client = object()
    clock = _fixed_clock
    source = create_source(
        {
            "name": "vosk-models",
            "adapter": "vosk_models",
            "min_models": 20,
            "max_models": 500,
            "max_response_bytes": 2 * 1024 * 1024,
        },
        client=client,
        clock=clock,
    )

    assert isinstance(source, VoskModelsSourceAdapter)
    assert source.min_models == 20
    assert source.max_models == 500
    assert source.client is client
    assert source.clock is clock


def test_keras_resnet_factory_passes_manifest_config_and_client() -> None:
    client = object()
    source = create_source(
        {
            "name": "keras-resnet-imagenet-weights",
            "adapter": "keras_resnet_weights",
            "repository": "keras-team/keras",
            "branch": "master",
            "source_path": "keras/src/applications/resnet.py",
            "max_source_bytes": 2 * 1024 * 1024,
            "max_records": 32,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, KerasResNetWeightsSourceAdapter)
    assert source.repository == "keras-team/keras"
    assert source.source_path == "keras/src/applications/resnet.py"
    assert source.max_records == 32
    assert source.client is client


def test_molmobot_factory_passes_limits_and_injections() -> None:
    client = object()
    clock = _fixed_clock
    source = create_source(
        {
            "name": "molmobot-policy-collection",
            "adapter": "molmobot_policy_collection",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_entries": 32,
        },
        client=client,
        clock=clock,
    )

    assert isinstance(source, MolmoBotPolicyCollectionAdapter)
    assert source.max_entries == 32
    assert source.client is client
    assert source.clock is clock


def test_pfrl_factory_works_without_injected_clock_and_accepts_one() -> None:
    source = create_source(
        {"name": "pfrl-pretrained-model-zoo", "adapter": "pfrl_pretrained_model_zoo"}
    )
    assert isinstance(source, PfrlPretrainedModelZooAdapter)
    assert source.max_entries == 600

    clock = _fixed_clock
    timed_source = create_source(
        {
            "name": "pfrl-pretrained-model-zoo",
            "adapter": "pfrl_pretrained_model_zoo",
            "max_entries": 530,
        },
        clock=clock,
    )
    assert isinstance(timed_source, PfrlPretrainedModelZooAdapter)
    assert timed_source.max_entries == 530
    assert timed_source.clock is clock


def test_paddlematerials_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "paddlematerials-model-registry",
            "adapter": "paddlematerials_registry",
            "max_response_bytes": 2 * 1024 * 1024,
        },
        client=client,
    )

    assert isinstance(source, PaddleMaterialsRegistryAdapter)
    assert source.max_response_bytes == 2 * 1024 * 1024
    assert source.client is client


def test_roboflow_factory_passes_query_scope_and_limits() -> None:
    client = object()
    source = create_source(
        {
            "name": "roboflow-universe-candidates",
            "adapter": "roboflow_universe_candidates",
            "search_url": "https://universe.roboflow.com/search",
            "query": "object detection",
            "max_pages": 20,
            "max_projects_per_page": 50,
            "max_anchors": 10_000,
        },
        client=client,
    )

    assert isinstance(source, RoboflowUniverseCandidatesAdapter)
    assert source.query == "object detection"
    assert source.max_pages == 20
    assert source.max_projects_per_page == 50
    assert source.client is client


def test_nvlabs_edm2_factory_passes_document_config() -> None:
    client = object()
    source = create_source(
        {
            "name": "nvlabs-edm2-checkpoints",
            "adapter": "nvlabs_edm2_checkpoints",
            "repository": "NVlabs/edm2",
            "branch": "main",
            "document_path": "README.md",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_checkpoints": 20,
        },
        client=client,
    )

    assert isinstance(source, NVlabsEDM2CheckpointSourceAdapter)
    assert source.repository == "NVlabs/edm2"
    assert source.document_path == "README.md"
    assert source.max_checkpoints == 20
    assert source.client is client
