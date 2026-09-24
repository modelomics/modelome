from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.apt_vla_checkpoint_registry import APTVLACheckpointRegistryAdapter
from modelome.sources.catalog import create_source
from modelome.sources.demucs_pretrained_registry import DemucsPretrainedRegistrySourceAdapter
from modelome.sources.fal_model_gallery import FalModelGalleryAdapter
from modelome.sources.figshare_model_candidates import FigshareModelCandidatesSourceAdapter
from modelome.sources.figshare_model_candidates_workflow import FigshareModelCandidatesWorkflow
from modelome.sources.huggingface_spaces_checkpoints import (
    HuggingFaceSpacesCheckpointSourceAdapter,
)
from modelome.sources.medicalnet_registry import MedicalNetRegistrySourceAdapter
from modelome.sources.msst_pretrained_models import MsstPretrainedModelsSourceAdapter
from modelome.sources.openai_consistency_checkpoints import (
    OpenAIConsistencyCheckpointSourceAdapter,
)
from modelome.sources.orb_models_pretrained_registry import OrbModelsPretrainedRegistryAdapter
from modelome.sources.paddlenlp_albert_registry import PaddleNlpAlbertRegistrySourceAdapter
from modelome.sources.radiologynet_registry import RadiologyNETCheckpointSourceAdapter


def test_huggingface_spaces_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "huggingface-spaces-checkpoints",
            "adapter": "huggingface_spaces_checkpoints",
            "url": "https://huggingface.co/api/spaces",
            "page_size": 10,
            "max_response_bytes": 16 * 1024 * 1024,
            "max_checkpoint_files": 10_000,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, HuggingFaceSpacesCheckpointSourceAdapter)
    assert source.url == "https://huggingface.co/api/spaces"
    assert source.page_size == 10
    assert source.max_checkpoint_files == 10_000
    assert source.client is client


def test_openai_consistency_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "openai-consistency-checkpoints",
            "adapter": "openai_consistency_checkpoints",
            "repository": "openai/consistency_models",
            "branch": "main",
            "document_path": "README.md",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_checkpoints": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, OpenAIConsistencyCheckpointSourceAdapter)
    assert source.repository == "openai/consistency_models"
    assert source.max_checkpoints == 100
    assert source.client is client


def test_orb_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "orb-models-pretrained-registry",
            "adapter": "orb_models_pretrained_registry",
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, OrbModelsPretrainedRegistryAdapter)
    assert source.max_response_bytes == 4 * 1024 * 1024
    assert source.client is client


def test_apt_vla_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "apt-vla-checkpoint-registry",
            "adapter": "apt_vla_checkpoint_registry",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_entries": 8,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, APTVLACheckpointRegistryAdapter)
    assert source.max_entries == 8
    assert source.client is client


def test_radiologynet_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "radiologynet-first-party-checkpoints",
            "adapter": "radiologynet_registry",
            "repository": "AIlab-RITEH/RadiologyNET-TL-models",
            "branch": "master",
            "max_response_bytes": 2 * 1024 * 1024,
            "max_entries": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, RadiologyNETCheckpointSourceAdapter)
    assert source.repository == "AIlab-RITEH/RadiologyNET-TL-models"
    assert source.max_entries == 100
    assert source.client is client


def test_paddlenlp_albert_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "paddlenlp-albert-pretrained-registry",
            "adapter": "paddlenlp_albert_registry",
            "repository": "PaddlePaddle/PaddleNLP",
            "branch": "develop",
            "source_path": "paddlenlp/transformers/albert/configuration.py",
            "provider_namespace": "paddlenlp:transformer-model",
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, PaddleNlpAlbertRegistrySourceAdapter)
    assert source.source_path == "paddlenlp/transformers/albert/configuration.py"
    assert source.provider_namespace == "paddlenlp:transformer-model"
    assert source.client is client


def test_fal_gallery_factory_accepts_client_without_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "fal-model-gallery",
            "adapter": "fal_model_gallery",
            "url": "https://fal.ai/explore/search",
            "page_size": 24,
            "max_pages": 1000,
            "max_anchors": 10000,
            "max_response_bytes": 8 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, FalModelGalleryAdapter)
    assert source.url == "https://fal.ai/explore/search"
    assert source.page_size == 24
    assert source.max_pages == 1000
    assert source.client is client


def test_figshare_factory_accepts_configured_window_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "figshare-model-candidates-2026-01",
            "adapter": "figshare_model_candidates",
            "from_date": "2026-01-01",
            "until_date": "2026-02-01",
            "max_response_bytes": 8 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 15, tzinfo=UTC),
    )

    assert isinstance(source, FigshareModelCandidatesSourceAdapter)
    assert source.from_date == "2026-01-01"
    assert source.until_date == "2026-02-01"
    assert source.client is client


def test_msst_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "msst-pretrained-checkpoints",
            "adapter": "msst_pretrained_models",
            "max_response_bytes": 2 * 1024 * 1024,
            "max_models": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, MsstPretrainedModelsSourceAdapter)
    assert source.max_models == 100
    assert source.client is client


def test_figshare_workflow_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "figshare-model-candidates-history",
            "adapter": "figshare_model_candidates_workflow",
            "oai_url": "https://api.figshare.com/v2/oai",
            "api_url": "https://api.figshare.com/v2/articles",
            "from_date": "2011-01-01",
            "until_date": "2027-01-01",
            "window_days": 1,
            "max_response_bytes": 8 * 1024 * 1024,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, FigshareModelCandidatesWorkflow)
    assert source.start.isoformat() == "2011-01-01"
    assert source.end.isoformat() == "2027-01-01"
    assert source.window_days == 1
    assert source.client is client


def test_demucs_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "demucs-pretrained-checkpoints",
            "adapter": "demucs_pretrained_registry",
            "max_response_bytes": 1024 * 1024,
            "max_models": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, DemucsPretrainedRegistrySourceAdapter)
    assert source.max_models == 100
    assert source.client is client


def test_medicalnet_factory_accepts_client_and_clock() -> None:
    client = object()
    source = create_source(
        {
            "name": "medicalnet-pretrained-checkpoints",
            "adapter": "medicalnet_registry",
            "repository": "Tencent/MedicalNet",
            "branch": "master",
            "max_response_bytes": 2 * 1024 * 1024,
            "max_entries": 100,
        },
        client=client,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(source, MedicalNetRegistrySourceAdapter)
    assert source.repository == "Tencent/MedicalNet"
    assert source.max_entries == 100
    assert source.client is client
