from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.catalog import create_source
from modelome.sources.dryad_model_candidates import DryadModelCandidatesSourceAdapter
from modelome.sources.geolink_checkpoint_registry import GeoLinkCheckpointRegistrySourceAdapter
from modelome.sources.paddlenlp_funnel_registry import PaddleNlpFunnelRegistrySourceAdapter


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_paddlenlp_funnel_factory_passes_source_config_and_injections() -> None:
    client = object()
    source = create_source(
        {
            "name": "paddlenlp-funnel-pretrained-registry",
            "adapter": "paddlenlp_funnel_registry",
            "repository": "PaddlePaddle/PaddleNLP",
            "branch": "develop",
            "source_path": "paddlenlp/transformers/funnel/configuration.py",
            "provider_namespace": "paddlenlp:transformer-model",
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=_fixed_clock,
    )

    assert isinstance(source, PaddleNlpFunnelRegistrySourceAdapter)
    assert source.repository == "PaddlePaddle/PaddleNLP"
    assert source.source_path == "paddlenlp/transformers/funnel/configuration.py"
    assert source.provider_namespace == "paddlenlp:transformer-model"
    assert source.client is client
    assert source.clock is _fixed_clock


def test_dryad_factory_passes_bounded_search_options_and_doi() -> None:
    client = object()
    source = create_source(
        {
            "name": "dryad-model-weight-candidates",
            "adapter": "dryad_model_candidates",
            "base_url": "https://datadryad.org/api/v2",
            "page_size": 5,
            "query": '"neural network weights"',
            "max_pages": 4,
            "dataset_doi": "10.5061/dryad.b2rbnzsq4",
        },
        client=client,
    )

    assert isinstance(source, DryadModelCandidatesSourceAdapter)
    assert source.page_size == 5
    assert source.query == '"neural network weights"'
    assert source.max_pages == 4
    assert source.dataset_doi == "10.5061/dryad.b2rbnzsq4"
    assert source.client is client


def test_geolink_factory_passes_limit_and_client() -> None:
    client = object()
    source = create_source(
        {
            "name": "geolink-checkpoint-registry",
            "adapter": "geolink_checkpoint_registry",
            "max_response_bytes": 1024 * 1024,
        },
        client=client,
    )

    assert isinstance(source, GeoLinkCheckpointRegistrySourceAdapter)
    assert source.max_response_bytes == 1024 * 1024
    assert source.client is client
