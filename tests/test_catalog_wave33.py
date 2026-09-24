from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.sources.catalog import create_source
from modelome.sources.huggingface import HuggingFaceSourceAdapter
from modelome.sources.rsna_atlas_registry import RsnaAtlasRegistrySourceAdapter


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_huggingface_factory_defaults_checkpoint_file_metadata_off() -> None:
    source = create_source(
        {
            "name": "huggingface",
            "adapter": "huggingface",
            "url": "https://huggingface.co/api/models",
        },
        client=object(),
        clock=_fixed_clock,
    )

    assert isinstance(source, HuggingFaceSourceAdapter)
    assert source.include_checkpoint_file_metadata is False


def test_huggingface_factory_forwards_checkpoint_file_metadata_option() -> None:
    source = create_source(
        {
            "name": "huggingface",
            "adapter": "huggingface",
            "url": "https://huggingface.co/api/models",
            "include_checkpoint_file_metadata": True,
        },
        client=object(),
        clock=_fixed_clock,
    )

    assert isinstance(source, HuggingFaceSourceAdapter)
    assert source.include_checkpoint_file_metadata is True


def test_huggingface_factory_rejects_checkpoint_metadata_with_revisions() -> None:
    with pytest.raises(
        ValueError,
        match="include_checkpoint_file_metadata cannot be combined with include_revisions",
    ):
        create_source(
            {
                "name": "huggingface",
                "adapter": "huggingface",
                "url": "https://huggingface.co/api/models",
                "include_checkpoint_file_metadata": True,
                "include_revisions": True,
            },
            client=object(),
            clock=_fixed_clock,
        )


def test_rsna_atlas_factory_uses_public_catalog_key_in_adapter() -> None:
    client = object()
    source = create_source(
        {
            "name": "rsna-atlas-medical-ai-model-cards",
            "adapter": "rsna_atlas_registry",
            "page_size": 100,
            "max_entries": 5000,
            "max_response_bytes": 32 * 1024 * 1024,
        },
        client=client,
    )

    assert isinstance(source, RsnaAtlasRegistrySourceAdapter)
    assert source.page_size == 100
    assert source.max_entries == 5000
    assert source.max_response_bytes == 32 * 1024 * 1024
    assert source.client is client
