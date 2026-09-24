from __future__ import annotations

from datetime import UTC, datetime

from modelome.http import HttpClient, HttpResponse
from modelome.sources.azure_asset_gallery_v2 import AzureAssetGalleryV2Adapter
from modelome.sources.catalog import create_source, load_sources
from modelome.sources.keras_efficientnet_weights import KerasEfficientNetWeightsSourceAdapter
from modelome.sources.panns_zenodo_models import PannsZenodoModelsAdapter
from modelome.sources.starvla_vlact_collection import StarVLAVLActCollectionAdapter
from modelome.sources.timm_legacy_vit import TimmLegacyViTSourceAdapter


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_keras_efficientnet_factory_passes_manifest_config_and_client() -> None:
    client = object()
    source = create_source(
        {
            "name": "keras-efficientnet-imagenet-weights",
            "adapter": "keras_efficientnet_weights",
            "repository": "keras-team/keras",
            "branch": "master",
            "source_path": "keras/src/applications/efficientnet.py",
            "max_source_bytes": 2 * 1024 * 1024,
            "max_records": 16,
        },
        client=client,
        clock=_fixed_clock,
    )

    assert isinstance(source, KerasEfficientNetWeightsSourceAdapter)
    assert source.repository == "keras-team/keras"
    assert source.source_path == "keras/src/applications/efficientnet.py"
    assert source.max_records == 16
    assert source.client is client


def test_panns_zenodo_factory_passes_file_limits_and_injections() -> None:
    client = object()
    source = create_source(
        {
            "name": "panns-zenodo-models",
            "adapter": "panns_zenodo_models",
            "max_files_per_record": 100,
            "max_response_bytes": 4 * 1024 * 1024,
        },
        client=client,
        clock=_fixed_clock,
    )

    assert isinstance(source, PannsZenodoModelsAdapter)
    assert source.max_files_per_record == 100
    assert source.max_response_bytes == 4 * 1024 * 1024
    assert source.client is client
    assert source.clock is _fixed_clock


def test_azure_asset_gallery_factory_uses_only_post_capable_injected_clients() -> None:
    config = {
        "name": "azure-asset-gallery-v2-public-models",
        "adapter": "azure_asset_gallery_v2",
        "url": "https://api.catalog.azureml.ms/asset-gallery/v1.0/models",
        "page_size": 100,
        "max_pages": 500,
        "max_response_bytes": 8 * 1024 * 1024,
        "max_hf_origin_details_per_page": 100,
    }
    source = create_source(config)

    assert isinstance(source, AzureAssetGalleryV2Adapter)
    assert source.page_size == 100
    assert source.max_pages == 500
    assert source.max_response_bytes == 8 * 1024 * 1024
    assert source.max_hf_origin_details_per_page == 100
    assert source.client is None

    client = HttpClient()
    injected_source = create_source(config, client=client)
    assert isinstance(injected_source, AzureAssetGalleryV2Adapter)
    assert injected_source.client is None

    class _PostClient:
        def post(self, url: str, *, data: bytes, headers: dict[str, str]) -> HttpResponse:
            return HttpResponse(
                status=200,
                headers={"content-type": "application/json"},
                body=b'{"totalCount":0,"summaries":[],"continuationToken":null}',
                url=url,
            )

    post_client = _PostClient()
    post_source = create_source(config, client=post_client)
    assert isinstance(post_source, AzureAssetGalleryV2Adapter)
    assert post_source.client is post_client
    assert post_source.fetch_page({}).complete


def test_azure_load_sources_uses_native_post_when_given_shared_httpclient(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "azure.toml"
    config_path.write_text(
        '[[source]]\nname = "azure-test"\nadapter = "azure_asset_gallery_v2"\n'
        'url = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"\n'
        'page_size = 100\nmax_pages = 500\nmax_response_bytes = 8388608\n'
        'max_hf_origin_details_per_page = 100\n',
        encoding="utf-8",
    )

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def geturl(self) -> str:
            return "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"

        def read(self, limit: int) -> bytes:
            assert limit > 0
            return b'{"totalCount":0,"summaries":[],"continuationToken":null}'

    def fake_urlopen(request, *, timeout: float):
        assert request.get_method() == "POST"
        assert timeout > 0
        return _Response()

    monkeypatch.setattr("modelome.sources.azure_asset_gallery_v2.urlopen", fake_urlopen)
    source = load_sources(config_path, environ={})["azure-test"]

    assert isinstance(source, AzureAssetGalleryV2Adapter)
    assert source.client is None
    assert source.max_hf_origin_details_per_page == 100
    assert source.fetch_page({}).complete


def test_starvla_factory_passes_limits_and_injections() -> None:
    client = object()
    source = create_source(
        {
            "name": "starvla-vlact-collection",
            "adapter": "starvla_vlact_collection",
            "max_response_bytes": 4 * 1024 * 1024,
            "max_entries": 24,
        },
        client=client,
        clock=_fixed_clock,
    )

    assert isinstance(source, StarVLAVLActCollectionAdapter)
    assert source.max_entries == 24
    assert source.client is client
    assert source.clock is _fixed_clock


def test_timm_legacy_vit_factory_passes_name_and_injections() -> None:
    client = object()
    source = create_source(
        {"name": "timm-legacy-vit-v0613", "adapter": "timm_legacy_vit"},
        client=client,
        clock=_fixed_clock,
    )

    assert isinstance(source, TimmLegacyViTSourceAdapter)
    assert source.name == "timm-legacy-vit-v0613"
    assert source.client is client
    assert source.clock is _fixed_clock
