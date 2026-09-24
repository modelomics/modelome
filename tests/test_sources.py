from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.pipeline import SyncEngine
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.arxiv_snapshot import ArxivCompleteSnapshotSourceAdapter
from modelome.sources.bioimageio import BioImageIoSourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.catalog import (
    create_source,
    load_benchmark_configs,
    load_source_configs,
    load_sources,
)
from modelome.sources.civitai import CivitaiModelsSourceAdapter
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.csv_source import CsvSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.detectron2_model_zoo import Detectron2ModelZooSourceAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.espnet_model_zoo import EspnetModelZooSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.github_repositories import GitHubPublicRepositoriesSourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter
from modelome.sources.huggingface import HuggingFaceSourceAdapter
from modelome.sources.json_catalog import JsonCatalogSourceAdapter
from modelome.sources.kaggle import KaggleModelsSourceAdapter
from modelome.sources.keras_hub_preset_registry import KerasHubPresetRegistrySourceAdapter
from modelome.sources.line_checkpoint_card_catalog import (
    LineCheckpointCardCatalogSourceAdapter,
)
from modelome.sources.markdown_checkpoint_list import MarkdownCheckpointListSourceAdapter
from modelome.sources.markdown_model_card_list import MarkdownModelCardListSourceAdapter
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter
from modelome.sources.modelscope import ModelScopeModelsSourceAdapter
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter
from modelome.sources.nemo_checkpoints import NemoCheckpointCatalogSourceAdapter
from modelome.sources.ngc import NgcModelsSourceAdapter
from modelome.sources.onnx_model_zoo import OnnxModelZooSourceAdapter
from modelome.sources.openai_models import OpenAIModelsSourceAdapter
from modelome.sources.openaire import OpenAireGraphSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter, reconstruct_abstract
from modelome.sources.opencsg import OpenCsgModelsSourceAdapter
from modelome.sources.openmmlab import OpenMMLabModelIndexSourceAdapter
from modelome.sources.openreview import OpenReviewSourceAdapter
from modelome.sources.openrouter import OpenRouterModelsSourceAdapter
from modelome.sources.openvino_model_zoo import OpenVinoModelZooSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.sources.paddle_detection_model_zoo import PaddleDetectionModelZooSourceAdapter
from modelome.sources.paddle_model_center import PaddleModelCenterSourceAdapter
from modelome.sources.paddleclas_model_registry import PaddleClasModelRegistrySourceAdapter
from modelome.sources.paddlegan_tutorial_model_zoo import (
    PaddleGanTutorialModelZooSourceAdapter,
)
from modelome.sources.paddleocr_current_model_list import (
    PaddleOcrCurrentModelListSourceAdapter,
)
from modelome.sources.paddlerec_catalog import PaddleRecCatalogSourceAdapter
from modelome.sources.paperswithcode import (
    PapersWithCodeEvaluationMethodsSourceAdapter,
    PapersWithCodeLinksSourceAdapter,
    PapersWithCodeValidatedMethodsSourceAdapter,
)
from modelome.sources.plos import PlosSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.sources.replicate import ReplicateModelsSourceAdapter
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter
from modelome.sources.spacy_models import SpacyModelsSourceAdapter
from modelome.sources.stanza_resources import StanzaResourcesSourceAdapter
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
)
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)
from modelome.sources.timm_model_registry import TimmModelRegistrySourceAdapter
from modelome.sources.torchaudio_pipeline_registry import (
    TorchaudioPipelineRegistrySourceAdapter,
)
from modelome.sources.torchvision_weight_registry import (
    TorchvisionWeightRegistrySourceAdapter,
)
from modelome.sources.zenodo import ZenodoModelRecordsSourceAdapter
from modelome.storage import Database

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(name: str, *, headers: Mapping[str, str] | None = None) -> HttpResponse:
    path = FIXTURES / name
    return HttpResponse(
        status=200,
        headers=dict(headers or {}),
        body=path.read_bytes(),
        url=f"https://fixtures.test/{name}",
    )


def test_csv_source_is_driven_by_mapping_and_preserves_identity_links() -> None:
    client = QueuedClient(
        response(
            "source_epoch.csv",
            headers={"ETag": '"csv-v1"', "Last-Modified": "Mon, 31 Aug 2026 10:00:00 GMT"},
        )
    )
    adapter = CsvSourceAdapter(
        name="catalog",
        url="https://catalog.example.test/all.csv",
        artifact_kind="catalog_record",
        mapping={
            "id_fields": ["Model", "Organization", "Publication date"],
            "title_field": "Model",
            "body_fields": ["Abstract", "Approach", "Task", "Domain"],
            "published_field": "Publication date",
            "modified_field": "Last modified",
            "link_fields": ["Link", "Reference", "Archived links"],
            "model_field": "Model",
            "model_status": "documented",
            "base_model_field": "Base model",
        },
        client=client,
        clock=lambda: NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["etag"] == '"csv-v1"'
    first, second = page.records
    assert first.kind is ArtifactKind.CATALOG_RECORD
    assert first.title == "Example Vision XL"
    assert first.published_at == "2026-08-20"
    assert "Approach: Convolution" in first.text
    assert Identifier("doi", "10.5555/example.1") in first.identifiers
    assert first.models[0].identifiers == (
        Identifier("huggingface:model", "example-lab/example-vision-xl"),
    )
    assert first.model_relations[0].predicate == "base_model"
    assert first.model_relations[0].target.name == "Example Vision Base"
    assert Identifier("github:repository", "example/scientific-generator") in second.identifiers
    assert second.models[0].identifiers == ()


def test_csv_source_uses_conditional_headers_and_completes_on_not_modified() -> None:
    client = QueuedClient(
        HttpResponse(
            status=304,
            headers={},
            body=b"",
            url="https://catalog.example.test/all.csv",
        )
    )
    adapter = CsvSourceAdapter(
        name="catalog",
        url="https://catalog.example.test/all.csv",
        mapping={"title_field": "name", "model_field": "name"},
        client=client,
        clock=lambda: NOW,
    )
    state = {"etag": '"csv-v1"', "http_last_modified": "yesterday"}

    page = adapter.fetch_page(state)

    assert page.complete is True
    assert page.authoritative_snapshot is False
    assert page.records == ()
    assert client.calls[0][2]["If-None-Match"] == '"csv-v1"'
    assert client.calls[0][2]["If-Modified-Since"] == "yesterday"


def test_csv_source_fails_closed_when_required_columns_drift() -> None:
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=b"renamed_model,organization\nExample,Lab\n",
            url="https://catalog.example.test/all.csv",
        )
    )
    adapter = CsvSourceAdapter(
        name="catalog",
        url="https://catalog.example.test/all.csv",
        mapping={
            "id_fields": ["Model", "Organization"],
            "title_field": "Model",
            "model_field": "Model",
        },
        client=client,
    )

    with pytest.raises(ValueError, match="missing required field.*Model"):
        adapter.fetch_page({})


def test_huggingface_follows_link_cursor_and_resumes_full_enumeration() -> None:
    next_url = "https://huggingface.co/api/models?cursor=opaque%3D%3D"
    client = QueuedClient(
        response(
            "source_huggingface_page1.json",
            headers={
                "Link": (
                    f'<{next_url}>; rel="next", '
                    '<https://huggingface.co/api/models?cursor=previous>; rel="prev"'
                ),
                "X-Total-Count": "3",
            },
        ),
        response("source_huggingface_page2.json"),
    )
    adapter = HuggingFaceSourceAdapter(client=client, clock=lambda: NOW)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 3
    assert first.next_state["next_url"] == next_url
    assert first.next_state["raw_items_seen"] == 2
    assert first.next_state["scan_total"] == 3
    assert client.calls[1][0] == next_url
    assert client.calls[1][1] == {}
    assert second.complete is True
    assert second.records[0].source_record_id == "archive/example-model"
    assert second.next_state["watermark"] == "2026-08-31T10:15:00Z"

    record = first.records[0]
    assert record.models[0].identifiers == (
        Identifier("huggingface:model", "example-lab/example-vision-xl"),
    )
    assert record.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "example-lab/example-vision-base"),
    )
    assert record.releases[0].model_local_id == record.models[0].local_id
    assert record.releases[0].revision == "abc123"
    assert record.releases[0].identifiers == (
        Identifier(
            "huggingface:revision",
            "example-lab/example-vision-xl@abc123",
        ),
    )
    assert record.releases[0].metadata["weight_files"] == ["weights.safetensors"]
    assert {link.relation for link in record.links} >= {
        "metadata",
        "model_card_source",
        "model_config",
        "weights",
        "paper_reference",
        "code_reference",
    }
    assert (
        "https://arxiv.org/abs/2608.12345",
        "paper_reference",
    ) in {(link.url, link.relation) for link in record.links}
    assert (
        "https://github.com/example-lab/example-vision-xl",
        "code_reference",
    ) in {(link.url, link.relation) for link in record.links}
    weight_link = next(link for link in record.links if link.relation == "weights")
    assert weight_link.url == (
        "https://huggingface.co/example-lab/example-vision-xl/resolve/"
        "abc123/weights.safetensors"
    )
    assert weight_link.crawl is False


def test_huggingface_rejects_truncated_scan_before_known_total() -> None:
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={"X-Total-Count": "2"},
            body=b'[{"id":"lab/only-first-model"}]',
            url="https://huggingface.co/api/models",
        )
    )

    with pytest.raises(ValueError, match="before the known total of 2"):
        HuggingFaceSourceAdapter(client=client, clock=lambda: NOW).fetch_page({})


def test_huggingface_rejects_non_success_or_oversized_responses() -> None:
    unavailable = QueuedClient(
        HttpResponse(
            status=503,
            headers={},
            body=b"[]",
            url="https://huggingface.co/api/models",
        )
    )
    with pytest.raises(ValueError, match="HTTP 503"):
        HuggingFaceSourceAdapter(client=unavailable).fetch_page({})

    oversized = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=b"x" * 65,
            url="https://huggingface.co/api/models",
        )
    )
    with pytest.raises(ValueError, match="exceeds 64 bytes"):
        HuggingFaceSourceAdapter(
            client=oversized,
            max_response_bytes=64,
        ).fetch_page({})


def test_kaggle_models_pages_public_models_and_preserves_variation_resources() -> None:
    first_payload = {
        "models": [
            {
                "ref": "example-lab/vision-xl",
                "title": "Example Vision XL",
                "description": "Paper: https://arxiv.org/abs/2608.12345",
                "url": "https://www.kaggle.com/models/example-lab/vision-xl",
                "provenanceSources": "https://github.com/example-lab/vision-xl",
                "tags": [{"fullPath": "task > image classification"}],
                "instances": [
                    {
                        "id": 42,
                        "slug": "resnet",
                        "framework": "TensorFlow",
                        "versionNumber": 2,
                        "versionId": 420,
                        "url": "/models/example-lab/vision-xl/TensorFlow/resnet",
                        "downloadUrl": (
                            "/models/example-lab/vision-xl/TensorFlow/resnet/2/download"
                        ),
                        "licenseName": "Apache 2.0",
                        "fineTunable": True,
                        "modelInstanceType": "externalVariant",
                        "externalBaseModelUrl": "https://huggingface.co/example-lab/base",
                        "trainingData": ["https://example.test/datasets/vision"],
                        "totalUncompressedBytes": 1234,
                    }
                ],
            }
        ],
        "nextPageToken": "opaque-next",
        "totalResults": 2,
    }
    second_payload = {
        "models": [{"ref": "archive/classic", "title": "Classic Model"}],
        "totalResults": 2,
    }
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(first_payload).encode(),
            url="https://www.kaggle.com/api/v1/models/list",
        ),
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(second_payload).encode(),
            url="https://www.kaggle.com/api/v1/models/list",
        ),
    )
    adapter = KaggleModelsSourceAdapter(client=client, page_size=100)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state == {
        "next_page_token": "opaque-next",
        "raw_items_seen": 1,
        "scan_total": 2,
    }
    assert client.calls[0][1] == {"sortBy": "createTime", "pageSize": 100}
    assert client.calls[1][1] == {
        "sortBy": "createTime",
        "pageSize": 100,
        "pageToken": "opaque-next",
    }
    assert second.complete is True
    assert second.upstream_count == 2

    record = first.records[0]
    assert record.models[0].identifiers == (
        Identifier("kaggle:model", "example-lab/vision-xl"),
    )
    assert record.releases[0].identifiers == (
        Identifier("kaggle:model-instance", "example-lab/vision-xl/TensorFlow/resnet"),
        Identifier(
            "kaggle:model-instance-version",
            "example-lab/vision-xl/TensorFlow/resnet/2",
        ),
        Identifier("kaggle:model-version", "420"),
        Identifier("kaggle:model-instance-id", "42"),
    )
    assert record.releases[0].metadata["total_uncompressed_bytes"] == 1234
    assert record.model_relations[0].predicate == "base_model"
    assert record.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "example-lab/base"),
    )
    weight_link = next(link for link in record.links if link.relation == "weights")
    assert weight_link.url.endswith("/TensorFlow/resnet/2/download")
    assert weight_link.crawl is False
    assert {link.relation for link in record.links} >= {
        "dataset",
        "model_variant",
        "provenance",
        "weights",
    }


def test_civitai_models_pages_versions_files_and_declared_base_lineage() -> None:
    next_url = (
        "https://civitai.com/api/v1/models?limit=1&sort=Newest&nsfw=true&cursor=opaque-next"
    )
    first_payload = {
        "items": [
            {
                "id": 123,
                "name": "Example Image LoRA",
                "description": "Paper: https://arxiv.org/abs/2608.12345",
                "type": "LORA",
                "baseModels": ["Stable Diffusion XL 1.0"],
                "tags": ["illustration", "adapter"],
                "modelVersions": [
                    {
                        "id": 456,
                        "index": 0,
                        "name": "v1",
                        "publishedAt": "2026-09-01T00:00:00Z",
                        "baseModel": "Stable Diffusion XL 1.0",
                        "baseModelType": "Standard",
                        "availability": "Public",
                        "files": [
                            {
                                "id": 789,
                                "name": "example.safetensors",
                                "downloadUrl": "https://civitai.com/api/download/models/456",
                                "hashes": {
                                    "SHA256": (
                                        "AB" * 32
                                    )
                                },
                            }
                        ],
                    }
                ],
            }
        ],
        "metadata": {"nextPage": next_url},
    }
    second_payload = {
        "items": [{"id": 124, "name": "Archived Model", "modelVersions": []}],
        "metadata": {},
    }
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(first_payload).encode(),
            url="https://civitai.com/api/v1/models",
        ),
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(second_payload).encode(),
            url=next_url,
        ),
    )
    adapter = CivitaiModelsSourceAdapter(client=client, page_size=1)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state == {
        "next_url": (
            "https://civitai.com/api/v1/models?cursor=opaque-next&earlyAccess=true&limit=1&nsfw=true&sort=Newest"
        ),
        "raw_items_seen": 1,
    }
    assert client.calls[0][1] == {
        "limit": 1,
        "sort": "Newest",
        "nsfw": "true",
        "earlyAccess": "true",
    }
    assert client.calls[1][0] == first.next_state["next_url"]
    assert client.calls[1][1] == {}
    assert second.complete is True

    record = first.records[0]
    assert record.models[0].identifiers == (Identifier("civitai:model", "123"),)
    assert record.releases[0].identifiers == (
        Identifier("civitai:model-version", "456"),
        Identifier("civitai:model-file", "789"),
        Identifier("sha256", "ab" * 32),
    )
    assert record.releases[0].released_at == "2026-09-01T00:00:00Z"
    assert record.model_relations[0].predicate == "base_model"
    assert record.model_relations[0].target.identifiers == (
        Identifier("civitai:base-model", "Stable Diffusion XL 1.0"),
    )
    weight_link = next(link for link in record.links if link.relation == "weights")
    assert weight_link.url == "https://civitai.com/api/download/models/456"
    assert weight_link.crawl is False
    assert "https://arxiv.org/abs/2608.12345" in {link.url for link in record.links}


def test_civitai_rejects_cross_endpoint_pagination_state() -> None:
    client = QueuedClient()
    adapter = CivitaiModelsSourceAdapter(client=client)

    with pytest.raises(ValueError, match="pagination URL changed endpoint"):
        adapter.fetch_page({"next_url": "https://attacker.example/models", "raw_items_seen": 1})

    assert client.calls == []


def test_opencsg_models_pages_repositories_and_declared_hub_mirrors() -> None:
    first_payload = {
        "msg": "OK",
        "data": [
            {
                "id": 10,
                "repository_id": 20,
                "path": "example-lab/vision-xl",
                "name": "vision-xl",
                "nickname": "Example Vision XL",
                "description": "Paper: https://arxiv.org/abs/2608.12345",
                "tags": [
                    {"show_name": "image generation", "name": "image-generation"},
                    {"name": "safetensors"},
                ],
                "repository": {
                    "http_clone_url": "https://opencsg.com/models/example-lab/vision-xl.git"
                },
                "hf_path": "example-lab/vision-xl",
                "ms_path": "example-lab/vision-xl",
                "created_at": "2026-09-01T00:00:00Z",
                "updated_at": "2026-09-02T00:00:00Z",
            }
        ],
        "total": 2,
    }
    second_payload = {
        "msg": "OK",
        "data": [{"id": 11, "path": "archive/classic", "name": "Classic"}],
        "total": 2,
    }
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(first_payload).encode(),
            url="https://hub.opencsg.com/api/v1/models",
        ),
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(second_payload).encode(),
            url="https://hub.opencsg.com/api/v1/models",
        ),
    )
    adapter = OpenCsgModelsSourceAdapter(client=client, page_size=1)

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.next_state == {"page": 2, "raw_items_seen": 1, "scan_total": 2}
    assert client.calls[0][1] == {"page": 1, "per": 1, "sort": "recently_update"}
    assert client.calls[1][1] == {"page": 2, "per": 1, "sort": "recently_update"}
    assert second.complete is True

    record = first.records[0]
    assert record.models[0].identifiers == (
        Identifier("opencsg:model", "example-lab/vision-xl"),
    )
    assert Identifier("opencsg:repository", "20") in record.identifiers
    assert [relation.predicate for relation in record.model_relations] == [
        "mirrors",
        "mirrors",
    ]
    assert record.model_relations[0].target.identifiers == (
        Identifier("huggingface:model", "example-lab/vision-xl"),
    )
    assert record.model_relations[1].target.identifiers == (
        Identifier("modelscope:model", "example-lab/vision-xl"),
    )
    clone_link = next(link for link in record.links if link.relation == "model_repository")
    assert clone_link.url == "https://opencsg.com/models/example-lab/vision-xl.git"
    assert clone_link.crawl is False
    assert "https://arxiv.org/abs/2608.12345" in {link.url for link in record.links}


def test_opencsg_rejects_inventory_that_changes_during_a_full_scan() -> None:
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps({"data": [{"path": "lab/first"}], "total": 2}).encode(),
            url="https://hub.opencsg.com/api/v1/models",
        ),
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps({"data": [{"path": "lab/second"}], "total": 3}).encode(),
            url="https://hub.opencsg.com/api/v1/models",
        ),
    )
    adapter = OpenCsgModelsSourceAdapter(client=client, page_size=1)

    first = adapter.fetch_page({})
    with pytest.raises(ValueError, match="provider total changed"):
        adapter.fetch_page(first.next_state)


def test_huggingface_incremental_scan_stops_below_overlap_watermark() -> None:
    next_url = "https://huggingface.co/api/models?cursor=page-2"
    client = QueuedClient(
        response(
            "source_huggingface_page1.json",
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        response(
            "source_huggingface_page2.json",
            headers={"Link": '<https://huggingface.co/api/models?cursor=page-3>; rel="next"'},
        ),
    )
    adapter = HuggingFaceSourceAdapter(
        client=client,
        clock=lambda: NOW,
        overlap_days=1,
    )

    first = adapter.fetch_page({"watermark": "2026-08-30T00:00:00Z"})
    second = adapter.fetch_page(first.next_state)

    assert first.next_state["cutoff"] == "2026-08-29T00:00:00Z"
    assert second.complete is True
    assert second.records == ()
    assert "next_url" not in second.next_state
    assert second.next_state["watermark"] == "2026-08-31T10:15:00Z"


def test_huggingface_quarantines_one_malformed_result_and_keeps_the_page() -> None:
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=(
                b'[{"id":"lab/valid","sha":"abc"},'
                b'{"lastModified":"2026-09-01T00:00:00Z"}]'
            ),
            url="https://huggingface.co/api/models",
        )
    )

    page = HuggingFaceSourceAdapter(client=client, clock=lambda: NOW).fetch_page({})

    assert page.complete is True
    assert [record.source_record_id for record in page.records] == ["lab/valid"]
    assert len(page.issues) == 1
    assert page.issues[0].stage == "source_normalize"


def test_huggingface_discards_private_results_without_explicit_opt_in() -> None:
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=b'[{"id":"private-org/confidential","private":true,"sha":"abc"}]',
            url="https://huggingface.co/api/models",
        )
    )

    page = HuggingFaceSourceAdapter(
        client=client,
        token="token-with-private-access",
        include_private=False,
        clock=lambda: NOW,
    ).fetch_page({})

    assert page.records == ()
    assert page.issues == ()


def test_huggingface_never_sends_token_to_cross_origin_pagination() -> None:
    client = QueuedClient()
    adapter = HuggingFaceSourceAdapter(client=client, token="hf-secret")

    with pytest.raises(ValueError, match="pagination URL changed origin"):
        adapter.fetch_page({"next_url": "https://attacker.example/steal"})

    assert client.calls == []


def test_openalex_reconstructs_abstract_and_resumes_frozen_window() -> None:
    client = QueuedClient(
        response("source_openalex_page1.json"),
        response("source_openalex_page2.json"),
    )
    adapter = OpenAlexSourceAdapter(
        client=client,
        clock=lambda: NOW,
        initial_lookback_days=7,
        sync_mode="updated",
        api_key="openalex-secret",
        mailto="registry@example.test",
    )

    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)

    assert first.complete is False
    assert first.upstream_count == 2
    assert first.next_state["raw_items_seen"] == 1
    assert first.next_state["scan_total"] == 2
    assert first.records[0].text == "A reconstructed abstract."
    assert Identifier("openalex", "W1234567890") in first.records[0].identifiers
    assert Identifier("doi", "10.5555/example.1") in first.records[0].identifiers
    first_params = client.calls[0][1]
    assert first_params["cursor"] == "*"
    assert first_params["per_page"] == 100
    assert first_params["corpus"] == "all"
    assert first_params["filter"] == (
        "from_updated_date:2026-08-25,to_updated_date:2026-09-01"
    )
    assert first_params["api_key"] == "openalex-secret"
    assert first_params["mailto"] == "registry@example.test"
    assert client.calls[1][1]["cursor"] == "opaque-cursor-page-2"
    assert client.calls[1][1]["filter"] == first_params["filter"]
    assert second.complete is True
    assert second.next_state["watermark"] == "2026-09-01T12:00:00Z"
    assert Identifier("arxiv", "2608.54321") in second.records[0].identifiers


def test_openalex_truncated_first_page_retries_the_same_frozen_window(tmp_path) -> None:
    truncated = {
        "meta": {"count": 2, "next_cursor": None},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Only first work",
                "publication_date": "2026-09-01",
            }
        ],
    }
    complete = {
        "meta": {"count": 2, "next_cursor": None},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "First work",
                "publication_date": "2026-09-01",
            },
            {
                "id": "https://openalex.org/W2",
                "title": "Second work",
                "publication_date": "2026-09-01",
            },
        ],
    }
    client = QueuedClient(
        HttpResponse(
            200, {}, json.dumps(truncated).encode(), "https://api.openalex.org/works"
        ),
        HttpResponse(200, {}, json.dumps(complete).encode(), "https://api.openalex.org/works"),
    )
    current_time = [NOW]
    source = OpenAlexSourceAdapter(
        client=client,
        clock=lambda: current_time[0],
        initial_lookback_days=1,
    )
    database = Database(tmp_path / "store")
    database.initialize()
    engine = SyncEngine(database, {source.name: source})

    failed = engine.sync()[0]
    held_state = database.get_source_state(source.name)
    current_time[0] = NOW + timedelta(days=1)
    retried = engine.sync()[0]

    assert failed.status == "failed"
    assert held_state["window_start"] == "2026-08-31T12:00:00Z"
    assert held_state["window_end"] == "2026-09-01T12:00:00Z"
    assert retried.status == "complete"
    assert client.calls[1][1]["filter"] == client.calls[0][1]["filter"]
    dead_letter = database.list_dead_letters("openalex")[0]
    assert dead_letter["stage"] == "source_pagination"
    assert "before the known total of 2" in dead_letter["error"]


def test_openalex_quarantined_first_page_retries_the_same_frozen_window(tmp_path) -> None:
    malformed = {
        "meta": {"count": 1, "next_cursor": None},
        "results": [
            {
                "title": "Missing work ID",
                "publication_date": "2026-09-01",
            }
        ],
    }
    corrected = {
        "meta": {"count": 1, "next_cursor": None},
        "results": [
            {
                "id": "https://openalex.org/W1",
                "title": "Corrected work",
                "publication_date": "2026-09-01",
            }
        ],
    }
    client = QueuedClient(
        HttpResponse(200, {}, json.dumps(malformed).encode(), "https://api.openalex.org/works"),
        HttpResponse(200, {}, json.dumps(corrected).encode(), "https://api.openalex.org/works"),
    )
    current_time = [NOW]
    source = OpenAlexSourceAdapter(
        client=client,
        clock=lambda: current_time[0],
        initial_lookback_days=1,
    )
    database = Database(tmp_path / "store")
    database.initialize()
    engine = SyncEngine(database, {source.name: source})

    failed = engine.sync()[0]
    held_state = database.get_source_state(source.name)
    current_time[0] = NOW + timedelta(days=1)
    retried = engine.sync()[0]

    assert failed.status == "failed"
    assert held_state["cursor"] == "*"
    assert held_state["window_start"] == "2026-08-31T12:00:00Z"
    assert held_state["window_end"] == "2026-09-01T12:00:00Z"
    assert retried.status == "complete"
    assert client.calls[1][1]["filter"] == client.calls[0][1]["filter"]
    assert client.calls[1][1]["filter"] == (
        "from_publication_date:2026-08-31,to_publication_date:2026-09-01"
    )


def test_openalex_abstract_reconstruction_tolerates_invalid_input() -> None:
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({"ordered": [1], "Words": [0], "ignored": ["x"]}) == (
        "Words ordered"
    )


def test_openalex_all_malformed_page_still_advances_its_cursor() -> None:
    payload = {
        "meta": {"count": 2, "next_cursor": "page-two"},
        "results": [{"title": "Missing ID"}],
    }
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={},
            body=json.dumps(payload).encode(),
            url="https://api.openalex.org/works",
        )
    )
    adapter = OpenAlexSourceAdapter(client=client, clock=lambda: NOW)

    page = adapter.fetch_page({})

    assert page.records == ()
    assert len(page.issues) == 1
    assert page.complete is False
    assert page.next_state["cursor"] == "page-two"


def test_pyod_config_uses_the_documented_class_as_component_identity() -> None:
    config = next(
        item
        for item in load_source_configs(Path("config/sources.toml"))
        if item["name"] == "pyod-anomaly-detection-components"
    )
    document = b"""
    <table>
      <tr><th>Type</th><th>Abbr</th><th>Algorithm</th><th>Year</th><th>Class</th><th>Ref</th></tr>
      <tr>
        <td>Probabilistic</td><td>ABOD</td>
        <td>Angle-Based Outlier Detection</td><td>2008</td>
        <td><a href="pyod.models.tabular.html#pyod.models.abod.ABOD">
          pyod.models.abod.ABOD
        </a></td><td>ref</td>
      </tr>
      <tr>
        <td>Probabilistic</td><td>FastABOD</td>
        <td>Fast Angle-Based Outlier Detection</td><td>2008</td>
        <td><a href="pyod.models.tabular.html#pyod.models.abod.ABOD">
          pyod.models.abod.ABOD
        </a></td><td>ref</td>
      </tr>
    </table>
    """
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={"Content-Type": "text/html"},
            body=document,
            url="https://pyod.readthedocs.io/en/latest/",
        )
    )

    page = create_source(config, client=client, clock=lambda: NOW).fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.models[0].identifiers == (
        Identifier("pyod:machine-learning-component", "pyod.models.abod.ABOD"),
    )
    assert record.models[0].name == "Angle-Based Outlier Detection"
    assert record.models[0].aliases == ("Fast Angle-Based Outlier Detection",)
    assert record.links[0].url == (
        "https://pyod.readthedocs.io/en/latest/pyod.models.tabular.html"
    )
    assert record.links[0].model_local_ids == (record.models[0].local_id,)


def test_catalog_loads_current_config_and_environment_tokens() -> None:
    configs = load_source_configs(Path("config/sources.toml"))
    names = [config["name"] for config in configs]
    assert len(names) == len(set(names))
    assert {
        "huggingface",
        "openrouter-models",
        "tensorflow-model-garden",
        "t5x-model-registry",
        "proteinmpnn-checkpoints",
        "openml-flows",
        "xai-models",
    } <= set(names)

    benchmarks = load_benchmark_configs(Path("config/sources.toml"))
    assert [benchmark["name"] for benchmark in benchmarks] == ["epoch"]
    assert benchmarks[0]["mapping"] == {"model_field": "Model"}
    with pytest.raises(ValueError, match="benchmark configuration"):
        create_source(benchmarks[0], client=QueuedClient())

    client = QueuedClient()
    sources = load_sources(
        Path("config/sources.toml"),
        client=client,
        clock=lambda: NOW,
        environ={
            "HF_TOKEN": "hf-secret",
            "OPENALEX_API_KEY": "oa-secret",
            "S2_API_KEY": "s2-secret",
            "MODELOME_CONTACT_EMAIL": "registry@example.test",
        },
    )

    assert "epoch" not in sources
    assert isinstance(sources["huggingface"], HuggingFaceSourceAdapter)
    assert sources["huggingface"].max_response_bytes == 16 * 1024 * 1024
    assert isinstance(sources["kaggle-models"], KaggleModelsSourceAdapter)
    assert sources["kaggle-models"].page_size == 100
    assert isinstance(sources["civitai-models"], CivitaiModelsSourceAdapter)
    assert sources["civitai-models"].include_nsfw is True
    assert isinstance(sources["opencsg-models"], OpenCsgModelsSourceAdapter)
    assert sources["opencsg-models"].sort_by == "recently_update"
    assert isinstance(sources["modelscope-models"], ModelScopeModelsSourceAdapter)
    assert sources["modelscope-models"].page_size == 50
    assert sources["modelscope-models"].max_pages_per_sort == 60
    assert sources["modelscope-models"].sorts == (
        "default",
        "downloads",
        "likes",
        "last_modified",
    )
    assert isinstance(sources["ngc-models"], NgcModelsSourceAdapter)
    assert sources["ngc-models"].page_size == 25
    assert sources["ngc-models"].include_all_versions is True
    for name in (
        "nemo-asr-checkpoints",
        "nemo-tts-checkpoints",
        "nemo-audio-checkpoints",
        "nemo-speaker-diarization-checkpoints",
        "nemo-ssl-checkpoints",
    ):
        assert isinstance(sources[name], NemoCheckpointCatalogSourceAdapter)
        assert sources[name].max_response_bytes == 8 * 1024 * 1024
    assert isinstance(sources["pyod-anomaly-detection-components"], HtmlCatalogSourceAdapter)
    assert sources["pyod-anomaly-detection-components"].provider_namespace == (
        "pyod:machine-learning-component"
    )
    assert sources["pyod-anomaly-detection-components"].model_status.value == "documented"
    assert isinstance(
        sources["pygod-graph-anomaly-detection-components"], HtmlCatalogSourceAdapter
    )
    assert sources["pygod-graph-anomaly-detection-components"].provider_namespace == (
        "pygod:machine-learning-component"
    )
    for name in (
        "statsmodels-statistical-model-components",
        "statsmodels-0-11-statistical-model-components",
        "statsmodels-0-12-statistical-model-components",
        "statsmodels-0-13-statistical-model-components",
        "statsmodels-0-14-statistical-model-components",
    ):
        assert isinstance(sources[name], HtmlCatalogSourceAdapter)
        assert sources[name].provider_namespace == "statsmodels:machine-learning-component"
        assert sources[name].rules[0].identity_source == "href"
    assert isinstance(
        sources["tslearn-time-series-components"], HtmlCatalogSourceAdapter
    )
    assert sources["tslearn-time-series-components"].provider_namespace == (
        "tslearn:machine-learning-component"
    )
    assert sources["tslearn-time-series-components"].rules[0].identity_source == "url"
    assert isinstance(
        sources["autogluon-tabular-model-components"], HtmlCatalogSourceAdapter
    )
    assert sources["autogluon-tabular-model-components"].provider_namespace == (
        "autogluon:machine-learning-component"
    )
    assert sources["autogluon-tabular-model-components"].rules[0].identity_source == "href"
    assert isinstance(
        sources["pytorch-forecasting-model-components"], HtmlCatalogSourceAdapter
    )
    assert sources["pytorch-forecasting-model-components"].provider_namespace == (
        "pytorch-forecasting:machine-learning-component"
    )
    assert sources["pytorch-forecasting-model-components"].rules[0].identity_source == "url"
    assert isinstance(
        sources["gluonts-forecasting-techniques"], HtmlCatalogSourceAdapter
    )
    assert sources["gluonts-forecasting-techniques"].provider_namespace == (
        "gluonts:machine-learning-component"
    )
    assert sources["gluonts-forecasting-techniques"].rules[0].name_source == (
        "before_first_anchor"
    )
    assert len(sources["gluonts-forecasting-techniques"].rules[0].row_link_rules) == 2
    assert isinstance(
        sources["neuralforecast-model-components"], HtmlCatalogSourceAdapter
    )
    assert sources["neuralforecast-model-components"].provider_namespace == (
        "neuralforecast:machine-learning-component"
    )
    assert sources["neuralforecast-model-components"].rules[0].identity_source == "url"
    assert isinstance(
        sources["statsforecast-model-components"], HtmlCatalogSourceAdapter
    )
    assert sources["statsforecast-model-components"].provider_namespace == (
        "statsforecast:machine-learning-component"
    )
    assert sources["statsforecast-model-components"].rules[0].identity_source == "href"
    assert isinstance(
        sources["pytorch-tabular-model-components"], HtmlCatalogSourceAdapter
    )
    assert sources["pytorch-tabular-model-components"].provider_namespace == (
        "pytorch-tabular:machine-learning-component"
    )
    assert sources["pytorch-tabular-model-components"].rules[0].identity_source == "href"
    assert isinstance(sources["torchgeo-model-components"], HtmlCatalogSourceAdapter)
    assert sources["torchgeo-model-components"].provider_namespace == (
        "torchgeo:machine-learning-component"
    )
    assert sources["torchgeo-model-components"].rules[0].identity_source == "url"
    assert isinstance(
        sources["segmentation-models-pytorch-components"], HtmlCatalogSourceAdapter
    )
    assert sources[
        "segmentation-models-pytorch-components"
    ].provider_namespace == "segmentation-models-pytorch:machine-learning-component"
    assert sources["segmentation-models-pytorch-components"].rules[0].identity_source == "href"
    assert isinstance(sources["ollama-library"], HtmlCatalogSourceAdapter)
    assert sources["ollama-library"].provider_namespace == "ollama:model-family"
    assert sources["ollama-library"].model_status.value == "released"
    assert sources["ollama-library"].max_response_bytes == 2 * 1024 * 1024
    assert isinstance(sources["cloudflare-workers-ai"], HtmlCatalogSourceAdapter)
    assert sources["cloudflare-workers-ai"].provider_namespace == (
        "cloudflare-workers-ai:model"
    )
    assert sources["cloudflare-workers-ai"].max_response_bytes == 1024 * 1024
    assert isinstance(sources["zenodo-model-records"], ZenodoModelRecordsSourceAdapter)
    assert sources["zenodo-model-records"].query == "resource_type.type:model"
    assert sources["zenodo-model-records"].sort == "oldest"
    assert sources["zenodo-model-records"].page_size == 25
    assert isinstance(sources["openrouter-models"], OpenRouterModelsSourceAdapter)
    assert sources["openrouter-models"].url == "https://openrouter.ai/api/v1/models"
    assert sources["openrouter-models"].max_response_bytes == 16 * 1024 * 1024
    assert "replicate-models" not in sources
    assert "openai-models" not in sources
    assert "anthropic-models" not in sources
    assert "google-gemini-models" not in sources
    assert "groq-models" not in sources
    assert isinstance(sources["openai-model-documentation"], HtmlCatalogSourceAdapter)
    assert sources["openai-model-documentation"].provider_namespace == "openai:model"
    assert isinstance(sources["mistral-model-documentation"], HtmlCatalogSourceAdapter)
    assert sources["mistral-model-documentation"].provider_namespace == (
        "mistral:model-documentation"
    )
    assert sources["mistral-model-documentation"].detail_batch_size == 10
    assert sources["mistral-model-documentation"].rules[0].name_source == "identity"
    mistral_detail_rules = sources["mistral-model-documentation"].detail_resource_rules
    assert [rule.relation for rule in mistral_detail_rules] == [
        "paper_reference",
        "model_card",
    ]
    assert isinstance(sources["cohere-model-documentation"], HtmlCatalogSourceAdapter)
    assert sources["cohere-model-documentation"].provider_namespace == (
        "cohere:model-documentation"
    )
    assert isinstance(sources["apple-coreml-model-gallery"], HtmlCatalogSourceAdapter)
    assert sources["apple-coreml-model-gallery"].provider_namespace == (
        "apple:coreml-model-package"
    )
    assert sources["apple-coreml-model-gallery"].model_status.value == "released"
    assert sources["apple-coreml-model-gallery"].rules[0].relation == "weights"
    assert sources["apple-coreml-model-gallery"].rules[0].crawl is False
    assert isinstance(sources["aws-bedrock-model-cards"], HtmlCatalogSourceAdapter)
    assert sources["aws-bedrock-model-cards"].provider_namespace == "aws:bedrock-model-card"
    assert isinstance(sources["aws-sagemaker-jumpstart-models"], HtmlCatalogSourceAdapter)
    assert sources["aws-sagemaker-jumpstart-models"].provider_namespace == (
        "aws:sagemaker-jumpstart-model"
    )
    assert isinstance(sources["paperswithcode-links"], PapersWithCodeLinksSourceAdapter)
    assert sources["paperswithcode-links"].dataset_id == (
        "pwc-archive/links-between-paper-and-code"
    )
    assert sources["paperswithcode-links"].page_size == 10_000
    assert isinstance(
        sources["paperswithcode-methods-validated"],
        PapersWithCodeValidatedMethodsSourceAdapter,
    )
    assert sources["paperswithcode-methods-validated"].arxiv_delay_seconds == 3
    assert sources["paperswithcode-method-candidates"].admission == "candidate"
    assert sources["paperswithcode-method-paper-candidates"].admission == (
        "paper_linked_candidate"
    )
    assert isinstance(
        sources["paperswithcode-evaluation-methods"],
        PapersWithCodeEvaluationMethodsSourceAdapter,
    )
    assert sources["paperswithcode-evaluation-methods"].max_dataset_bytes == 80 * 1024 * 1024
    assert isinstance(sources["wikidata-ml-models"], JsonCatalogSourceAdapter)
    assert sources["wikidata-ml-models"].provider_namespace == "wikidata:entity"
    assert sources["wikidata-ml-models"].model_status.value == "documented"
    assert isinstance(sources["wikipedia-neural-network-architectures"], JsonCatalogSourceAdapter)
    assert sources["wikipedia-neural-network-architectures"].provider_namespace == "wikipedia:page"
    assert sources["wikipedia-neural-network-architectures"].model_status.value == "documented"
    assert sources["wikipedia-neural-network-architectures"].page_size == 500
    assert isinstance(sources["wikidata-neural-network-taxonomy"], JsonCatalogSourceAdapter)
    assert sources["wikidata-neural-network-taxonomy"].model_status.value == "documented"
    assert isinstance(sources["torchvision-models"], HtmlCatalogSourceAdapter)
    assert sources["torchvision-models"].provider_namespace == (
        "torchvision:model-family"
    )
    assert isinstance(
        sources["torchvision-weight-registry"],
        TorchvisionWeightRegistrySourceAdapter,
    )
    assert sources["torchvision-weight-registry"].repository == "pytorch/vision"
    assert sources["torchvision-weight-registry"].package_path == "torchvision/models"
    assert sources["torchvision-weight-registry"].max_archive_bytes == 64 * 1024 * 1024
    assert isinstance(sources["pytorch-hub-model-pages"], HtmlCatalogSourceAdapter)
    assert sources["pytorch-hub-model-pages"].provider_namespace == (
        "pytorch:hub-model-page"
    )
    assert sources["pytorch-hub-model-pages"].detail_batch_size == 10
    assert [
        rule.relation for rule in sources["pytorch-hub-model-pages"].detail_resource_rules
    ] == [
        "paper_reference",
        "official_implementation",
        "weights",
        "weights",
        "demo",
        "demo",
    ]
    assert isinstance(sources["gluoncv-classification-zoo"], HtmlCatalogSourceAdapter)
    assert sources["gluoncv-classification-zoo"].provider_namespace == (
        "gluoncv:classification-model"
    )
    assert sources["gluoncv-classification-zoo"].model_status.value == "released"
    for name in (
        "gluoncv-detection-zoo",
        "gluoncv-segmentation-zoo",
        "gluoncv-pose-zoo",
        "gluoncv-action-recognition-zoo",
        "gluoncv-depth-zoo",
    ):
        assert isinstance(sources[name], HtmlCatalogSourceAdapter)
        assert sources[name].model_status.value == "released"
    assert isinstance(sources["ultralytics-model-families"], HtmlCatalogSourceAdapter)
    assert sources["ultralytics-model-families"].provider_namespace == (
        "ultralytics:model-family"
    )
    assert isinstance(sources["torchaudio-models"], HtmlCatalogSourceAdapter)
    assert sources["torchaudio-models"].provider_namespace == "torchaudio:model-family"
    assert sources["torchaudio-models"].model_status.value == "documented"
    assert isinstance(sources["torchaudio-pipelines"], HtmlCatalogSourceAdapter)
    assert sources["torchaudio-pipelines"].provider_namespace == "torchaudio:pipeline"
    assert sources["torchaudio-pipelines"].model_status.value == "released"
    assert isinstance(
        sources["torchaudio-pipeline-registry"], TorchaudioPipelineRegistrySourceAdapter
    )
    assert sources["torchaudio-pipeline-registry"].repository == "pytorch/audio"
    assert sources["torchaudio-pipeline-registry"].max_archive_bytes == 128 * 1024 * 1024
    assert isinstance(sources["keras-applications"], HtmlCatalogSourceAdapter)
    assert sources["keras-applications"].model_status.value == "released"
    assert isinstance(sources["keras-hub-presets"], HtmlCatalogSourceAdapter)
    assert sources["keras-hub-presets"].provider_namespace == "keras-hub:model-family"
    assert sources["keras-hub-presets"].model_status.value == "documented"
    assert isinstance(
        sources["keras-hub-preset-registry"], KerasHubPresetRegistrySourceAdapter
    )
    assert sources["keras-hub-preset-registry"].repository == "keras-team/keras-hub"
    assert sources["keras-hub-preset-registry"].preset_root == "keras_hub/src/models"
    assert isinstance(sources["pytorch-geometric-models"], HtmlCatalogSourceAdapter)
    assert sources["pytorch-geometric-models"].provider_namespace == (
        "pytorch-geometric:model-family"
    )
    assert sources["pytorch-geometric-models"].model_status.value == "documented"
    assert isinstance(sources["scikit-learn-machine-learning-components"], HtmlCatalogSourceAdapter)
    assert sources["scikit-learn-machine-learning-components"].provider_namespace == (
        "scikit-learn:machine-learning-component"
    )
    assert sources["scikit-learn-machine-learning-components"].model_status.value == (
        "documented"
    )
    assert sources["scikit-learn-machine-learning-components"].rules[0].href_pattern.search(
        "https://scikit-learn.org/stable/modules/generated/"
        "sklearn.ensemble.RandomForestClassifier.html"
    )
    assert not sources["scikit-learn-machine-learning-components"].rules[0].href_pattern.search(
        "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"
    )
    for name in (
        "scikit-learn-0-18-machine-learning-components",
        "scikit-learn-0-20-machine-learning-components",
        "scikit-learn-0-22-machine-learning-components",
        "scikit-learn-0-24-machine-learning-components",
        "scikit-learn-1-0-machine-learning-components",
        "scikit-learn-1-2-machine-learning-components",
        "scikit-learn-1-4-machine-learning-components",
        "scikit-learn-1-6-machine-learning-components",
        "scikit-learn-1-8-machine-learning-components",
    ):
        assert isinstance(sources[name], HtmlCatalogSourceAdapter)
        assert sources[name].provider_namespace == "scikit-learn:machine-learning-component"
        assert sources[name].model_status.value == "documented"
        assert sources[name].rules[0].href_pattern.search(
            sources[name].url.rsplit("/", 2)[0]
            + "/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
        )
    for name in (
        "sktime-forecasting-components",
        "sktime-transformations-components",
        "sktime-classification-components",
        "sktime-regression-components",
        "sktime-clustering-components",
        "sktime-alignment-components",
        "sktime-detection-components",
        "sktime-parameter-estimation-components",
        "sktime-pipeline-components",
    ):
        assert isinstance(sources[name], HtmlCatalogSourceAdapter)
        assert sources[name].provider_namespace == "sktime:machine-learning-component"
        assert sources[name].model_status.value == "documented"
        assert sources[name].rules[0].identity_source == "href"
        assert sources[name].rules[0].raw_href_pattern.search(
            "https://www.sktime.net/docs/api-reference/"
            "sktimeforecastingarimaarima/#sktime.forecasting.arima.ARIMA"
        )
        assert sources[name].rules[0].href_pattern.search(
            "https://www.sktime.net/docs/api-reference/"
            "sktimeforecastingarimaarima"
        )
    for name in (
        "aeon-classification-components",
        "aeon-regression-components",
        "aeon-clustering-components",
        "aeon-transformations-components",
        "aeon-segmentation-components",
        "aeon-anomaly-detection-components",
    ):
        assert isinstance(sources[name], HtmlCatalogSourceAdapter)
        assert sources[name].provider_namespace == "aeon:machine-learning-component"
        assert sources[name].model_status.value == "documented"
        category = (
            name.removeprefix("aeon-")
            .removesuffix("-components")
            .replace("-", "_")
        )
        assert sources[name].rules[0].href_pattern.search(
            sources[name].url.rsplit("/", 1)[0]
            + "/auto_generated/aeon."
            + category
            + ".BaseExample.html"
        )
    assert isinstance(sources["darts-forecasting-components"], HtmlCatalogSourceAdapter)
    assert sources["darts-forecasting-components"].provider_namespace == (
        "darts:machine-learning-component"
    )
    assert sources["darts-forecasting-components"].rules[0].identity_source == "href"
    assert sources["darts-forecasting-components"].rules[0].raw_href_pattern.search(
        "https://unit8co.github.io/darts/generated_api/"
        "darts.models.forecasting.nbeats.html#darts.models.forecasting.nbeats.NBEATSModel"
    )
    assert isinstance(sources["torchdrug-models"], HtmlCatalogSourceAdapter)
    assert sources["torchdrug-models"].provider_namespace == "torchdrug:model-family"
    assert sources["torchdrug-models"].model_status.value == "documented"
    assert sources["torchdrug-models"].rules[0].entry_pattern.search(
        '<h3>Example Graph Model<a class="headerlink" href="#example">#</a></h3>'
        '<dl class="py class"><dt id="torchdrug.models.ExampleGraphModel">'
    )
    assert isinstance(sources["deepchem-models"], HtmlCatalogSourceAdapter)
    assert sources["deepchem-models"].provider_namespace == "deepchem:model-family"
    assert sources["deepchem-models"].model_status.value == "documented"
    assert isinstance(sources["fasttext-crawl-vectors"], HtmlCatalogSourceAdapter)
    assert sources["fasttext-crawl-vectors"].provider_namespace == "fasttext:cc-vector"
    assert sources["fasttext-crawl-vectors"].model_status.value == "released"
    assert sources["fasttext-crawl-vectors"].max_entries == 512
    assert len(sources["fasttext-crawl-vectors"].shared_link_rules) == 3
    assert sources["fasttext-crawl-vectors"].rules[0].entry_pattern.search(
        '<td>English: <a href="https://dl.fbaipublicfiles.com/fasttext/'
        'vectors-crawl/cc.en.300.bin.gz">bin</a>'
    )
    assert isinstance(sources["transformers-model-docs"], HtmlCatalogSourceAdapter)
    assert sources["transformers-model-docs"].max_entries == 5_000
    assert isinstance(sources["paddlenlp-transformer-families"], HtmlCatalogSourceAdapter)
    assert sources["paddlenlp-transformer-families"].provider_namespace == (
        "paddlenlp:transformer-family"
    )
    assert sources["paddlenlp-transformer-families"].model_status.value == "documented"
    assert isinstance(sources["paddlenlp-pretrained-embeddings"], HtmlCatalogSourceAdapter)
    assert sources["paddlenlp-pretrained-embeddings"].provider_namespace == (
        "paddlenlp:embedding"
    )
    assert sources["paddlenlp-pretrained-embeddings"].model_status.value == "documented"
    assert isinstance(
        sources["sentence-transformers-pretrained-models"], HtmlCatalogSourceAdapter
    )
    assert sources["sentence-transformers-pretrained-models"].provider_namespace == (
        "huggingface:model"
    )
    assert sources["sentence-transformers-pretrained-models"].model_status.value == (
        "documented"
    )
    assert isinstance(sources["spacy-models"], SpacyModelsSourceAdapter)
    assert sources["spacy-models"].max_models == 10_000
    assert sources["spacy-models"].max_releases == 100_000
    assert isinstance(sources["stanza-resources"], StanzaResourcesSourceAdapter)
    assert sources["stanza-resources"].repository == "stanfordnlp/stanza-resources"
    assert sources["stanza-resources"].max_models_per_manifest == 20_000
    assert isinstance(sources["diffusers-pipelines"], HtmlCatalogSourceAdapter)
    assert isinstance(sources["onnx-model-zoo"], OnnxModelZooSourceAdapter)
    assert sources["onnx-model-zoo"].repository_url == "https://github.com/onnx/models"
    for name, document_path in (
        (
            "tensorflow-1-detection-model-zoo",
            "research/object_detection/g3doc/tf1_detection_zoo.md",
        ),
        (
            "tensorflow-2-detection-model-zoo",
            "research/object_detection/g3doc/tf2_detection_zoo.md",
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == "tensorflow/models"
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == "tensorflow:object-detection-model"
    assert isinstance(sources["tensorflow-deeplab-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["tensorflow-deeplab-model-zoo"].repository == "tensorflow/models"
    assert sources["tensorflow-deeplab-model-zoo"].document_path == (
        "research/deeplab/g3doc/model_zoo.md"
    )
    assert sources["tensorflow-deeplab-model-zoo"].provider_namespace == (
        "tensorflow:deeplab-model"
    )
    assert sources["tensorflow-deeplab-model-zoo"].model_header_pattern.pattern == (
        "^(?:Checkpoint|Model) name$"
    )
    assert isinstance(
        sources["tensorflow-slim-pretrained-classifiers"],
        MarkdownModelTableSourceAdapter,
    )
    assert sources["tensorflow-slim-pretrained-classifiers"].repository == (
        "tensorflow/models"
    )
    assert sources["tensorflow-slim-pretrained-classifiers"].document_path == (
        "research/slim/README.md"
    )
    assert sources["tensorflow-slim-pretrained-classifiers"].provider_namespace == (
        "tensorflow:slim-pretrained-classifier"
    )
    assert isinstance(
        sources["tensorflow-model-garden-nlp-pretrained"],
        MarkdownModelTableSourceAdapter,
    )
    assert sources["tensorflow-model-garden-nlp-pretrained"].repository == (
        "tensorflow/models"
    )
    assert sources["tensorflow-model-garden-nlp-pretrained"].document_path == (
        "official/nlp/docs/pretrained_models.md"
    )
    assert sources["tensorflow-model-garden-nlp-pretrained"].provider_namespace == (
        "tensorflow:model-garden-nlp-model"
    )
    assert isinstance(sources["tensorflow-tpu-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["tensorflow-tpu-model-zoo"].repository == "tensorflow/tpu"
    assert sources["tensorflow-tpu-model-zoo"].document_path == (
        "models/official/detection/MODEL_ZOO.md"
    )
    assert sources["tensorflow-tpu-model-zoo"].model_header_pattern.pattern == "^model$"
    assert isinstance(sources["timm-model-registry"], TimmModelRegistrySourceAdapter)
    assert sources["timm-model-registry"].repository == "huggingface/pytorch-image-models"
    assert sources["timm-model-registry"].max_archive_bytes == 64 * 1024 * 1024
    assert isinstance(sources["paddle-model-center"], PaddleModelCenterSourceAdapter)
    assert sources["paddle-model-center"].repository == "PaddlePaddle/models"
    assert sources["paddle-model-center"].max_families == 1_000
    assert isinstance(
        sources["paddleclas-model-registry"], PaddleClasModelRegistrySourceAdapter
    )
    assert sources["paddleclas-model-registry"].repository == "PaddlePaddle/PaddleClas"
    assert sources["paddleclas-model-registry"].branch == "release/2.6"
    assert sources["paddleclas-model-registry"].max_entries == 10_000
    assert isinstance(
        sources["paddledetection-model-zoo"], PaddleDetectionModelZooSourceAdapter
    )
    assert sources["paddledetection-model-zoo"].repository == "PaddlePaddle/PaddleDetection"
    assert sources["paddledetection-model-zoo"].document_prefix == "configs/"
    assert sources["paddledetection-model-zoo"].max_archive_bytes == 128 * 1024 * 1024
    assert isinstance(sources["paddle3d-model-zoo"], PaddleDetectionModelZooSourceAdapter)
    assert sources["paddle3d-model-zoo"].repository == "PaddlePaddle/Paddle3D"
    assert sources["paddle3d-model-zoo"].branch == "develop"
    assert sources["paddle3d-model-zoo"].project_name == "Paddle3D"
    assert sources["paddle3d-model-zoo"].provider_namespace == "paddle3d"
    assert sources["paddle3d-model-zoo"].document_prefix == "docs/models/"
    assert isinstance(
        sources["paddlegan-tutorial-model-zoo"], PaddleGanTutorialModelZooSourceAdapter
    )
    assert sources["paddlegan-tutorial-model-zoo"].repository == "PaddlePaddle/PaddleGAN"
    assert sources["paddlegan-tutorial-model-zoo"].branch == "develop"
    assert (
        sources["paddlegan-tutorial-model-zoo"].document_prefix
        == "docs/en_US/tutorials/"
    )
    assert isinstance(sources["paddleseg-model-zoo"], PaddleDetectionModelZooSourceAdapter)
    assert sources["paddleseg-model-zoo"].project_name == "PaddleSeg"
    assert sources["paddleseg-model-zoo"].provider_namespace == "paddleseg"
    assert sources["paddleseg-model-zoo"].branch == "release/2.10"
    assert isinstance(sources["paddlevideo-model-zoo"], PaddleDetectionModelZooSourceAdapter)
    assert sources["paddlevideo-model-zoo"].project_name == "PaddleVideo"
    assert sources["paddlevideo-model-zoo"].provider_namespace == "paddlevideo"
    assert sources["paddlevideo-model-zoo"].document_prefix == "docs/en/model_zoo/"
    assert sources["paddlevideo-model-zoo"].document_suffix == ".md"
    assert isinstance(sources["espnet-model-zoo"], EspnetModelZooSourceAdapter)
    assert sources["espnet-model-zoo"].repository == "espnet/espnet_model_zoo"
    assert sources["espnet-model-zoo"].max_rows == 10_000
    for name, document_path, namespace, header_pattern in (
        (
            "fairseq-wav2vec-model-zoo",
            "examples/wav2vec/README.md",
            "fairseq:wav2vec-model",
            "^Description$",
        ),
        (
            "fairseq-wav2vec2-model-zoo",
            "examples/wav2vec/README.md",
            "fairseq:wav2vec2-model",
            "^Model$",
        ),
        (
            "fairseq-translation-model-zoo",
            "examples/translation/README.md",
            "fairseq:translation-model",
            "^Model$",
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == "facebookresearch/fairseq"
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].model_header_pattern.pattern == header_pattern
    assert isinstance(
        sources["omnilingual-asr-model-cards"],
        LineCheckpointCardCatalogSourceAdapter,
    )
    assert sources["omnilingual-asr-model-cards"].repository == (
        "facebookresearch/omnilingual-asr"
    )
    assert sources["omnilingual-asr-model-cards"].source_path == (
        "src/omnilingual_asr/cards/models/rc_models_v2.yaml"
    )
    assert sources["omnilingual-asr-model-cards"].provider_namespace == (
        "omnilingual-asr:model"
    )
    assert isinstance(
        sources["seamless-communication-model-zoo"],
        MarkdownModelTableSourceAdapter,
    )
    assert sources["seamless-communication-model-zoo"].repository == (
        "facebookresearch/seamless_communication"
    )
    assert sources["seamless-communication-model-zoo"].document_path == "README.md"
    assert sources["seamless-communication-model-zoo"].provider_namespace == (
        "seamless:model"
    )
    assert sources["seamless-communication-model-zoo"].model_header_pattern.pattern == (
        "^Model Name$"
    )
    assert isinstance(sources["sonar-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["sonar-model-zoo"].repository == "facebookresearch/SONAR"
    assert sources["sonar-model-zoo"].document_path == "README.md"
    assert sources["sonar-model-zoo"].provider_namespace == "sonar:model"
    assert sources["sonar-model-zoo"].model_header_pattern.pattern == "^model$"
    assert sources["sonar-model-zoo"].model_name_pattern is not None
    assert sources["sonar-model-zoo"].model_name_pattern.pattern == (
        "^(?:encoder|decoder|finetuned decoder)$"
    )
    assert isinstance(sources["paddleslim-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["paddleslim-model-zoo"].provider_namespace == "paddleslim:model"
    assert sources["paddleslim-model-zoo"].document_path == "docs/en/model_zoo_en.md"
    assert isinstance(sources["paddlerec-algorithm-catalog"], PaddleRecCatalogSourceAdapter)
    assert sources["paddlerec-algorithm-catalog"].repository == "PaddlePaddle/PaddleRec"
    assert sources["paddlerec-algorithm-catalog"].source_path == "README_EN.md"
    assert isinstance(sources["paddlespeech-released-models"], MarkdownModelTableSourceAdapter)
    assert sources["paddlespeech-released-models"].provider_namespace == "paddlespeech:model"
    assert "NGram" in sources["paddlespeech-released-models"].excluded_headings[0].pattern
    assert isinstance(sources["paddleocr-v2-model-list"], MarkdownModelTableSourceAdapter)
    assert sources["paddleocr-v2-model-list"].provider_namespace == "paddleocr:model"
    assert isinstance(
        sources["paddleocr-current-model-list"], PaddleOcrCurrentModelListSourceAdapter
    )
    assert sources["paddleocr-current-model-list"].source_path == (
        "docs/version3.x/model_list.md"
    )
    assert isinstance(sources["detectron2-model-zoo"], Detectron2ModelZooSourceAdapter)
    assert sources["detectron2-model-zoo"].repository == "facebookresearch/detectron2"
    assert sources["detectron2-model-zoo"].max_entries == 10_000
    for name, repository, document_path, namespace, header_pattern in (
        (
            "centernet-model-zoo",
            "xingyizhou/CenterNet",
            "readme/MODEL_ZOO.md",
            "centernet:model",
            "^Model$",
        ),
        (
            "centertrack-model-zoo",
            "xingyizhou/CenterTrack",
            "readme/MODEL_ZOO.md",
            "centertrack:model",
            "^Model$",
        ),
        (
            "alphapose-model-zoo",
            "MVIG-SJTU/AlphaPose",
            "docs/MODEL_ZOO.md",
            "alphapose:model",
            "^Model$",
        ),
        (
            "pycls-model-zoo",
            "facebookresearch/pycls",
            "MODEL_ZOO.md",
            "pycls:model",
            "^Model$",
        ),
        (
            "dino-model-zoo",
            "facebookresearch/dino",
            "README.md",
            "dino:model",
            "^arch$",
        ),
        (
            "dinov2-model-zoo",
            "facebookresearch/dinov2",
            "README.md",
            "dinov2:model",
            "^model$",
        ),
        (
            "barlow-twins-model-zoo",
            "facebookresearch/barlowtwins",
            "README.md",
            "barlow-twins:model",
            "^download$",
        ),
        (
            "swav-model-zoo",
            "facebookresearch/swav",
            "README.md",
            "swav:model",
            "^method$",
        ),
        (
            "ijepa-model-zoo",
            "facebookresearch/ijepa",
            "README.md",
            "ijepa:model",
            "^arch\\.$",
        ),
        (
            "dit-model-zoo",
            "facebookresearch/DiT",
            "README.md",
            "dit:model",
            "^DiT Model$",
        ),
        (
            "esm-pretrained-models",
            "facebookresearch/esm",
            "README.md",
            "esm:model",
            "^esm\\.pretrained\\.$",
        ),
        (
            "beit-model-zoo",
            "microsoft/unilm",
            "beit/README.md",
            "beit:model",
            "^name$",
        ),
        (
            "beitv2-model-zoo",
            "microsoft/unilm",
            "beit2/README.md",
            "beitv2:model",
            "^name$",
        ),
        (
            "vissl-model-zoo",
            "facebookresearch/vissl",
            "MODEL_ZOO.md",
            "vissl:model",
            "^Model$",
        ),
        (
            "maskformer-model-zoo",
            "facebookresearch/MaskFormer",
            "MODEL_ZOO.md",
            "maskformer:model",
            "^Name$",
        ),
        (
            "mask2former-model-zoo",
            "facebookresearch/Mask2Former",
            "MODEL_ZOO.md",
            "mask2former:model",
            "^Name$",
        ),
        (
            "detectron-legacy-model-zoo",
            "facebookresearch/Detectron",
            "MODEL_ZOO.md",
            "detectron-legacy:model",
            "^backbone$",
        ),
        (
            "detrex-model-zoo",
            "IDEA-Research/detrex",
            "docs/source/tutorials/Model_Zoo.md",
            "detrex:model",
            "^Name$",
        ),
        (
            "detic-model-zoo",
            "facebookresearch/Detic",
            "docs/MODEL_ZOO.md",
            "detic:model",
            "^Name$",
        ),
        (
            "x-anylabeling-model-zoo",
            "CVHub520/X-AnyLabeling",
            "docs/en/model_zoo.md",
            "x-anylabeling:model",
            "^Name$",
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == repository
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].model_header_pattern.pattern == header_pattern
    assert isinstance(sources["mae-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["mae-model-zoo"].repository == "facebookresearch/mae"
    assert sources["mae-model-zoo"].document_path == "README.md"
    assert sources["mae-model-zoo"].provider_namespace == "mae:model"
    assert sources["mae-model-zoo"].checkpoint_row_pattern is not None
    assert sources["mae-model-zoo"].checkpoint_row_pattern.pattern == "^pre-trained checkpoint$"
    for name, context_columns in (
        ("barlow-twins-model-zoo", (0, 1)),
        ("swav-model-zoo", (1, 2, 3)),
        ("ijepa-model-zoo", (1, 2, 3, 4)),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].identity_context_columns == context_columns
    assert sources["ijepa-model-zoo"].model_header_pattern.pattern == "^arch\\.$"
    assert isinstance(sources["dit-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["dit-model-zoo"].repository == "facebookresearch/DiT"
    assert sources["dit-model-zoo"].dataset_column == 1
    assert sources["dit-model-zoo"].model_header_pattern.pattern == "^DiT Model$"
    for name, repository, document_path, namespace, heading in (
        (
            "audiocraft-musicgen-model-cards",
            "facebookresearch/audiocraft",
            "docs/MUSICGEN.md",
            "audiocraft-musicgen:model",
            "^API$",
        ),
        (
            "audiocraft-audiogen-model-cards",
            "facebookresearch/audiocraft",
            "docs/AUDIOGEN.md",
            "audiocraft-audiogen:model",
            "^API and usage$",
        ),
        (
            "encodec-model-cards",
            "facebookresearch/encodec",
            "README.md",
            "encodec:model",
            "Transformers$",
        ),
        (
            "audiocraft-magnet-model-cards",
            "facebookresearch/audiocraft",
            "docs/MAGNET.md",
            "audiocraft-magnet:model",
            "^API$",
        ),
        (
            "audiocraft-jasco-model-cards",
            "facebookresearch/audiocraft",
            "docs/JASCO.md",
            "audiocraft-jasco:model",
            "^API$",
        ),
        (
            "audiocraft-musicgen-style-model-cards",
            "facebookresearch/audiocraft",
            "docs/MUSICGEN_STYLE.md",
            "audiocraft-musicgen-style:model",
            "^API$",
        ),
        (
            "audioseal-model-card",
            "facebookresearch/audioseal",
            "README.md",
            "audioseal:model",
            "Quick Links",
        ),
    ):
        assert isinstance(sources[name], MarkdownModelCardListSourceAdapter)
        assert sources[name].repository == repository
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].section_heading_pattern.pattern == heading
    for name, document_path, namespace in (
        ("beit-model-zoo", "beit/README.md", "beit:model"),
        ("beitv2-model-zoo", "beit2/README.md", "beitv2:model"),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == "microsoft/unilm"
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].dataset_column == 1
        assert sources[name].identity_include_heading is False
        assert sources[name].model_header_pattern.pattern == "^name$"
    assert isinstance(sources["beit3-model-zoo"], MarkdownCheckpointListSourceAdapter)
    assert sources["beit3-model-zoo"].repository == "microsoft/unilm"
    assert sources["beit3-model-zoo"].document_path == "beit3/README.md"
    assert sources["beit3-model-zoo"].provider_namespace == "beit3:model"
    assert sources["beit3-model-zoo"].section_heading_pattern.pattern == "^Download Checkpoints$"
    assert isinstance(
        sources["segment-anything-model-zoo"],
        MarkdownCheckpointListSourceAdapter,
    )
    assert sources["segment-anything-model-zoo"].repository == (
        "facebookresearch/segment-anything"
    )
    assert sources["segment-anything-model-zoo"].document_path == "README.md"
    assert sources["segment-anything-model-zoo"].provider_namespace == (
        "segment-anything:model"
    )
    assert sources["segment-anything-model-zoo"].section_heading_pattern.pattern == (
        "Model Checkpoints$"
    )
    for name, repository, namespace in (
        ("sam2-model-zoo", "facebookresearch/sam2", "sam2:model"),
        ("imagebind-model-zoo", "facebookresearch/ImageBind", "imagebind:model"),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == repository
        assert sources[name].document_path == "README.md"
        assert sources[name].provider_namespace == namespace
        assert sources[name].model_header_pattern.pattern == "^Model$"
    assert sources["sam2-model-zoo"].model_name_pattern is not None
    assert sources["sam2-model-zoo"].model_name_pattern.pattern == (
        "^(sam2(?:\\.1)?_hiera_[a-z_]+)"
    )
    for name, repository, namespace, model_column, context_column in (
        ("detr-model-zoo", "facebookresearch/detr", "detr:model", 1, 2),
        ("convnext-model-zoo", "facebookresearch/ConvNeXt", "convnext:model", 0, 1),
        (
            "swin-transformer-model-zoo",
            "microsoft/Swin-Transformer",
            "swin-transformer:model",
            0,
            1,
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == repository
        assert sources[name].document_path == "README.md"
        assert sources[name].provider_namespace == namespace
        assert sources[name].model_column == model_column
        assert sources[name].dataset_column == context_column
        assert sources[name].model_header_pattern.pattern == "^name$"
    for name, document_path, namespace, context_column in (
        ("deit-model-zoo", "README_deit.md", "deit:model", None),
        ("cait-model-zoo", "README_cait.md", "cait:model", 2),
        ("resmlp-model-zoo", "README_resmlp.md", "resmlp:model", None),
        (
            "patchconvnet-model-zoo",
            "README_patchconvnet.md",
            "patchconvnet:model",
            2,
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == "facebookresearch/deit"
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].dataset_column == context_column
        assert sources[name].model_header_pattern.pattern == "^name$"
    for name, repository, document_path, namespace, header_pattern in (
        (
            "pyslowfast-model-zoo",
            "facebookresearch/SlowFast",
            "MODEL_ZOO.md",
            "pyslowfast:model",
            "^architecture$",
        ),
        (
            "timesformer-model-zoo",
            "facebookresearch/TimeSformer",
            "README.md",
            "timesformer:model",
            "^name$",
        ),
    ):
        assert isinstance(sources[name], MarkdownModelTableSourceAdapter)
        assert sources[name].repository == repository
        assert sources[name].document_path == document_path
        assert sources[name].provider_namespace == namespace
        assert sources[name].dataset_column == 1
        assert sources[name].model_header_pattern.pattern == header_pattern
    assert sources["vissl-model-zoo"].model_column == 1
    assert sources["vissl-model-zoo"].dataset_column == 2
    assert sources["detectron-legacy-model-zoo"].dataset_column == 12
    assert isinstance(sources["openvino-model-zoo"], OpenVinoModelZooSourceAdapter)
    assert sources["openvino-model-zoo"].repository == "openvinotoolkit/open_model_zoo"
    assert sources["openvino-model-zoo"].branch == "master"
    assert isinstance(
        sources["mmengine-openmmlab-pretrained-registry"],
        StaticJsonCheckpointRegistrySourceAdapter,
    )
    assert sources["mmengine-openmmlab-pretrained-registry"].repository == (
        "open-mmlab/mmengine"
    )
    assert sources["mmengine-openmmlab-pretrained-registry"].source_path == (
        "mmengine/hub/openmmlab.json"
    )
    assert isinstance(
        sources["mmengine-mmclassification-pretrained-registry"],
        StaticJsonCheckpointRegistrySourceAdapter,
    )
    assert sources["mmengine-mmclassification-pretrained-registry"].source_path == (
        "mmengine/hub/mmcls.json"
    )
    assert isinstance(
        sources["openai-clip-checkpoint-registry"],
        StaticPythonCheckpointRegistrySourceAdapter,
    )
    assert sources["openai-clip-checkpoint-registry"].source_path == "clip/clip.py"
    assert sources["openai-clip-checkpoint-registry"].mapping_variable == "_MODELS"
    assert isinstance(
        sources["openai-whisper-checkpoint-registry"],
        StaticPythonCheckpointRegistrySourceAdapter,
    )
    assert sources["openai-whisper-checkpoint-registry"].source_path == (
        "whisper/__init__.py"
    )
    assert sources["openai-whisper-checkpoint-registry"].mapping_variable == "_MODELS"
    assert isinstance(sources["openmmlab-mmdetection"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmdetection"].repository == "open-mmlab/mmdetection"
    assert isinstance(sources["openmmlab-mmpose"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmpose"].repository == "open-mmlab/mmpose"
    assert isinstance(sources["openmmlab-mmsegmentation"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmsegmentation"].repository == "open-mmlab/mmsegmentation"
    assert isinstance(
        sources["openmmlab-mmclassification-0x"], OpenMMLabModelIndexSourceAdapter
    )
    assert sources["openmmlab-mmclassification-0x"].repository == (
        "open-mmlab/mmclassification"
    )
    assert sources["openmmlab-mmclassification-0x"].branch == "v0.25.0"
    assert isinstance(sources["openmmlab-mmpretrain"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmpretrain"].repository == "open-mmlab/mmpretrain"
    assert isinstance(sources["openmmlab-mmaction2"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmaction2"].repository == "open-mmlab/mmaction2"
    assert isinstance(sources["openmmlab-mmdetection3d"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmdetection3d"].repository == "open-mmlab/mmdetection3d"
    assert isinstance(sources["openmmlab-mmocr"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmocr"].repository == "open-mmlab/mmocr"
    assert isinstance(sources["openmmlab-mmrotate"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmrotate"].repository == "open-mmlab/mmrotate"
    assert isinstance(sources["openmmlab-mmyolo"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmyolo"].repository == "open-mmlab/mmyolo"
    assert isinstance(sources["openmmlab-mmagic"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmagic"].repository == "open-mmlab/mmagic"
    assert isinstance(sources["openmmlab-mmrazor"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmrazor"].repository == "open-mmlab/mmrazor"
    assert isinstance(sources["openmmlab-mmtracking"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmtracking"].repository == "open-mmlab/mmtracking"
    assert sources["openmmlab-mmtracking"].branch == "master"
    assert isinstance(sources["openmmlab-mmselfsup"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmselfsup"].repository == "open-mmlab/mmselfsup"
    assert isinstance(sources["openmmlab-mmfewshot"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmfewshot"].repository == "open-mmlab/mmfewshot"
    assert isinstance(sources["openmmlab-mmflow"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmflow"].repository == "open-mmlab/mmflow"
    assert sources["openmmlab-mmflow"].branch == "master"
    assert isinstance(sources["openmmlab-mmgeneration"], OpenMMLabModelIndexSourceAdapter)
    assert sources["openmmlab-mmgeneration"].repository == "open-mmlab/mmgeneration"
    assert isinstance(sources["mmhuman3d-model-zoo"], PaddleDetectionModelZooSourceAdapter)
    assert sources["mmhuman3d-model-zoo"].repository == "open-mmlab/mmhuman3d"
    assert sources["mmhuman3d-model-zoo"].branch == "main"
    assert sources["mmhuman3d-model-zoo"].provider_namespace == "mmhuman3d"
    assert sources["mmhuman3d-model-zoo"].document_prefix == "configs/"
    assert isinstance(
        sources["mmaction-legacy-model-zoo"], PaddleDetectionModelZooSourceAdapter
    )
    assert sources["mmaction-legacy-model-zoo"].repository == "open-mmlab/mmaction"
    assert sources["mmaction-legacy-model-zoo"].branch == "master"
    assert sources["mmaction-legacy-model-zoo"].document_paths == ("MODEL_ZOO.md",)
    assert isinstance(sources["mmfashion-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["mmfashion-model-zoo"].repository == "open-mmlab/mmfashion"
    assert sources["mmfashion-model-zoo"].document_path == "docs/MODEL_ZOO.md"
    assert sources["mmfashion-model-zoo"].model_header_pattern.pattern == (
        "^(?:Backbone|Model type)$"
    )
    assert isinstance(sources["openunreid-model-zoo"], MarkdownModelTableSourceAdapter)
    assert sources["openunreid-model-zoo"].repository == "open-mmlab/OpenUnReID"
    assert sources["openunreid-model-zoo"].document_path == "docs/MODEL_ZOO.md"
    assert sources["openunreid-model-zoo"].model_header_pattern.pattern == "^Method$"
    assert isinstance(sources["bioimageio"], BioImageIoSourceAdapter)
    assert sources["bioimageio"].index_url == (
        "https://bioimage-io.github.io/collection/index.json"
    )
    assert sources["bioimageio"].page_size == 20
    assert isinstance(sources["monai-model-zoo"], MonaiModelZooSourceAdapter)
    assert sources["monai-model-zoo"].repository == "Project-MONAI/model-zoo"
    assert sources["monai-model-zoo"].branch == "dev"
    assert isinstance(sources["openalex"], OpenAlexSourceAdapter)
    assert "openalex-updates" not in sources
    assert isinstance(sources["arxiv"], ArxivSourceAdapter)
    assert sources["arxiv"].url == "https://oaipmh.arxiv.org/oai"
    assert sources["arxiv"].initial_lookback_days == 7
    assert sources["arxiv"].overlap_days == 2
    assert isinstance(sources["eartharxiv"], EarthArxivSourceAdapter)
    assert sources["eartharxiv"].url == "https://eartharxiv.org/api/oai"
    assert isinstance(sources["hal"], HalSourceAdapter)
    assert sources["hal"].initial_start_date.isoformat() == "2002-09-23"
    assert isinstance(sources["plos"], PlosSourceAdapter)
    assert sources["plos"].initial_start_date.isoformat() == "2003-08-18"
    assert isinstance(sources["arxiv-complete-snapshot"], ArxivCompleteSnapshotSourceAdapter)
    assert sources["arxiv-complete-snapshot"].artifact_source == "arxiv"
    assert sources["arxiv-complete-snapshot"].page_size == 10_000
    assert sources["arxiv-complete-snapshot"].max_range_bytes == 32 * 1024 * 1024
    assert isinstance(sources["openreview"], OpenReviewSourceAdapter)
    assert sources["openreview"].api_v1_url == "https://api.openreview.net/notes"
    assert sources["openreview"].api_v2_url == "https://api2.openreview.net/notes"
    assert sources["openreview"].page_size == 1_000
    assert isinstance(sources["crossref"], CrossrefSourceAdapter)
    assert sources["crossref"].page_size == 1_000
    assert sources["crossref"].mailto == "registry@example.test"
    assert isinstance(sources["europe-pmc"], EuropePmcSourceAdapter)
    assert sources["europe-pmc"].page_size == 1_000
    assert sources["europe-pmc"].email == "registry@example.test"
    assert isinstance(sources["datacite"], DataCiteSourceAdapter)
    assert sources["datacite"].page_size == 1_000
    assert sources["datacite"].artifact_kind is None
    assert isinstance(sources["openaire-graph"], OpenAireGraphSourceAdapter)
    assert sources["openaire-graph"].url == "https://zenodo.org/api/records"
    assert sources["openaire-graph"].concept_record_id == "3516917"
    assert isinstance(
        sources["semantic-scholar-datasets"],
        SemanticScholarDatasetSourceAdapter,
    )
    assert sources["semantic-scholar-datasets"].url == (
        "https://api.semanticscholar.org/datasets/v1"
    )
    assert sources["semantic-scholar-datasets"].datasets == (
        "papers",
        "abstracts",
        "paper-ids",
    )
    assert isinstance(sources["pubmed-bulk"], PubMedBulkSourceAdapter)
    assert sources["pubmed-bulk"].baseline_url == (
        "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
    )
    assert sources["pubmed-bulk"].update_url == (
        "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
    )
    assert isinstance(sources["pmc"], PmcSourceAdapter)
    assert sources["pmc"].url == "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
    assert sources["pmc"].metadata_prefix == "pmc"
    assert sources["pmc"].set_spec == "pmc-open"
    assert isinstance(sources["commoncrawl-wet"], CommonCrawlWetSourceAdapter)
    assert sources["commoncrawl-wet"].catalog_url == (
        "https://index.commoncrawl.org/collinfo.json"
    )
    assert sources["commoncrawl-wet"].data_url == "https://data.commoncrawl.org/"
    assert isinstance(sources["gharchive"], GhArchiveSourceAdapter)
    assert sources["gharchive"].data_url == "https://data.gharchive.org/"
    assert sources["gharchive"].initial_lookback_hours == 48
    assert isinstance(
        sources["github-public-repositories"], GitHubPublicRepositoriesSourceAdapter
    )
    assert sources["github-public-repositories"].page_size == 100
    assert sources["github-public-repositories"].initial_since == 0
    assert sources["github-public-repositories"].token == ""
    assert isinstance(
        sources["software-heritage-origins"],
        SoftwareHeritageOriginSourceAdapter,
    )
    assert sources["software-heritage-origins"].settlement_age_hours == 168
    assert isinstance(sources["biorxiv"], BioRxivSourceAdapter)
    assert isinstance(sources["medrxiv"], BioRxivSourceAdapter)
    assert isinstance(
        sources["biorxiv-publications"], BioRxivPublicationSourceAdapter
    )
    assert isinstance(
        sources["medrxiv-publications"], BioRxivPublicationSourceAdapter
    )
    assert isinstance(sources["osf-preprints"], OsfPreprintSourceAdapter)
    assert sources["osf-preprints"].url == "https://api.osf.io/v2/preprints"
    assert sources["osf-preprints"].page_size == 100
    assert sources["huggingface"].token is None
    assert sources["openalex"].api_key == "oa-secret"
    assert sources["openalex"].mailto == "registry@example.test"
    assert sources["openalex"].sync_mode == "published"


def test_apple_coreml_gallery_config_keeps_each_package_link_model_scoped() -> None:
    config = next(
        config
        for config in load_source_configs(Path("config/sources.toml"))
        if config["name"] == "apple-coreml-model-gallery"
    )
    document = """
    <table>
      <tr><th>Model Name</th><th>Size</th><th>Action</th></tr>
      <tr>
        <td>DepthAnythingV2SmallF16.mlpackage</td><td>112 MB</td>
        <td><a href="https://ml-assets.apple.com/coreml/models/DepthAnythingV2SmallF16.mlpackage.zip">Download</a></td>
      </tr>
      <tr>
        <td>FastViTT8F16.mlpackage</td><td>14 MB</td>
        <td><a href="https://ml-assets.apple.com/coreml/models/FastViTT8F16.mlpackage.zip">Download</a></td>
      </tr>
    </table>
    <a href="https://github.com/apple/coremltools">gallery code</a>
    """
    source = create_source(
        config,
        client=QueuedClient(
            HttpResponse(
                status=200,
                headers={"Content-Type": "text/html"},
                body=document.encode(),
                url="https://developer.apple.com/machine-learning/models/",
            )
        ),
        clock=lambda: NOW,
    )

    page = source.fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 2
    record = page.records[0]
    assert [model.identifiers for model in record.models] == [
        (
            Identifier(
                "apple:coreml-model-package", "DepthAnythingV2SmallF16.mlpackage"
            ),
        ),
        (Identifier("apple:coreml-model-package", "FastViTT8F16.mlpackage"),),
    ]
    links_by_model = {
        model.name: [
            link
            for link in record.links
            if link.model_local_ids == (model.local_id,)
        ]
        for model in record.models
    }
    assert [
        (link.url, link.relation, link.crawl)
        for link in links_by_model["DepthAnythingV2SmallF16.mlpackage"]
    ] == [
        (
            "https://ml-assets.apple.com/coreml/models/"
            "DepthAnythingV2SmallF16.mlpackage.zip",
            "weights",
            False,
        )
    ]
    assert [
        (link.url, link.relation, link.crawl)
        for link in links_by_model["FastViTT8F16.mlpackage"]
    ] == [
        (
            "https://ml-assets.apple.com/coreml/models/FastViTT8F16.mlpackage.zip",
            "weights",
            False,
        )
    ]
    assert all("github.com" not in link.url for link in record.links)


def test_pytorch_hub_config_resolves_only_model_scoped_detail_resources() -> None:
    config = next(
        config
        for config in load_source_configs(Path("config/sources.toml"))
        if config["name"] == "pytorch-hub-model-pages"
    )
    index = """
    <a href="https://pytorch.org/hub/example-net">Example Net</a>
    """
    detail = """
    <a href="https://arxiv.org/abs/2401.12345">paper</a>
    <a href="https://github.com/example/research/blob/main/example/model.py">source</a>
    <a href="https://github.com/example/research/releases/download/v1/example.pth">weights</a>
    <a href="https://download.pytorch.org/models/example-123.pth">weights</a>
    <a href="https://colab.research.google.com/github/example/notebooks/blob/main/example.ipynb">colab</a>
    <a href="https://huggingface.co/spaces/example/example-net">demo</a>
    <a href="https://github.com/example/research">footer repository</a>
    <a href="https://github.com/pytorch/pytorch">site footer</a>
    """
    source = create_source(
        config,
        client=QueuedClient(
            HttpResponse(
                status=200,
                headers={"Content-Type": "text/html"},
                body=index.encode(),
                url="https://pytorch.org/hub/",
            ),
            HttpResponse(
                status=200,
                headers={"Content-Type": "text/html"},
                body=detail.encode(),
                url="https://pytorch.org/hub/example-net",
            ),
        ),
        clock=lambda: NOW,
    )

    catalog_page = source.fetch_page({})
    detail_page = source.fetch_page(catalog_page.next_state)

    assert catalog_page.complete is False
    assert catalog_page.authoritative_snapshot is False
    assert detail_page.complete is True
    assert detail_page.authoritative_snapshot is True
    record = detail_page.records[0]
    model_id = record.models[0].local_id
    assert {
        (link.url, link.relation, link.crawl)
        for link in record.links
        if link.model_local_ids == (model_id,)
    } == {
        ("https://pytorch.org/hub/example-net", "model_card", False),
        ("https://arxiv.org/abs/2401.12345", "paper_reference", True),
        (
            "https://github.com/example/research/blob/main/example/model.py",
            "official_implementation",
            True,
        ),
        (
            "https://github.com/example/research/releases/download/v1/example.pth",
            "weights",
            False,
        ),
        (
            "https://download.pytorch.org/models/example-123.pth",
            "weights",
            False,
        ),
        (
            "https://colab.research.google.com/github/example/notebooks/blob/main/example.ipynb",
            "demo",
            True,
        ),
        ("https://huggingface.co/spaces/example/example-net", "demo", True),
    }


def test_tensorflow_detection_zoo_configs_keep_pinned_release_rows() -> None:
    configs = {
        config["name"]: config
        for config in load_source_configs(Path("config/sources.toml"))
        if config["name"]
        in {
            "tensorflow-1-detection-model-zoo",
            "tensorflow-2-detection-model-zoo",
        }
    }
    document = (
        "# TensorFlow Detection Model Zoo\n\n"
        "## COCO-trained models\n\n"
        "Model name | Speed | COCO mAP | Outputs\n"
        "---------- | ----- | -------- | -------\n"
        "[ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/"
        "object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) | 31 | 22 | Boxes\n"
    )
    revision = "f" * 40

    for name, document_path in (
        (
            "tensorflow-1-detection-model-zoo",
            "research/object_detection/g3doc/tf1_detection_zoo.md",
        ),
        (
            "tensorflow-2-detection-model-zoo",
            "research/object_detection/g3doc/tf2_detection_zoo.md",
        ),
    ):
        source = create_source(
            configs[name],
            client=QueuedClient(
                HttpResponse(
                    status=200,
                    headers={},
                    body=json.dumps({"sha": revision}).encode(),
                    url="https://api.github.test/revision",
                ),
                HttpResponse(
                    status=200,
                    headers={"Content-Type": "text/markdown"},
                    body=document.encode(),
                    url=f"https://raw.githubusercontent.com/tensorflow/models/{revision}/{document_path}",
                ),
            ),
            clock=lambda: NOW,
        )

        page = source.fetch_page({})

        assert page.complete is True
        assert page.authoritative_snapshot is True
        assert page.upstream_count == 1
        record = page.records[0]
        assert record.identifiers == (
            Identifier(
                "tensorflow:object-detection-model",
                "COCO-trained models / ssd_mobilenet_v2_coco",
            ),
        )
        assert (
            "http://download.tensorflow.org/models/object_detection/"
            "ssd_mobilenet_v2_coco_2018_03_29.tar.gz",
            "weights",
            False,
        ) in {(link.url, link.relation, link.crawl) for link in record.links}
        assert record.releases[0].revision == revision


def test_catalog_activation_env_skips_unprovisioned_optional_source() -> None:
    sources = load_sources(
        Path("config/sources.toml"),
        client=QueuedClient(),
        clock=lambda: NOW,
        environ={"S2_API_KEY": "   "},
    )

    assert "semantic-scholar-datasets" not in sources
    assert "openalex-updates" not in sources
    assert isinstance(sources["pubmed-bulk"], PubMedBulkSourceAdapter)
    assert isinstance(sources["pmc"], PmcSourceAdapter)
    assert isinstance(sources["commoncrawl-wet"], CommonCrawlWetSourceAdapter)
    assert isinstance(
        sources["software-heritage-origins"],
        SoftwareHeritageOriginSourceAdapter,
    )


def test_vertex_sources_require_project_substitution_when_activated() -> None:
    configs = load_source_configs(Path("config/sources.toml"))
    vertex_names = {
        config["name"]
        for config in configs
        if config.get("activation_env") == "VERTEX_AI_ACCESS_TOKEN"
        and config.get("headers", {}).get("x-goog-user-project")
        == "${GOOGLE_CLOUD_PROJECT}"
    }
    assert vertex_names

    with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
        load_sources(
            Path("config/sources.toml"),
            client=QueuedClient(),
            clock=lambda: NOW,
            environ={"VERTEX_AI_ACCESS_TOKEN": "dummy-token"},
        )

    sources = load_sources(
        Path("config/sources.toml"),
        client=QueuedClient(),
        clock=lambda: NOW,
        environ={
            "VERTEX_AI_ACCESS_TOKEN": "dummy-token",
            "GOOGLE_CLOUD_PROJECT": "fixture-project",
        },
    )
    assert vertex_names <= set(sources)
    assert all(
        sources[name]._static_headers["x-goog-user-project"] == "fixture-project"
        for name in vertex_names
    )


def test_paid_openalex_update_stream_activates_only_with_its_sync_key() -> None:
    sources = load_sources(
        Path("config/sources.toml"),
        client=QueuedClient(),
        clock=lambda: NOW,
        environ={"OPENALEX_SYNC_API_KEY": "sync-secret"},
    )

    updates = sources["openalex-updates"]
    assert isinstance(updates, OpenAlexSourceAdapter)
    assert updates.sync_mode == "updated"
    assert updates.api_key == "sync-secret"
    assert updates.artifact_source == "openalex"


def test_semantic_scholar_factory_still_requires_its_key_explicitly() -> None:
    config = {
        "name": "semantic-scholar-datasets",
        "adapter": "semantic_scholar_datasets",
        "activation_env": "S2_API_KEY",
        "url": "https://api.semanticscholar.org/datasets/v1",
        "datasets": ["papers", "abstracts", "paper-ids"],
        "token_env": "S2_API_KEY",
    }

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=QueuedClient(), environ={})

    source = create_source(
        config,
        client=QueuedClient(),
        environ={"S2_API_KEY": "s2-secret"},
    )
    assert isinstance(source, SemanticScholarDatasetSourceAdapter)


def test_replicate_factory_requires_and_uses_its_explicit_token() -> None:
    config = {
        "name": "replicate-models",
        "adapter": "replicate_models",
        "activation_env": "REPLICATE_API_TOKEN",
        "url": "https://api.replicate.com/v1/models",
        "token_env": "REPLICATE_API_TOKEN",
    }

    with pytest.raises(ValueError, match="Replicate API token"):
        create_source(config, client=QueuedClient(), environ={})

    source = create_source(
        config,
        client=QueuedClient(),
        environ={"REPLICATE_API_TOKEN": "replicate-secret"},
    )
    assert isinstance(source, ReplicateModelsSourceAdapter)


def test_openai_factory_requires_and_uses_its_explicit_key() -> None:
    config = {
        "name": "openai-models",
        "adapter": "openai_models",
        "activation_env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/models",
        "token_env": "OPENAI_API_KEY",
        "public_owners": ["openai", "system"],
    }

    with pytest.raises(ValueError, match="OpenAI API key"):
        create_source(config, client=QueuedClient(), environ={})

    source = create_source(
        config,
        client=QueuedClient(),
        environ={"OPENAI_API_KEY": "openai-secret"},
    )
    assert isinstance(source, OpenAIModelsSourceAdapter)
    assert source.public_owners == ("openai", "system")


def test_google_gemini_config_requires_its_key_and_keeps_it_out_of_model_links() -> None:
    config = next(
        item
        for item in load_source_configs(Path("config/sources.toml"))
        if item["name"] == "google-gemini-models"
    )
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "models": [
                        {
                            "name": "models/gemini-fixture-001",
                            "displayName": "Gemini Fixture",
                            "baseModelId": "gemini-fixture",
                        }
                    ]
                }
            ).encode(),
            url="https://generativelanguage.googleapis.com/v1beta/models",
        )
    )

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    source = create_source(
        config,
        client=client,
        environ={"GEMINI_API_KEY": "gemini-test-secret"},
    )
    assert isinstance(source, JsonCatalogSourceAdapter)
    page = source.fetch_page({})

    assert page.records[0].canonical_url == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-fixture-001"
    )
    assert page.records[0].links[0].crawl is False
    assert client.calls[0][1] == {"pageSize": 1000, "key": "gemini-test-secret"}
    assert "gemini-test-secret" not in json.dumps(
        {"state": page.next_state, "raw": page.records[0].raw}
    )

    default_client_source = create_source(
        config,
        environ={"GEMINI_API_KEY": "gemini-test-secret"},
    )
    assert isinstance(default_client_source, JsonCatalogSourceAdapter)
    assert default_client_source.client.max_response_bytes == 4 * 1024 * 1024


def test_groq_config_requires_its_key_and_keeps_it_out_of_model_links() -> None:
    config = next(
        item
        for item in load_source_configs(Path("config/sources.toml"))
        if item["name"] == "groq-models"
    )
    client = QueuedClient(
        HttpResponse(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "openai/gpt-oss-fixture",
                            "object": "model",
                            "created": 1_700_000_000,
                            "owned_by": "OpenAI",
                            "active": True,
                        }
                    ],
                }
            ).encode(),
            url="https://api.groq.com/openai/v1/models",
        )
    )

    with pytest.raises(ValueError, match="credential environment variable is unset"):
        create_source(config, client=client, environ={})

    source = create_source(
        config,
        client=client,
        environ={"GROQ_API_KEY": "groq-test-secret"},
    )
    assert isinstance(source, JsonCatalogSourceAdapter)
    page = source.fetch_page({})

    assert page.records[0].canonical_url == (
        "https://api.groq.com/openai/v1/models/openai/gpt-oss-fixture"
    )
    assert page.records[0].links[0].crawl is False
    assert client.calls[0][2]["Authorization"] == "Bearer groq-test-secret"
    assert "groq-test-secret" not in json.dumps(
        {"state": page.next_state, "raw": page.records[0].raw}
    )


def test_catalog_rejects_invalid_activation_environment_name(tmp_path) -> None:
    catalog = tmp_path / "sources.toml"
    catalog.write_text(
        """
[[source]]
name = "conditional"
adapter = "openalex"
enabled = true
activation_env = "not a variable"
url = "https://api.openalex.org/works"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="activation_env"):
        load_sources(catalog, client=QueuedClient(), environ={})


def test_catalog_rejects_a_name_assigned_to_both_source_and_benchmark(tmp_path) -> None:
    catalog = tmp_path / "sources.toml"
    catalog.write_text(
        """
[[benchmark]]
name = "answer-key"
adapter = "csv"

[[source]]
name = "answer-key"
adapter = "csv"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both sources and benchmarks"):
        load_source_configs(catalog)


def test_huggingface_factory_requires_explicit_auth_and_private_opt_in() -> None:
    adapter = create_source(
        {
            "name": "private-hub",
            "adapter": "huggingface",
            "url": "https://huggingface.co/api/models",
            "auth_env": "HF_TOKEN",
            "include_private": True,
        },
        client=QueuedClient(),
        environ={"HF_TOKEN": "hf-secret"},
    )

    assert isinstance(adapter, HuggingFaceSourceAdapter)
    assert adapter.token == "hf-secret"
    assert adapter.include_private is True
