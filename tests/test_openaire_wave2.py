from __future__ import annotations

from pathlib import Path
from typing import Any

from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder
from modelome.models import ArtifactKind
from modelome.openaire_projection import (
    OpenAireSoftwareProjector,
    project_openaire_relation,
    project_openaire_software,
)
from modelome.storage import Database


def _software() -> dict[str, Any]:
    return {
        "id": "openaire-product-123",
        "type": "software",
        "title": [{"value": "Explicit Software Tool", "language": "en"}],
        "pids": [
            {"scheme": "doi", "value": "10.5281/zenodo.12345"},
            {"scheme": "github", "value": "https://github.com/example/tool"},
        ],
        "instances": [
            {"urls": ["https://github.com/example/tool", "http://example.org/provider"]},
            {"url": [{"url": "https://repository.example.org/download/tool.zip"}]},
        ],
    }


def test_projects_exact_software_type_pids_and_provider_urls() -> None:
    record = project_openaire_software(_software())

    assert record is not None
    assert record.kind is ArtifactKind.OTHER
    assert record.title == "Explicit Software Tool"
    assert record.canonical_url == "https://doi.org/10.5281/zenodo.12345"
    assert {item.namespace for item in record.identifiers} == {
        "openaire:graph-product",
        "openaire-pid:doi",
        "openaire-pid:github",
    }
    assert [item.url for item in record.links] == [
        "https://github.com/example/tool",
        "http://example.org/provider",
        "https://repository.example.org/download/tool.zip",
    ]
    assert all(item.relation == "provider_resource" for item in record.links)
    assert record.raw["model_classification_performed"] is False


def test_does_not_infer_software_or_model_from_text() -> None:
    payload = _software()
    payload["type"] = "otherresearchproduct"
    payload["title"] = "A software model"

    assert project_openaire_software(payload) is None


def test_supports_legacy_singular_pid_and_instance_url_fields() -> None:
    record = project_openaire_software(
        {
            "id": "legacy-software-1",
            "type": "software",
            "title": "Legacy format",
            "pid": [{"scheme": "doi", "value": "10.1234/tool"}],
            "instance": [{"url": ["https://code.example.org/tool"]}],
        }
    )

    assert record is not None
    assert record.canonical_url == "https://doi.org/10.1234/tool"
    assert [item.url for item in record.links] == ["https://code.example.org/tool"]


def test_projects_openAIRE_software_repository_and_documentation_fields() -> None:
    record = project_openaire_software(
        {
            "id": "openaire-software-567",
            "type": "software",
            "mainTitle": "Explicit repository fields",
            "pid": [{"scheme": "doi", "value": "10.5281/zenodo.567"}],
            "codeRepositoryUrl": "https://github.com/example/project",
            "documentationUrl": ["https://docs.example.org/project"],
        }
    )

    assert record is not None
    assert record.title == "Explicit repository fields"
    assert [(item.relation, item.url) for item in record.links] == [
        ("code_repository", "https://github.com/example/project"),
        ("documentation", "https://docs.example.org/project"),
    ]
    assert record.raw["code_repository_urls"] == ["https://github.com/example/project"]
    assert record.raw["documentation_urls"] == ["https://docs.example.org/project"]


def test_projects_current_plural_documentation_urls_schema_field() -> None:
    record = project_openaire_software(
        {
            "id": "openaire-software-plural-docs",
            "type": "software",
            "documentationUrls": [
                "https://docs.example.org/guide",
                "https://docs.example.org/api",
            ],
        }
    )

    assert record is not None
    assert [(link.url, link.relation, link.locator) for link in record.links] == [
        ("https://docs.example.org/guide", "documentation", "documentationUrls[0]"),
        ("https://docs.example.org/api", "documentation", "documentationUrls[1]"),
    ]
    assert record.raw["documentation_urls"] == [
        "https://docs.example.org/guide",
        "https://docs.example.org/api",
    ]


def test_preserves_http_instance_artifact_urls_from_graph_schema() -> None:
    record = project_openaire_software(
        {
            "id": "openaire-software-http-resource",
            "type": "software",
            "mainTitle": "Software with an HTTP artifact URL",
            "instance": [{"url": ["http://repository.example.org/software/archive.zip"]}],
        }
    )

    assert record is not None
    assert record.canonical_url == (
        "https://api.openaire.eu/graph/v3/research-products/openaire-software-http-resource"
    )
    assert [(item.url, item.relation, item.crawl) for item in record.links] == [
        (
            "http://repository.example.org/software/archive.zip",
            "provider_resource",
            True,
        )
    ]


def test_duplicate_url_occurrences_keep_distinct_source_locators() -> None:
    repeated_url = "https://github.com/example/repository"
    record = project_openaire_software(
        {
            "id": "openaire-software-repeated-links",
            "type": "software",
            "documentationUrl": [
                "https://docs.example.org/project",
                "https://docs.example.org/project",
                "ftp://invalid.example.org/docs",
            ],
            "instance": [{"url": [repeated_url, repeated_url]}],
        }
    )

    assert record is not None
    assert [(item.relation, item.url, item.locator) for item in record.links] == [
        ("documentation", "https://docs.example.org/project", "documentationUrl[0]"),
        ("documentation", "https://docs.example.org/project", "documentationUrl[1]"),
        ("provider_resource", repeated_url, "instance[0].url[0]"),
        ("provider_resource", repeated_url, "instance[0].url[1]"),
    ]


def test_exact_code_repository_url_enters_existing_crawl_frontier(tmp_path: Path) -> None:
    database = Database(tmp_path / "registry")
    database.initialize()
    record = project_openaire_software(
        {
            "id": "openaire-github-project",
            "type": "software",
            "mainTitle": "Project with a GitHub repository",
            "pid": [{"scheme": "url", "value": "https://github.com/example/project"}],
            "codeRepositoryUrl": "https://github.com/example/project",
        }
    )
    assert record is not None
    assert record.links[0].relation == "code_repository"
    assert record.links[0].crawl is True

    database.ingest_page("openaire-graph", (record,), {}, extractor="fixture")

    assert [item["url"] for item in database.list_frontier(status="pending")] == [
        "https://github.com/example/project"
    ]


def test_preserves_instance_identifiers_as_version_links_not_product_identity() -> None:
    record = project_openaire_software(
        {
            "id": "openaire-software-instance-pids",
            "type": "software",
            "mainTitle": "Versioned software",
            "pid": [{"scheme": "openaire", "value": "software-product-pid"}],
            "instance": [
                {
                    "pid": [{"scheme": "doi", "value": "10.5281/zenodo.111"}],
                    "alternateIdentifier": [
                        {"scheme": "url", "value": "https://archive.example.org/v1"}
                    ],
                }
            ],
        }
    )

    assert record is not None
    assert [(item.namespace, item.value) for item in record.identifiers] == [
        ("openaire:graph-product", "openaire-software-instance-pids"),
        ("openaire-pid:openaire", "software-product-pid"),
    ]
    assert [(item.relation, item.url, item.crawl) for item in record.links] == [
        ("instance_identifier", "https://doi.org/10.5281/zenodo.111", False),
        ("instance_identifier", "https://archive.example.org/v1", False),
    ]
    assert record.raw["instance_pids"] == [
        {"scheme": "doi", "value": "10.5281/zenodo.111"},
        {"scheme": "url", "value": "https://archive.example.org/v1"},
    ]


def test_keeps_pid_only_software_identity_when_doi_provides_canonical_url() -> None:
    record = project_openaire_software(
        {
            "id": "software-no-provider-url",
            "type": "software",
            "pids": [{"scheme": "doi", "value": "10.5281/zenodo.98765"}],
        }
    )

    assert record is not None
    assert record.canonical_url == "https://doi.org/10.5281/zenodo.98765"
    assert record.links == ()
    assert record.title == "software-no-provider-url"


def test_uses_graph_resolver_for_products_without_safe_external_urls() -> None:
    record = project_openaire_software(
        {
            "id": "software-no-url",
            "type": "software",
            "pids": [{"scheme": "handle", "value": "123/456"}],
            "instances": [{"urls": ["ftp://repository.example.org/software"]}],
        }
    )
    assert record is not None
    assert record.canonical_url == (
        "https://api.openaire.eu/graph/v3/research-products/software-no-url"
    )
    assert record.links == ()


def test_projects_explicit_software_to_dataset_relation_as_non_model_evidence() -> None:
    record = project_openaire_relation(
        {
            "source": "openaire-software-1",
            "sourceType": "software",
            "target": "openaire-dataset-2",
            "targetType": "dataset",
            "relType": {"type": "relationship", "name": "IsSupplementTo"},
            "validated": True,
        }
    )

    assert record is not None
    assert record.source_record_id.startswith("openaire-graph:relation-evidence:")
    assert record.identifiers[0].namespace == "openaire:graph-product"
    assert record.identifiers[0].value == "openaire-software-1"
    assert record.links[0].url == (
        "https://api.openaire.eu/graph/v3/research-products/openaire-dataset-2"
    )
    assert record.links[0].relation == "openaire_related_issupplementto"
    assert record.links[0].crawl is False
    assert record.raw["target_product_type"] == "dataset"
    assert record.raw["relation_validated"] is True
    assert record.raw["model_classification_performed"] is False


def test_projects_nested_relation_nodes_and_rejects_non_product_targets() -> None:
    record = project_openaire_relation(
        {
            "source": {"id": "openaire-software-3", "type": "software"},
            "target": {"id": "openaire-publication-4", "type": "publication"},
            "reltype": {"type": "citation", "name": "Cites"},
        }
    )
    assert record is not None
    assert record.raw["target_product_type"] == "publication"
    assert record.links[0].url.endswith("/openaire-publication-4")

    assert project_openaire_relation(
        {
            "source": "openaire-software-3",
            "sourceType": "software",
            "target": "openaire-project-5",
            "targetType": "project",
            "relation": "produces",
        }
    ) is None


def test_landed_projection_is_bounded_and_advances_over_unselected_rows(tmp_path: Path) -> None:
    lake = ParquetLandingZone(tmp_path / "lake")
    receipt = lake.commit_shard(
        source="openaire-graph",
        dataset="graph",
        release="20428976",
        shard="openaire-graph:file:20428976:software.tar",
        control_sha256="a" * 64,
        upstream_sha256="b" * 64,
        upstream_url="https://zenodo.org/records/20428976/files/software.tar",
        upstream_bytes=123,
        application_order=ShardApplicationOrder.snapshot(0),
        records=(
            LakeRecord("row:publication", {"id": "publication-1", "type": "publication"}),
            LakeRecord("row:software", _software()),
            LakeRecord("row:other", {"id": "other-1", "type": "otherresearchproduct"}),
            LakeRecord(
                "row:relation",
                {
                    "source": "openaire-product-123",
                    "sourceType": "software",
                    "target": "dataset-1",
                    "targetType": "dataset",
                    "relType": {"type": "relationship", "name": "IsSupplementTo"},
                },
            ),
        ),
        expected_rows=4,
        batch_rows=1,
    )
    lake.seal_release(
        source="openaire-graph",
        dataset="graph",
        release="20428976",
        expected_shards={"openaire-graph:file:20428976:software.tar": receipt},
    )

    projector = OpenAireSoftwareProjector(lake)
    page = projector.page("20428976", max_rows=2, arrow_batch_size=1)
    assert page.rows_examined == 2
    assert page.next_row == 2
    assert page.complete is False
    assert [record.source_record_id for record in page.records] == [
        "openaire-graph:software:openaire-product-123"
    ]
    final = projector.page("20428976", start_row=page.next_row, max_rows=2)
    assert final.rows_examined == 2
    assert final.complete is True
    assert len(final.records) == 1
    assert final.records[0].raw["record_type"] == "openaire_related_product_evidence"
