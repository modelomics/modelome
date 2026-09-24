from __future__ import annotations

from pathlib import Path
from typing import Any

from modelome.lake import LakeRecord, ParquetLandingZone, ShardApplicationOrder
from modelome.models import ArtifactKind
from modelome.openaire_projection import OpenAireSoftwareProjector, project_openaire_software


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
            {"urls": ["https://github.com/example/tool", "http://example.org/insecure"]},
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


def test_skips_products_without_a_safe_canonical_url() -> None:
    assert project_openaire_software(
        {
            "id": "software-no-url",
            "type": "software",
            "pids": [{"scheme": "handle", "value": "123/456"}],
            "instances": [{"urls": ["http://repository.example.org/software"]}],
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
        ),
        expected_rows=3,
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
    assert final.rows_examined == 1
    assert final.complete is True
    assert final.records == ()
