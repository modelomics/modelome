from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.pipeline import SyncEngine
from modelome.sources.bioimageio import BioImageIoSourceAdapter
from modelome.storage import Database

NOW = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
INDEX_URL = "https://bioimage-io.github.io/collection/index.json"
ARTIFACT_BASE = "https://hypha.aicell.io/bioimage-io/artifacts"


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[
            tuple[str, Mapping[str, Any] | None, Mapping[str, str]]
        ] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params) if params is not None else None, dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(
    body: bytes,
    *,
    url: str,
    status: int = 200,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return HttpResponse(
        status=status,
        headers=dict(headers or {}),
        body=body,
        url=url,
    )


def json_response(
    value: Any,
    *,
    url: str = INDEX_URL,
    headers: Mapping[str, str] | None = None,
) -> HttpResponse:
    return response(
        json.dumps(value, sort_keys=True).encode(),
        url=url,
        headers={"content-type": "application/json", **dict(headers or {})},
    )


def rdf_response(value: bytes, source: str) -> HttpResponse:
    return response(value, url=source, headers={"content-type": "application/yaml"})


def version(alias: str, name: str, rdf: bytes, *, created_at: str) -> dict[str, Any]:
    return {
        "version": name,
        "comment": f"publication {name}",
        "created_at": created_at,
        "sha256": hashlib.sha256(rdf).hexdigest(),
        "source": f"{ARTIFACT_BASE}/{alias}/files/bioimageio.yaml?version={name}",
    }


def item(alias: str, *versions: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": f"bioimage-io/{alias}",
        "type": "model",
        "versions": list(versions),
    }


def index(*items: Mapping[str, Any], timestamp: str = "2026-09-04T03:37:46") -> dict[str, Any]:
    all_items = [
        {
            "id": "bioimage-io/non-model-resource",
            "type": "dataset",
            "versions": [],
        },
        *items,
    ]
    return {
        "count_per_type": {"dataset": 1, "model": len(items)},
        "items": all_items,
        "timestamp": timestamp,
        "total": len(all_items),
    }


def detail(alias: str, name: str, *, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": f"bioimage-io/{alias}",
        "type": "model",
        "current_version": "v1",
        "created_at": 1_756_000_000,
        "last_modified": 1_756_000_100,
        "file_count": 8,
        "download_count": 42,
        "manifest": {
            "type": "model",
            "format_version": "0.5.11",
            "name": name,
            "description": "A documented neural image model.",
            **dict(manifest or {}),
        },
    }


def adapter(client: QueuedClient, **kwargs: Any) -> BioImageIoSourceAdapter:
    return BioImageIoSourceAdapter(client=client, clock=lambda: NOW, **kwargs)


def scan_all(source: BioImageIoSourceAdapter) -> list[Any]:
    pages = []
    state: Mapping[str, Any] = {}
    while True:
        page = source.fetch_page(state)
        pages.append(page)
        if page.complete:
            return pages
        state = page.next_state


def test_enumerates_every_exact_model_version_and_retains_rdf_weight_and_link_evidence() -> None:
    first_v0 = b"format_version: 0.5.11\ntype: model\nname: First Model\n"
    first_v1 = b"format_version: 0.5.11\ntype: model\nname: First Model Updated\n"
    second_v0 = b"format_version: 0.5.11\ntype: model\nname: Second Model\n"
    first_v0_control = version(
        "first-model", "v0", first_v0, created_at="2025-07-01T22:53:02Z"
    )
    first_v1_control = version(
        "first-model", "v1", first_v1, created_at="2026-08-01T08:00:00Z"
    )
    second_v0_control = version(
        "second-model", "v0", second_v0, created_at="2026-08-02T08:00:00Z"
    )
    catalog = index(
        item("first-model", first_v0_control, first_v1_control),
        item("second-model", second_v0_control),
    )
    rich_manifest = {
        "id": "10.5281/zenodo.1234567",
        "parent": {"id": "bioimage-io/parent-model", "version": "v2"},
        "inputs": [
            {"id": "normalized", "output_of": "bioimage-io/preprocessing-model"}
        ],
        "new_version": "bioimage-io/replacement-model",
        "evaluations": [{"model_id": "bioimage-io/evaluated-model"}],
        "documentation": {"source": "README.md", "sha256": "a" * 64},
        "git_repo": "https://github.com/example-lab/neural-image-model",
        "links": [
            {
                "label": "project page",
                "url": "https://models.example.org/neural-image-model",
            }
        ],
        "cite": [
            {
                "text": "Model publication",
                "doi": "https://doi.org/10.1038/s41592-026-12345-6",
            }
        ],
        "weights": {
            "pytorch_state_dict": {
                "source": {"source": "weights.pt", "sha256": "b" * 64},
                "sha256": "b" * 64,
                "architecture": {
                    "source": "architecture.py",
                    "callable": "Network",
                },
            },
            "onnx": {
                "source": "weights.onnx",
                "sha256": "c" * 64,
                "parent": "pytorch_state_dict",
                "external_data": {
                    "source": "weights.onnx.data",
                    "sha256": "d" * 64,
                },
                "attachments": {
                    "files": [
                        {"source": "onnx-metadata.bin", "sha256": "e" * 64},
                        "quantization.json",
                    ]
                },
            },
        },
    }
    client = QueuedClient(
        json_response(catalog, headers={"ETag": '"catalog-v1"'}),
        json_response(
            detail("first-model", "First Model", manifest=rich_manifest),
            url=f"{ARTIFACT_BASE}/first-model",
        ),
        rdf_response(first_v0, first_v0_control["source"]),
        json_response(
            detail("first-model", "First Model Updated"),
            url=f"{ARTIFACT_BASE}/first-model",
        ),
        rdf_response(first_v1, first_v1_control["source"]),
        json_response(catalog, headers={"ETag": '"catalog-v1"'}),
        json_response(
            detail("second-model", "Second Model"),
            url=f"{ARTIFACT_BASE}/second-model",
        ),
        rdf_response(second_v0, second_v0_control["source"]),
    )

    pages = scan_all(adapter(client, page_size=2))

    assert len(pages) == 2
    assert pages[0].complete is False
    assert pages[0].upstream_count == 3
    assert pages[0].next_state["operation_offset"] == 2
    assert pages[1].complete is True
    assert pages[1].upstream_count == 3
    assert pages[1].next_state["watermark"] == "2026-09-04T03:37:46"
    assert pages[1].next_state["etag"] == '"catalog-v1"'
    assert len(pages[1].next_state["known_records"]) == 3

    records = [record for page in pages for record in page.records]
    assert [record.source_record_id for record in records] == [
        "bioimage-io/first-model@v0",
        "bioimage-io/first-model@v1",
        "bioimage-io/second-model@v0",
    ]
    first = records[0]
    assert first.kind is ArtifactKind.MODEL_CARD
    assert first.raw["rdf"]["text"] == first_v0.decode()
    assert first.raw["rdf"]["sha256"] == hashlib.sha256(first_v0).hexdigest()
    assert first.identifiers == (
        Identifier(
            "bioimageio:resource-version",
            "bioimage-io/first-model@v0",
        ),
    )
    assert first.models[0].identifiers == (
        Identifier("bioimageio:model", "bioimage-io/first-model"),
        Identifier("doi", "10.5281/zenodo.1234567"),
    )
    assert first.models[0].aliases == (
        "first-model",
        "10.5281/zenodo.1234567",
    )
    assert len(first.model_relations) == 4
    parent_relation, pipeline_relation, successor_relation, evaluation_relation = (
        first.model_relations
    )
    assert parent_relation.subject_local_id == "bioimage-io/first-model@v0#model"
    assert parent_relation.predicate == "derived_from"
    assert parent_relation.target.identifiers == (
        Identifier("bioimageio:model", "bioimage-io/parent-model"),
    )
    assert parent_relation.target.locator == "$.artifact.manifest.parent.id"
    assert parent_relation.locator == "$.artifact.manifest.parent.id"
    assert pipeline_relation.predicate == "pipeline_input_from"
    assert pipeline_relation.target.identifiers == (
        Identifier("bioimageio:model", "bioimage-io/preprocessing-model"),
    )
    assert pipeline_relation.locator == "$.artifact.manifest.inputs[0].output_of"
    assert successor_relation.predicate == "superseded_by"
    assert successor_relation.target.identifiers == (
        Identifier("bioimageio:model", "bioimage-io/replacement-model"),
    )
    assert successor_relation.locator == "$.artifact.manifest.new_version"
    assert evaluation_relation.predicate == "evaluates"
    assert evaluation_relation.target.identifiers == (
        Identifier("bioimageio:model", "bioimage-io/evaluated-model"),
    )
    assert evaluation_relation.locator == "$.artifact.manifest.evaluations[0].model_id"
    assert first.releases[0].identifiers == (
        Identifier(
            "bioimageio:version",
            "bioimage-io/first-model@v0",
        ),
    )
    assert first.releases[0].revision == hashlib.sha256(first_v0).hexdigest()
    assert first.releases[0].metadata["weight_formats"] == [
        "onnx",
        "pytorch_state_dict",
    ]
    assert first.releases[0].metadata["weight_sources"] == [
        {
            "format": "onnx",
            "source": "weights.onnx",
            "sha256": "c" * 64,
            "external_data_source": "weights.onnx.data",
            "external_data_sha256": "d" * 64,
            "attachment_files": [
                {"source": "onnx-metadata.bin", "sha256": "e" * 64},
                {"source": "quantization.json", "sha256": None},
            ],
        },
        {
            "format": "pytorch_state_dict",
            "source": "weights.pt",
            "sha256": "b" * 64,
        },
    ]
    assert Identifier("doi", "10.1038/s41592-026-12345-6") not in first.identifiers
    assert Identifier(
        "github:repository", "example-lab/neural-image-model"
    ) not in first.identifiers
    relation_urls = {(link.relation, link.url, link.crawl) for link in first.links}
    assert (
        "weights",
        f"{ARTIFACT_BASE}/first-model/files/weights.pt?version=v0",
        False,
    ) in relation_urls
    assert (
        "weights",
        f"{ARTIFACT_BASE}/first-model/files/weights.onnx?version=v0",
        False,
    ) in relation_urls
    assert (
        "weights",
        f"{ARTIFACT_BASE}/first-model/files/weights.onnx.data?version=v0",
        False,
    ) in relation_urls
    assert (
        "weights",
        f"{ARTIFACT_BASE}/first-model/files/onnx-metadata.bin?version=v0",
        False,
    ) in relation_urls
    assert (
        "weights",
        f"{ARTIFACT_BASE}/first-model/files/quantization.json?version=v0",
        False,
    ) in relation_urls
    assert (
        "implementation",
        f"{ARTIFACT_BASE}/first-model/files/architecture.py?version=v0",
        True,
    ) in relation_urls
    assert (
        "documentation",
        f"{ARTIFACT_BASE}/first-model/files/README.md?version=v0",
        True,
    ) in relation_urls
    assert ("publication", "https://doi.org/10.1038/s41592-026-12345-6", True) in relation_urls
    assert client.calls[1][1] == {"version": "v0"}
    assert client.calls[3][1] == {"version": "v1"}
    assert client.calls[5][0] == INDEX_URL


def test_incremental_scan_fetches_only_new_versions_and_tombstones_removed_versions() -> None:
    keep_rdf = b"type: model\nname: Kept\n"
    remove_rdf = b"type: model\nname: Removed\n"
    add_rdf = b"type: model\nname: Added\n"
    keep = version("kept", "v0", keep_rdf, created_at="2026-01-01T00:00:00Z")
    remove = version("removed", "v0", remove_rdf, created_at="2026-01-02T00:00:00Z")
    old_index = index(item("kept", keep), item("removed", remove), timestamp="2026-09-03")
    initial_client = QueuedClient(
        json_response(old_index),
        json_response(detail("kept", "Kept"), url=f"{ARTIFACT_BASE}/kept"),
        rdf_response(keep_rdf, keep["source"]),
        json_response(detail("removed", "Removed"), url=f"{ARTIFACT_BASE}/removed"),
        rdf_response(remove_rdf, remove["source"]),
    )
    completed = scan_all(adapter(initial_client))[0].next_state

    add = version("added", "v0", add_rdf, created_at="2026-09-04T00:00:00Z")
    new_index = index(item("kept", keep), item("added", add), timestamp="2026-09-04")
    incremental_client = QueuedClient(
        json_response(new_index),
        json_response(detail("added", "Added"), url=f"{ARTIFACT_BASE}/added"),
        rdf_response(add_rdf, add["source"]),
    )

    page = adapter(incremental_client).fetch_page(completed)

    assert page.complete is True
    assert page.upstream_count == 2
    assert [record.source_record_id for record in page.records] == [
        "bioimage-io/added@v0",
        "bioimage-io/removed@v0",
    ]
    assert page.records[0].deleted is False
    assert page.records[1].deleted is True
    assert page.records[1].models == ()
    assert page.records[1].releases == ()
    assert [call[0] for call in incremental_client.calls] == [
        INDEX_URL,
        f"{ARTIFACT_BASE}/added",
        add["source"],
    ]
    assert set(page.next_state["known_records"]) == {
        "bioimage-io/kept@v0",
        "bioimage-io/added@v0",
    }


def test_conditional_not_modified_is_a_complete_noop() -> None:
    rdf = b"type: model\nname: Stable\n"
    control = version("stable", "v0", rdf, created_at="2026-01-01T00:00:00Z")
    initial_client = QueuedClient(
        json_response(
            index(item("stable", control)),
            headers={"ETag": '"stable-index"', "Last-Modified": "yesterday"},
        ),
        json_response(detail("stable", "Stable"), url=f"{ARTIFACT_BASE}/stable"),
        rdf_response(rdf, control["source"]),
    )
    state = scan_all(adapter(initial_client))[0].next_state
    client = QueuedClient(
        response(
            b"",
            url=INDEX_URL,
            status=304,
            headers={"ETag": '"stable-index"'},
        )
    )

    page = adapter(client).fetch_page(state)

    assert page.complete is True
    assert page.records == ()
    assert page.upstream_count == 1
    assert client.calls[0][2]["If-None-Match"] == '"stable-index"'
    assert client.calls[0][2]["If-Modified-Since"] == "yesterday"


def test_rdf_checksum_mismatch_is_quarantined_and_holds_operation_page() -> None:
    expected_rdf = b"type: model\nname: Expected\n"
    actual_rdf = b"type: model\nname: Tampered\n"
    control = version(
        "checksum-model", "v0", expected_rdf, created_at="2026-01-01T00:00:00Z"
    )
    client = QueuedClient(
        json_response(index(item("checksum-model", control))),
        json_response(
            detail("checksum-model", "Checksum Model"),
            url=f"{ARTIFACT_BASE}/checksum-model",
        ),
        rdf_response(actual_rdf, control["source"]),
    )

    page = adapter(client).fetch_page({})

    assert page.records == ()
    assert page.issues[0].stage == "source_normalize"
    assert "RDF SHA-256 mismatch" in page.issues[0].error
    assert page.retry_state is not None
    assert page.retry_state["operation_offset"] == 0
    assert "index_digest" in page.retry_state


def test_index_change_during_scan_requests_restart_from_completed_checkpoint() -> None:
    a_rdf = b"type: model\nname: A\n"
    b_rdf = b"type: model\nname: B\n"
    a = version("a-model", "v0", a_rdf, created_at="2026-01-01T00:00:00Z")
    b = version("b-model", "v0", b_rdf, created_at="2026-01-02T00:00:00Z")
    original = index(item("a-model", a), item("b-model", b), timestamp="one")
    changed = index(item("a-model", a), item("b-model", b), timestamp="two")
    client = QueuedClient(
        json_response(original),
        json_response(detail("a-model", "A"), url=f"{ARTIFACT_BASE}/a-model"),
        rdf_response(a_rdf, a["source"]),
        json_response(changed),
    )
    source = adapter(client, page_size=1)

    first = source.fetch_page({})
    second = source.fetch_page(first.next_state)

    assert first.complete is False
    assert second.complete is False
    assert second.records == ()
    assert second.issues[0].stage == "source_consistency"
    assert second.retry_state is not None
    assert "index_digest" not in second.retry_state
    assert "operation_offset" not in second.retry_state


def test_restart_after_index_change_preserves_exact_version_addition_and_tombstone() -> None:
    v0_rdf = b"type: model\nname: Stable version\n"
    v1_rdf = b"type: model\nname: Historical version\n"
    v2_rdf = b"type: model\nname: First replacement\n"
    v3_rdf = b"type: model\nname: Final replacement\n"
    v0 = version("versioned", "v0", v0_rdf, created_at="2026-01-01")
    v1 = version("versioned", "v1", v1_rdf, created_at="2026-01-02")
    v2 = version("versioned", "v2", v2_rdf, created_at="2026-01-03")
    v3 = version("versioned", "v3", v3_rdf, created_at="2026-01-04")
    initial_index = index(item("versioned", v0, v1), timestamp="initial")
    initial_client = QueuedClient(
        json_response(initial_index),
        json_response(detail("versioned", "Stable version"), url=f"{ARTIFACT_BASE}/versioned"),
        rdf_response(v0_rdf, v0["source"]),
        json_response(initial_index),
        json_response(detail("versioned", "Historical version"), url=f"{ARTIFACT_BASE}/versioned"),
        rdf_response(v1_rdf, v1["source"]),
    )
    initial_source = adapter(initial_client, page_size=1)
    first_initial = initial_source.fetch_page({})
    assert first_initial.complete is False
    completed_state = initial_source.fetch_page(first_initial.next_state).next_state

    first_update_index = index(item("versioned", v0, v2), timestamp="first-update")
    first_update_client = QueuedClient(
        json_response(first_update_index),
        json_response(detail("versioned", "First replacement"), url=f"{ARTIFACT_BASE}/versioned"),
        rdf_response(v2_rdf, v2["source"]),
    )
    first_update_source = adapter(first_update_client, page_size=1)
    partial_update = first_update_source.fetch_page(completed_state)
    assert partial_update.complete is False
    assert [record.source_record_id for record in partial_update.records] == [
        "bioimage-io/versioned@v2"
    ]

    final_index = index(item("versioned", v0, v3), timestamp="final-update")
    stale_source = adapter(
        QueuedClient(json_response(final_index)),
        page_size=1,
    )
    restarted = stale_source.fetch_page(partial_update.next_state)
    assert restarted.complete is False
    assert restarted.records == ()
    assert restarted.retry_state is not None
    assert "index_digest" not in restarted.retry_state
    assert set(restarted.retry_state["known_records"]) == {
        "bioimage-io/versioned@v0",
        "bioimage-io/versioned@v1",
    }

    final_client = QueuedClient(
        json_response(final_index),
        json_response(detail("versioned", "Final replacement"), url=f"{ARTIFACT_BASE}/versioned"),
        rdf_response(v3_rdf, v3["source"]),
        json_response(final_index),
    )
    final_source = adapter(final_client, page_size=1)
    final_upsert = final_source.fetch_page(restarted.retry_state)
    final_tombstone = final_source.fetch_page(final_upsert.next_state)

    assert final_upsert.complete is False
    assert [record.source_record_id for record in final_upsert.records] == [
        "bioimage-io/versioned@v3"
    ]
    assert final_tombstone.complete is True
    assert len(final_tombstone.records) == 1
    tombstone = final_tombstone.records[0]
    assert tombstone.source_record_id == "bioimage-io/versioned@v1"
    assert tombstone.deleted is True
    assert final_tombstone.next_state["known_records"].keys() == {
        "bioimage-io/versioned@v0",
        "bioimage-io/versioned@v3",
    }


def test_rejects_catalog_count_drift_and_cross_origin_rdf_source() -> None:
    rdf = b"type: model\nname: Unsafe\n"
    control = version("unsafe", "v0", rdf, created_at="2026-01-01T00:00:00Z")
    bad_count = index(item("unsafe", control))
    bad_count["total"] = 99
    with pytest.raises(ValueError, match="index.total 99"):
        adapter(QueuedClient(json_response(bad_count))).fetch_page({})

    control["source"] = "https://attacker.invalid/model.yaml?version=v0"
    with pytest.raises(ValueError, match="outside artifact origin"):
        adapter(QueuedClient(json_response(index(item("unsafe", control))))).fetch_page({})


def test_rejects_duplicate_resource_ids_even_when_versions_do_not_overlap() -> None:
    first_rdf = b"type: model\nname: Duplicate Resource\n"
    second_rdf = b"type: model\nname: Duplicate Resource\nversion: second\n"
    duplicated = index(
        item("duplicate", version("duplicate", "v1", first_rdf, created_at="2026-01-01")),
        item("duplicate", version("duplicate", "v2", second_rdf, created_at="2026-01-02")),
    )

    with pytest.raises(ValueError, match="duplicate model resource id"):
        adapter(QueuedClient(json_response(duplicated))).fetch_page({})


def test_bounds_page_size_and_accepts_catalog_only_unversioned_model() -> None:
    with pytest.raises(ValueError, match="page_size must not exceed 100"):
        adapter(QueuedClient(), page_size=101)

    catalog = index(item("unversioned"))
    client = QueuedClient(
        json_response(catalog),
        json_response(
            detail("unversioned", "Catalog Only"),
            url=f"{ARTIFACT_BASE}/unversioned",
        ),
    )

    page = adapter(client).fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 1
    assert page.records[0].source_record_id == "bioimage-io/unversioned@unversioned"
    assert page.records[0].raw["rdf"]["text"] is None
    assert page.records[0].releases == ()
    assert len(client.calls) == 2


def test_resolved_evidence_ingests_into_parquet_store(tmp_path: Any) -> None:
    rdf = b"type: model\nname: Persisted Model\n"
    control = version("persisted", "v0", rdf, created_at="2026-01-01T00:00:00Z")
    client = QueuedClient(
        json_response(index(item("persisted", control))),
        json_response(
            detail("persisted", "Persisted Model"),
            url=f"{ARTIFACT_BASE}/persisted",
        ),
        rdf_response(rdf, control["source"]),
    )
    source = adapter(client)
    database = Database(tmp_path / "store")
    database.initialize()

    outcome = SyncEngine(database, {source.name: source}).sync()[0]

    assert outcome.status == "complete"
    artifact = database.table_rows("artifacts")[0]
    assert artifact["source_record_id"] == "bioimage-io/persisted@v0"
    assert database.get_source_state(source.name)["watermark"] == "2026-09-04T03:37:46"
