from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from modelome.export import export_public_metadata
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourceRecord,
)
from modelome.storage import Database


def test_metadata_export_preserves_joins_without_raw_source_content(tmp_path) -> None:
    store = Database(tmp_path / "store")
    store.initialize()
    model_identifier = Identifier("catalog:model", "example-model")
    release_identifier = Identifier("weights:sha256", "a" * 64)
    paper = SourceRecord(
        source_record_id="paper-1",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example.test/paper-1",
        title="Paper title that is intentionally not exported",
        raw={"abstract": "SENSITIVE SOURCE BODY"},
        text="SENSITIVE SOURCE BODY",
        models=(
            ModelHint(
                "example-model",
                "Example Model",
                identifiers=(model_identifier,),
                aliases=("Example",),
            ),
        ),
        releases=(
            ReleaseHint(
                "example-release",
                "example-model",
                version="1.0",
                identifiers=(release_identifier,),
            ),
        ),
        links=(
            Link(
                "https://code.example.test/example-model",
                relation="implementation",
                crawl=False,
            ),
        ),
    )
    code = SourceRecord(
        source_record_id="code-1",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://code.example.test/example-model",
        title="Code title that is intentionally not exported",
        raw={"readme": "SENSITIVE SOURCE BODY"},
        text="SENSITIVE SOURCE BODY",
    )
    store.ingest_page("catalog", (paper,), {"page": 1}, extractor="fixture")
    store.ingest_page("code", (code,), {"page": 1}, extractor="fixture")

    output = tmp_path / "bundle"
    receipt = export_public_metadata(
        store,
        output,
        source_configs=(
            {
                "name": "catalog",
                "adapter": "html_catalog",
                "url": "https://catalog.example.test/models",
                "license": "CC-BY-SA-4.0",
            },
            {"name": "code", "adapter": "json_catalog"},
            {"name": "never-run", "adapter": "html_catalog"},
        ),
    )

    assert receipt.output == str(output)
    assert set(receipt.files) == {
        "README.md",
        "artifact_identifiers.parquet",
        "artifact_url_links.parquet",
        "artifacts.parquet",
        "model_aliases.parquet",
        "model_artifacts.parquet",
        "model_identifiers.parquet",
        "model_releases.parquet",
        "models.parquet",
        "release_identifiers.parquet",
        "source-manifest.json",
    }
    assert (output / "README.md").is_file()
    assert (output / "export-receipt.json").is_file()
    assert "SENSITIVE SOURCE BODY" not in (output / "README.md").read_text()

    models = pq.read_table(output / "models.parquet").to_pylist()
    assert models[0]["canonical_name"] == "Example Model"
    aliases = pq.read_table(output / "model_aliases.parquet").to_pylist()
    assert {
        (row["model_id"], row["alias"], row["normalized_alias"])
        for row in aliases
    } >= {
        (models[0]["model_id"], "Example", "example"),
    }
    links = pq.read_table(output / "artifact_url_links.parquet").to_pylist()
    assert len(links) == 1
    assert links[0]["target_url"] == "https://code.example.test/example-model"
    assert links[0]["target_artifact_id"] is not None

    manifest = json.loads((output / "source-manifest.json").read_text())
    catalog = next(row for row in manifest["sources"] if row["source"] == "catalog")
    assert catalog["configured"]["license"] == "CC-BY-SA-4.0"
    assert catalog["is_configured"] is True
    assert catalog["checkpoint_observed"] is True
    assert catalog["checkpoint"]["complete"] is False
    assert catalog["checkpoint"]["records_seen"] == 1
    never_run = next(row for row in manifest["sources"] if row["source"] == "never-run")
    assert never_run["is_configured"] is True
    assert never_run["checkpoint_observed"] is False
    assert never_run["checkpoint"] is None
    assert (
        "it does not claim exhaustive historical or global coverage"
        in " ".join((output / "README.md").read_text().split())
    )
    assert b"SENSITIVE SOURCE BODY" not in b"".join(
        path.read_bytes() for path in output.iterdir() if path.is_file()
    )

    with pytest.raises(FileExistsError, match="already exists"):
        export_public_metadata(store, output, source_configs=())


def test_metadata_export_projects_only_huggingface_release_weight_filenames(tmp_path) -> None:
    store = Database(tmp_path / "store")
    store.initialize()

    def record(
        revision: str,
        filenames: list[str],
        raw_body: str,
        *,
        files_complete: bool = True,
    ) -> SourceRecord:
        return SourceRecord(
            source_record_id="lab/checkpoint",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url="https://huggingface.co/lab/checkpoint",
            title="Checkpoint",
            raw={"private_body": raw_body},
            text=raw_body,
            models=(
                ModelHint(
                    "model",
                    "Checkpoint",
                    identifiers=(Identifier("huggingface:model", "lab/checkpoint"),),
                ),
            ),
            releases=(
                ReleaseHint(
                    f"release:{revision}",
                    "model",
                    revision=revision,
                    identifiers=(
                        Identifier("huggingface:revision", f"lab/checkpoint@{revision}"),
                    ),
                    metadata={
                        "weight_files": filenames,
                        "weight_files_complete": files_complete,
                        "private_note": "PRIVATE RELEASE METADATA",
                    },
                ),
            ),
        )

    store.ingest_page(
        "huggingface",
        (
            record(
                "abc123",
                ["model.safetensors", "shards/part-00001.safetensors"],
                "PRIVATE OLD MODEL CARD BODY",
            ),
        ),
        {},
        extractor="fixture",
    )
    store.ingest_page(
        "huggingface",
        (
            record(
                "def456",
                ["model-v2.safetensors", "../outside.safetensors", "https://private.example/secret.safetensors"],
                "PRIVATE CURRENT MODEL CARD BODY",
                files_complete=False,
            ),
        ),
        {},
        extractor="fixture",
    )

    output = tmp_path / "bundle"
    export_public_metadata(store, output, source_configs=())

    releases = pq.read_table(output / "model_releases.parquet").to_pylist()
    assert {row["revision"] for row in releases} == {"abc123", "def456"}
    release_files = {row["revision"]: json.loads(row["weight_files_json"]) for row in releases}
    release_files_complete = {row["revision"]: row["weight_files_complete"] for row in releases}
    assert release_files == {
        "abc123": ["model.safetensors", "shards/part-00001.safetensors"],
        "def456": ["model-v2.safetensors"],
    }
    assert release_files_complete == {"abc123": True, "def456": False}
    revision_identifiers = {
        (row["namespace"], row["value"])
        for row in pq.read_table(output / "release_identifiers.parquet").to_pylist()
    }
    assert revision_identifiers == {
        ("huggingface:revision", "lab/checkpoint@abc123"),
        ("huggingface:revision", "lab/checkpoint@def456"),
    }
    bundle_bytes = b"".join(path.read_bytes() for path in output.iterdir() if path.is_file())
    assert b"PRIVATE OLD MODEL CARD BODY" not in bundle_bytes
    assert b"PRIVATE CURRENT MODEL CARD BODY" not in bundle_bytes
    assert b"PRIVATE RELEASE METADATA" not in bundle_bytes
    assert b"outside.safetensors" not in bundle_bytes
    assert b"private.example" not in bundle_bytes


def test_metadata_export_keeps_candidate_status_and_exact_release_versions(tmp_path) -> None:
    store = Database(tmp_path / "store")
    store.initialize()
    store.ingest_page(
        "candidate-catalog",
        (
            SourceRecord(
                source_record_id="candidate-1",
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url="https://catalog.example.test/candidate-1",
                title="Candidate record",
                raw={"private_body": "PRIVATE CANDIDATE BODY"},
                models=(
                    ModelHint(
                        "candidate",
                        "Candidate Model",
                        identifiers=(Identifier("catalog:model", "candidate-1"),),
                        status=ModelStatus.CANDIDATE,
                    ),
                    ModelHint(
                        "versioned",
                        "Versioned Model",
                        identifiers=(Identifier("catalog:model", "versioned-1"),),
                    ),
                ),
                releases=(
                    ReleaseHint(
                        "versioned-release",
                        "versioned",
                        version="v1.2.3+build.7",
                        revision="commit-0123456789abcdef",
                        metadata={
                            "private_note": "PRIVATE RELEASE BODY",
                            "source_url": "https://private.example.test/release-details",
                        },
                    ),
                ),
            ),
        ),
        {},
        extractor="fixture",
    )

    output = tmp_path / "bundle"
    export_public_metadata(store, output, source_configs=())

    models = pq.read_table(output / "models.parquet").to_pylist()
    models_by_name = {row["canonical_name"]: row for row in models}
    assert models_by_name["Candidate Model"]["status"] == "candidate"
    assert models_by_name["Versioned Model"]["status"] == "released"
    releases = pq.read_table(output / "model_releases.parquet").to_pylist()
    assert len(releases) == 1
    assert releases[0]["version"] == "v1.2.3+build.7"
    assert releases[0]["revision"] == "commit-0123456789abcdef"
    bundle_bytes = b"".join(path.read_bytes() for path in output.iterdir() if path.is_file())
    assert b"PRIVATE CANDIDATE BODY" not in bundle_bytes
    assert b"PRIVATE RELEASE BODY" not in bundle_bytes
    assert b"https://private.example.test/release-details" not in bundle_bytes
