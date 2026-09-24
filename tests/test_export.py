from __future__ import annotations

import json

import pyarrow.parquet as pq
import pytest

from modelome.export import export_public_metadata
from modelome.models import ArtifactKind, Identifier, Link, ModelHint, ReleaseHint, SourceRecord
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
    assert b"SENSITIVE SOURCE BODY" not in b"".join(
        path.read_bytes() for path in output.iterdir() if path.is_file()
    )

    with pytest.raises(FileExistsError, match="already exists"):
        export_public_metadata(store, output, source_configs=())


def test_metadata_export_projects_only_huggingface_release_weight_filenames(tmp_path) -> None:
    store = Database(tmp_path / "store")
    store.initialize()
    record = SourceRecord(
        source_record_id="lab/checkpoint",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://huggingface.co/lab/checkpoint",
        title="Checkpoint",
        raw={"private_body": "PRIVATE MODEL CARD BODY"},
        text="PRIVATE MODEL CARD BODY",
        models=(
            ModelHint(
                "model",
                "Checkpoint",
                identifiers=(Identifier("huggingface:model", "lab/checkpoint"),),
            ),
        ),
        releases=(
            ReleaseHint(
                "release",
                "model",
                revision="abc123",
                metadata={
                    "weight_files": [
                        "model.safetensors",
                        "shards/part-00001.safetensors",
                        "../outside.safetensors",
                        "https://private.example/secret.safetensors",
                    ],
                    "private_note": "PRIVATE RELEASE METADATA",
                },
            ),
        ),
    )
    store.ingest_page("huggingface", (record,), {}, extractor="fixture")

    output = tmp_path / "bundle"
    export_public_metadata(store, output, source_configs=())

    releases = pq.read_table(output / "model_releases.parquet").to_pylist()
    assert len(releases) == 1
    assert releases[0]["revision"] == "abc123"
    assert json.loads(releases[0]["weight_files_json"]) == [
        "model.safetensors",
        "shards/part-00001.safetensors",
    ]
    bundle_bytes = b"".join(path.read_bytes() for path in output.iterdir() if path.is_file())
    assert b"PRIVATE MODEL CARD BODY" not in bundle_bytes
    assert b"PRIVATE RELEASE METADATA" not in bundle_bytes
    assert b"outside.safetensors" not in bundle_bytes
    assert b"private.example" not in bundle_bytes
