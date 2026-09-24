from __future__ import annotations

import json

import pytest

from modelome.artifact_relations import ArtifactRelationMaterializer
from modelome.entries import build_entries, read_entry_seeds
from modelome.entry_seed_export import (
    assess_entry_readiness,
    export_current_entry_seeds,
    source_tags_from_configs,
)
from modelome.extract import IntroductionCueExtractor
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ReleaseHint,
    SourceRecord,
)
from modelome.storage import Database


def _declared_record() -> SourceRecord:
    return SourceRecord(
        source_record_id="paper-with-technique",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example.test/technique",
        title="A declared technique",
        raw={"unexported": "source payload"},
        identifiers=(Identifier("arxiv", "2401.00001"),),
        links=(
            Link(
                "https://github.com/example/technique",
                relation="official_implementation",
            ),
        ),
        models=(
            ModelHint(
                local_id="technique",
                name="Example Technique",
                identifiers=(Identifier("arxiv", "2401.00001"),),
            ),
        ),
        releases=(
            ReleaseHint(
                local_id="release",
                model_local_id="technique",
                version="1.0",
                identifiers=(Identifier("provider:release", "example-technique-v1"),),
            ),
        ),
    )


def _ordinary_paper() -> SourceRecord:
    return SourceRecord(
        source_record_id="ordinary-paper",
        kind=ArtifactKind.PAPER,
        canonical_url="https://papers.example.test/ordinary",
        title="An ordinary paper",
        raw={},
        identifiers=(Identifier("arxiv", "2401.00002"),),
    )


def test_current_export_streams_only_explicit_assertions_without_mutating_store(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    store.ingest_page("example", (_declared_record(), _ordinary_paper()), extractor="fixture")
    head_before = (store_path / "HEAD.json").read_bytes()

    output = tmp_path / "seeds.jsonl"
    receipt = export_current_entry_seeds(
        store_path,
        output,
        source_tags={"example": ("field:vision", "department:cs")},
    )

    assert receipt.seed_count == 1
    assert receipt.skipped_without_assertion == 1
    assert receipt.source_count == 1
    assert (store_path / "HEAD.json").read_bytes() == head_before
    seeds = read_entry_seeds(output)
    assert seeds == (
        {
            "source": "example",
            "source_record_id": "paper-with-technique",
            "canonical_url": "https://papers.example.test/technique",
            "title": "A declared technique",
            "kind": "paper",
            "identifiers": [{"namespace": "arxiv", "value": "2401.00001"}],
            "links": [
                {
                    "url": "https://github.com/example/technique",
                    "relation": "official_implementation",
                    "locator": None,
                    "crawl": True,
                }
            ],
            "models": [
                {
                    "local_id": "technique",
                    "name": "Example Technique",
                    "aliases": [],
                    "identifiers": [{"namespace": "arxiv", "value": "2401.00001"}],
                    "status": "documented",
                    "confidence": 1.0,
                    "locator": None,
                }
            ],
            "model_relations": [],
            "releases": [
                {
                    "local_id": "release",
                    "model_local_id": "technique",
                    "version": "1.0",
                    "revision": None,
                    "identifiers": [
                        {"namespace": "provider:release", "value": "example-technique-v1"}
                    ],
                    "released_at": None,
                    "metadata": {},
                    "confidence": 1.0,
                    "locator": None,
                }
            ],
            "tags": ["field:vision", "department:cs"],
        },
    )
    manifest = json.loads((tmp_path / "seeds.jsonl.manifest.json").read_text())
    assert manifest["seed_count"] == 1
    assert manifest["skipped_without_assertion"] == 1
    assert manifest["store_commit"] == receipt.commit

    with pytest.raises(ValueError, match="already exists"):
        export_current_entry_seeds(store_path, output)


def test_current_export_includes_direct_urls_found_in_a_candidate_paper_text(tmp_path) -> None:
    """Paper-local text links do not require a global join opt-in."""

    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    code_url = "https://github.com/example/technique"
    store.ingest_page(
        "papers",
        (
            SourceRecord(
                source_record_id="paper",
                kind=ArtifactKind.PAPER,
                canonical_url="https://papers.example.test/technique",
                title="A technique paper",
                raw={},
                text=(
                    "We introduce Example Technique, a neural network. "
                    f"Official implementation: {code_url}."
                ),
                models=(
                    ModelHint(
                        "technique",
                        "Example Technique",
                        identifiers=(Identifier("example:model", "technique"),),
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )

    output = tmp_path / "seeds.jsonl"
    receipt = export_current_entry_seeds(store_path, output)

    assert receipt.direct_resource_count == 1
    assert receipt.targeted_resource_count == 0
    seed = read_entry_seeds(output)[0]
    assert len(seed["links"]) == 1
    assert seed["links"][0]["url"] == code_url
    assert seed["links"][0]["relation"] == "official_implementation"
    assert seed["links"][0]["locator"].startswith("text:")
    assert seed["links"][0]["crawl"] is True
    result = build_entries((seed,))
    resource = next(item for item in result.entries[0].resources if item.url == code_url)
    assert resource.category == "code"
    assert resource.relation == "official_implementation"


def test_current_export_keeps_multi_model_links_on_their_declared_subject(tmp_path) -> None:
    """A catalog row's resource must not leak onto sibling entry subjects."""

    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    english_weights = "https://models.example.test/cc.en.300.bin.gz"
    french_weights = "https://models.example.test/cc.fr.300.bin.gz"
    store.ingest_page(
        "vectors",
        (
            SourceRecord(
                source_record_id="catalog",
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url="https://models.example.test/catalog",
                title="Vector catalog",
                raw={},
                models=(
                    ModelHint("english", "English vectors"),
                    ModelHint("french", "French vectors"),
                ),
                links=(
                    Link(
                        english_weights,
                        relation="weights",
                        crawl=False,
                        model_local_ids=("english",),
                    ),
                    Link(
                        french_weights,
                        relation="weights",
                        crawl=False,
                        model_local_ids=("french",),
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )

    output = tmp_path / "scoped-seeds.jsonl"
    export_current_entry_seeds(store_path, output)

    seed = read_entry_seeds(output)[0]
    assert {tuple(link["model_local_ids"]) for link in seed["links"]} == {
        ("english",),
        ("french",),
    }
    entries = {entry.canonical_name: entry for entry in build_entries((seed,)).entries}
    assert {resource.url for resource in entries["English vectors"].resources} == {
        "https://models.example.test/catalog",
        english_weights,
    }
    assert {resource.url for resource in entries["French vectors"].resources} == {
        "https://models.example.test/catalog",
        french_weights,
    }


def test_source_tag_config_is_optional_and_rejects_non_list_tags() -> None:
    assert source_tags_from_configs(
        (
            {"name": "papers", "entry_tags": ["field:ml", "department:cs"]},
            {"name": "code"},
        )
    ) == {"papers": ("field:ml", "department:cs"), "code": ()}
    with pytest.raises(ValueError, match="entry_tags must be a list"):
        source_tags_from_configs(({"name": "papers", "entry_tags": "field:ml"},))


def test_export_can_attach_sealed_cross_record_resources_to_model_entries(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    card_url = "https://models.example.test/technique/card"
    paper_url = "https://papers.example.test/technique"
    store.ingest_page(
        "papers",
        (
            SourceRecord(
                source_record_id="paper",
                kind=ArtifactKind.PAPER,
                canonical_url=paper_url,
                title="Supporting paper",
                raw={},
                links=(Link(card_url, relation="paper_reference", locator="metadata:card"),),
            ),
        ),
        extractor="fixture",
    )
    store.ingest_page(
        "cards",
        (
            SourceRecord(
                source_record_id="card",
                kind=ArtifactKind.MODEL_CARD,
                canonical_url=card_url,
                title="Technique card",
                raw={},
                models=(
                    ModelHint(
                        "technique",
                        "Example Technique",
                        identifiers=(Identifier("provider:model", "example-technique"),),
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )
    relation_root = tmp_path / "relations"
    ArtifactRelationMaterializer(store_path, output_root=relation_root).materialize()

    output = tmp_path / "relation-seeds.jsonl"
    receipt = export_current_entry_seeds(
        store_path,
        output,
        relation_root=relation_root,
    )

    assert receipt.seed_count == 1
    assert receipt.skipped_without_assertion == 1
    assert receipt.relation_resource_count == 1
    seed = read_entry_seeds(output)[0]
    relation_link = seed["links"][0]
    assert relation_link["url"] == paper_url
    assert relation_link["relation"] == "paper_reference"
    assert relation_link["crawl"] is False
    resolved = relation_link["resolved_artifact"]
    assert resolved["id"]
    assert resolved["revision_id"]
    assert resolved["kind"] == "paper"
    assert resolved["source"] == "papers"
    assert resolved["source_record_id"] == "paper"
    assert resolved["canonical_url"] == paper_url
    assert relation_link["relation_evidence"]["direction"] == "incoming"
    assert relation_link["relation_evidence"]["evidence_type"] == "url_mention"
    result = build_entries((seed,))
    resource = next(item for item in result.entries[0].resources if item.url == paper_url)
    assert resource.category == "paper"
    assert resource.resolved_artifact is not None
    assert resource.resolved_artifact.source == "papers"
    assert resource.relation_evidence is not None
    assert resource.relation_evidence.direction == "incoming"


def test_relation_enriched_export_requires_a_current_sealed_projection(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()

    with pytest.raises(ValueError, match="no sealed artifact-relation projection"):
        export_current_entry_seeds(
            store_path,
            tmp_path / "seeds.jsonl",
            relation_root=tmp_path / "relations",
        )


def test_targeted_export_attaches_current_paper_evidence_without_global_projection(
    tmp_path,
) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    card_url = "https://models.example.test/technique/card"
    paper_url = "https://papers.example.test/technique"
    store.ingest_page(
        "papers",
        (
            SourceRecord(
                source_record_id="paper",
                kind=ArtifactKind.PAPER,
                canonical_url=paper_url,
                title="Supporting paper",
                raw={},
                links=(Link(card_url, relation="paper_reference", locator="metadata:card"),),
            ),
        ),
        extractor="fixture",
    )
    store.ingest_page(
        "cards",
        (
            SourceRecord(
                source_record_id="card",
                kind=ArtifactKind.MODEL_CARD,
                canonical_url=card_url,
                title="Technique card",
                raw={},
                models=(
                    ModelHint(
                        "technique",
                        "Example Technique",
                        identifiers=(Identifier("provider:model", "example-technique"),),
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )
    head_before = (store_path / "HEAD.json").read_bytes()

    output = tmp_path / "targeted-seeds.jsonl"
    receipt = export_current_entry_seeds(
        store_path,
        output,
        sources=("cards",),
        link_current_resources=True,
    )

    assert receipt.seed_count == 1
    assert receipt.targeted_resource_count == 1
    assert (store_path / "HEAD.json").read_bytes() == head_before
    seed = read_entry_seeds(output)[0]
    relation_link = seed["links"][0]
    assert relation_link["url"] == paper_url
    assert relation_link["relation"] == "paper_reference"
    assert relation_link["crawl"] is False
    assert relation_link["resolved_artifact"]["source"] == "papers"
    assert relation_link["relation_evidence"]["direction"] == "incoming"
    assert relation_link["relation_evidence"]["evidence_type"] == "url_mention"
    result = build_entries((seed,))
    resource = next(item for item in result.entries[0].resources if item.url == paper_url)
    assert resource.category == "paper"
    assert resource.resolved_artifact is not None
    assert resource.resolved_artifact.source == "papers"


def test_export_uses_current_derived_model_claims_not_only_original_record_json(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    store.ingest_page(
        "papers",
        (
            SourceRecord(
                source_record_id="paper",
                kind=ArtifactKind.PAPER,
                canonical_url="https://papers.example.test/lumen",
                title="A paper",
                raw={},
                text="We introduce LumenNet, a deep neural network for image analysis.",
            ),
        ),
        extractor=IntroductionCueExtractor(),
    )

    output = tmp_path / "derived-seeds.jsonl"
    receipt = export_current_entry_seeds(store_path, output)

    assert receipt.seed_count == 1
    seed = read_entry_seeds(output)[0]
    assert seed["models"][0]["name"] == "LumenNet"
    assert seed["models"][0]["status"] == "candidate"
    assert seed["models"][0]["identifiers"] == []
    result = build_entries((seed,))
    assert result.entries[0].canonical_name == "LumenNet"


def test_export_preserves_source_declared_model_lineage(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    store.ingest_page(
        "cards",
        (
            SourceRecord(
                source_record_id="derived",
                kind=ArtifactKind.MODEL_CARD,
                canonical_url="https://models.example.test/derived",
                title="Derived model",
                raw={},
                models=(
                    ModelHint(
                        "derived",
                        "Derived Model",
                        identifiers=(Identifier("provider:model", "derived"),),
                    ),
                ),
                model_relations=(
                    ModelRelationHint(
                        "derived",
                        "base_model",
                        ModelHint(
                            "base",
                            "Base Model",
                            identifiers=(Identifier("provider:model", "base"),),
                        ),
                        locator="card:base_model",
                    ),
                ),
            ),
        ),
        extractor="fixture",
    )

    output = tmp_path / "lineage-seeds.jsonl"
    export_current_entry_seeds(store_path, output)

    relation = read_entry_seeds(output)[0]["model_relations"][0]
    assert relation["subject_local_id"] == "derived"
    assert relation["predicate"] == "base_model"
    assert relation["target"]["identifiers"] == [
        {"namespace": "provider:model", "value": "base"}
    ]


def test_readiness_audits_current_evidence_without_creating_entries(tmp_path) -> None:
    store_path = tmp_path / "store"
    store = Database(store_path)
    store.initialize()
    store.ingest_page("example", (_declared_record(),), extractor="fixture", complete=True)
    head_before = (store_path / "HEAD.json").read_bytes()

    incomplete = assess_entry_readiness(
        store_path,
        ({"name": "example", "enabled": True},),
    )

    assert incomplete["candidate_artifact_count"] == 1
    assert incomplete["model_claim_count"] == 1
    assert incomplete["gates"] == {
        "configured_sources_observed": True,
        "configured_sources_complete": True,
        "sealed_cross_record_resources_ready": False,
        "targeted_cross_record_resources_available": True,
        "current_evidence_entry_ready": True,
        "configured_full_run_ready": True,
    }
    assert (store_path / "HEAD.json").read_bytes() == head_before

    ArtifactRelationMaterializer(store_path).materialize()
    complete = assess_entry_readiness(
        store_path,
        ({"name": "example", "enabled": True},),
    )

    assert complete["relation_projection"]["current"] is True
    assert complete["gates"]["current_evidence_entry_ready"] is True
    assert complete["gates"]["configured_full_run_ready"] is True
