from __future__ import annotations

import json

import pytest

from modelome.entries import (
    EntryIdentifier,
    build_entries,
    plan_entry_seed,
    read_entry_seeds,
    source_record_to_entry_seed,
    write_entry_bundle,
)
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ReleaseHint,
    SourceRecord,
)


def _seed(
    *,
    source_record_id: str = "paper-1",
    name: str = "ExampleNet",
    identifier: tuple[str, str] = ("arxiv", "2401.00001"),
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "source": "example",
        "source_record_id": source_record_id,
        "canonical_url": f"https://example.test/{source_record_id}",
        "title": "Example paper",
        "kind": "paper",
        "tags": tags or ["field:computer-vision", "department:computer-science"],
        "links": [
            {
                "url": "https://arxiv.org/abs/2401.00001",
                "relation": "paper_reference",
                "locator": "metadata:paper",
            },
            {
                "url": "https://github.com/example/example-net",
                "relation": "official_implementation",
                "locator": "metadata:code",
            },
            {
                "url": "https://download.example.test/example-net.safetensors",
                "relation": "weights",
                "locator": "metadata:weights",
            },
        ],
        "models": [
            {
                "local_id": "model",
                "name": name,
                "aliases": ["ENet"],
                "identifiers": [{"namespace": identifier[0], "value": identifier[1]}],
            }
        ],
    }


def test_plan_attaches_every_declared_resource_and_does_not_fetch_weights() -> None:
    seed = _seed()
    seed["releases"] = [
        {
            "local_id": "release",
            "model_local_id": "model",
            "version": "1.0",
            "identifiers": [{"namespace": "provider:release", "value": "example-v1"}],
            "metadata": {"format": "safetensors"},
        }
    ]
    plan = plan_entry_seed(seed)

    actions = plan["actions"]
    assert actions[0]["action"] == "upsert_entry"
    attached = [item for item in actions if item["action"] == "attach_resource"]
    assert {(item["url"], item["category"]) for item in attached} == {
        ("https://arxiv.org/abs/2401.00001", "paper"),
        ("https://download.example.test/example-net.safetensors", "weights"),
        ("https://example.test/paper-1", "paper"),
        ("https://github.com/example/example-net", "code"),
    }
    resolved = {item["url"] for item in actions if item["action"] == "resolve_resource"}
    assert "https://download.example.test/example-net.safetensors" not in resolved
    assert "https://github.com/example/example-net" in resolved
    release_action = next(item for item in actions if item["action"] == "attach_release")
    assert release_action["release"]["identifiers"] == [
        {"namespace": "provider:release", "value": "example-v1"}
    ]


def test_builder_merges_only_exact_identifiers_and_keeps_tags_as_non_identity_metadata() -> None:
    same_identity = _seed(source_record_id="code-1", name="Example Net", tags=["domain:vision"])
    different_identity = _seed(
        source_record_id="paper-2",
        name="ExampleNet",
        identifier=("arxiv", "2401.00002"),
    )

    result = build_entries([_seed(), same_identity, different_identity])

    assert result.seed_count == 3
    assert result.candidate_count == 3
    assert len(result.entries) == 2
    merged = next(entry for entry in result.entries if len(entry.members) == 2)
    assert merged.canonical_name == "Example Net"
    assert merged.aliases == ("ENet", "ExampleNet")
    assert merged.tags == (
        "department:computer-science",
        "domain:vision",
        "field:computer-vision",
    )
    assert {resource.category for resource in merged.resources} == {"code", "paper", "weights"}


def test_paper_identifiers_are_provenance_not_technique_identity() -> None:
    first = _seed(source_record_id="paper-a")
    second = _seed(source_record_id="paper-b")
    first["identifiers"] = [{"namespace": "doi", "value": "10.1000/example"}]
    second["identifiers"] = [{"namespace": "doi", "value": "10.1000/example"}]
    first["models"] = [{"local_id": "subject", "name": "Example Technique"}]
    second["models"] = [{"local_id": "subject", "name": "Example Technique"}]

    result = build_entries([first, second])

    assert len(result.entries) == 2
    assert all(entry.identifiers == () for entry in result.entries)
    assert {
        member.artifact_identifiers[0].key
        for entry in result.entries
        for member in entry.members
    } == {"doi:10.1000/example"}


def test_builder_retains_model_lineage_and_resolves_only_exact_targets() -> None:
    base = _seed(
        source_record_id="base",
        name="Base Model",
        identifier=("provider:model", "base"),
    )
    derived = _seed(
        source_record_id="derived",
        name="Derived Model",
        identifier=("provider:model", "derived"),
    )
    derived["model_relations"] = [
        {
            "subject_local_id": "model",
            "predicate": "base_model",
            "target": {
                "local_id": "external-base",
                "name": "Base Model",
                "identifiers": [{"namespace": "provider:model", "value": "base"}],
            },
            "confidence": 0.9,
            "locator": "card:base_model",
        }
    ]

    plan = plan_entry_seed(derived)
    relation_action = next(
        action for action in plan["actions"] if action["action"] == "attach_model_relation"
    )
    assert relation_action["predicate"] == "base_model"
    assert relation_action["target"]["name"] == "Base Model"

    result = build_entries([base, derived])
    assert len(result.entries) == 2
    by_name = {entry.canonical_name: entry for entry in result.entries}
    relation = by_name["Derived Model"].model_relations[0]
    assert relation.predicate == "base_model"
    assert relation.target_entry_id == by_name["Base Model"].id
    assert result.manifest()["model_relation_count"] == 1


def test_source_record_converter_preserves_declarations_for_a_later_entry_build() -> None:
    record = SourceRecord(
        source_record_id="row-7",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://huggingface.co/example/model",
        title="Example model",
        raw={},
        identifiers=(Identifier("huggingface:model", "example/model"),),
        links=(Link("https://github.com/example/model", relation="official_implementation"),),
        models=(
            ModelHint(
                local_id="model",
                name="Example model",
                identifiers=(Identifier("huggingface:model", "example/model"),),
            ),
        ),
        releases=(
            ReleaseHint(
                local_id="release",
                model_local_id="model",
                version="1.0",
                identifiers=(Identifier("huggingface:revision", "example/model@main"),),
                metadata={"precision": "fp16"},
                locator="card:release",
            ),
        ),
        model_relations=(
            ModelRelationHint(
                subject_local_id="model",
                predicate="base_model",
                target=ModelHint(
                    local_id="base",
                    name="Example base",
                    identifiers=(Identifier("huggingface:model", "example/base"),),
                ),
                locator="card:base_model",
            ),
        ),
    )

    seed = source_record_to_entry_seed(
        record,
        source="huggingface",
        tags=("field:language", "department:computer-science"),
    )
    result = build_entries([seed])

    assert result.entries[0].identifiers[0].key == "huggingface:model:example/model"
    assert result.entries[0].tags == ("department:computer-science", "field:language")
    assert (
        result.entries[0].members[0].artifact_identifiers[0].key
        == "huggingface:model:example/model"
    )
    assert result.entries[0].members[0].status == "documented"
    assert {resource.category for resource in result.entries[0].resources} == {"code", "resource"}
    assert (
        result.entries[0].releases[0].identifiers[0].key
        == "huggingface:revision:example/model@main"
    )
    assert result.entries[0].releases[0].metadata_json == '{"precision":"fp16"}'
    relation = result.entries[0].model_relations[0]
    assert relation.predicate == "base_model"
    assert relation.target_name == "Example base"
    assert relation.target_entry_id is None


def test_entry_builder_classifies_source_declared_paper_code_and_model_artifact_links() -> None:
    record = SourceRecord(
        source_record_id="model-zoo-row",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://catalog.example.test/model-zoo",
        title="ExampleNet",
        raw={},
        models=(
            ModelHint(
                local_id="model",
                name="ExampleNet",
                identifiers=(Identifier("example:model", "example-net"),),
            ),
        ),
        links=(
            Link("https://arxiv.org/abs/2401.12345", relation="paper_reference"),
            Link(
                "https://github.com/example/example-net",
                relation="source_implementation",
            ),
            Link(
                "https://drive.example.test/file/d/example-net",
                relation="model_artifact",
                crawl=False,
            ),
        ),
    )

    result = build_entries([source_record_to_entry_seed(record, source="model-zoo")])

    entry = result.entries[0]
    assert {(resource.url, resource.category) for resource in entry.resources} == {
        ("https://catalog.example.test/model-zoo", "resource"),
        ("https://arxiv.org/abs/2401.12345", "paper"),
        ("https://github.com/example/example-net", "code"),
        ("https://drive.example.test/file/d/example-net", "weights"),
    }
    plan = plan_entry_seed(source_record_to_entry_seed(record, source="model-zoo"))
    assert "https://drive.example.test/file/d/example-net" not in {
        action["url"]
        for action in plan["actions"]
        if action["action"] == "resolve_resource"
    }


def test_builder_joins_catalog_and_resolved_card_through_the_card_identity() -> None:
    card_url = (
        "https://docs.aws.amazon.com/bedrock/latest/userguide/"
        "model-card-example-aurora-net-2.html"
    )
    card_identifier = Identifier("aws:bedrock-model-card", "example-aurora-net-2")
    catalog = SourceRecord(
        source_record_id="aws-bedrock-model-cards:catalog",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
        title="Models at a glance",
        raw={},
        models=(
            ModelHint(
                local_id="catalog:example-aurora-net-2",
                name="AuroraNet 2",
                identifiers=(card_identifier,),
            ),
        ),
        links=(
            Link(
                card_url,
                relation="model_card",
                model_local_ids=("catalog:example-aurora-net-2",),
            ),
        ),
    )
    resolved_card = SourceRecord(
        source_record_id="aws-bedrock:example-aurora-net-2",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=card_url,
        title="AuroraNet 2",
        raw={},
        models=(
            ModelHint(
                local_id="aws-bedrock:example-aurora-net-2#model",
                name="AuroraNet 2",
                identifiers=(
                    card_identifier,
                    Identifier("aws:bedrock:model", "example.aurora-net-2-v1:0"),
                ),
            ),
        ),
        links=(
            Link(
                "https://provider.example/cards/aurora-net-2",
                relation="documentation_reference",
                locator="html:a[0]",
            ),
        ),
    )

    result = build_entries(
        (
            source_record_to_entry_seed(catalog, source="aws-bedrock-model-cards"),
            source_record_to_entry_seed(resolved_card, source="frontier"),
        )
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert {identifier.key for identifier in entry.identifiers} == {
        "aws:bedrock-model-card:example-aurora-net-2",
        "aws:bedrock:model:example.aurora-net-2-v1:0",
    }
    assert {member.source for member in entry.members} == {
        "aws-bedrock-model-cards",
        "frontier",
    }
    assert {
        (resource.url, resource.relation) for resource in entry.resources
    } >= {
        (card_url, "model_card"),
        ("https://provider.example/cards/aurora-net-2", "documentation_reference"),
    }


def test_builder_joins_openai_public_catalog_and_snapshot_card_exactly() -> None:
    card_url = "https://developers.openai.com/api/docs/models/gpt-4"
    model_identifier = Identifier("openai:model", "gpt-4")
    catalog = SourceRecord(
        source_record_id="openai-model-documentation:catalog",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url="https://developers.openai.com/api/docs/models/all",
        title="All models",
        raw={},
        models=(
            ModelHint(
                local_id="catalog:gpt-4",
                name="gpt-4",
                identifiers=(model_identifier,),
            ),
        ),
        links=(
            Link(
                card_url,
                relation="model_documentation",
                model_local_ids=("catalog:gpt-4",),
            ),
        ),
    )
    resolved_card = SourceRecord(
        source_record_id="openai-documentation:gpt-4",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=card_url,
        title="GPT-4",
        raw={},
        models=(
            ModelHint(
                local_id="openai-documentation:gpt-4#model",
                name="GPT-4",
                aliases=("gpt-4",),
                identifiers=(model_identifier,),
            ),
        ),
        releases=(
            ReleaseHint(
                local_id="openai-documentation:gpt-4#model:snapshot:gpt-4-0613",
                model_local_id="openai-documentation:gpt-4#model",
                version="gpt-4-0613",
                identifiers=(Identifier("openai:model-snapshot", "gpt-4-0613"),),
            ),
        ),
    )

    result = build_entries(
        (
            source_record_to_entry_seed(catalog, source="openai-model-documentation"),
            source_record_to_entry_seed(resolved_card, source="frontier"),
        )
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert entry.canonical_name == "GPT-4"
    assert {identifier.key for identifier in entry.identifiers} == {"openai:model:gpt-4"}
    assert {member.source for member in entry.members} == {
        "openai-model-documentation",
        "frontier",
    }
    assert {(resource.url, resource.relation) for resource in entry.resources} >= {
        (card_url, "model_documentation"),
    }
    assert entry.releases[0].identifiers == (
        EntryIdentifier("openai:model-snapshot", "gpt-4-0613"),
    )


def test_builder_rejects_a_release_that_does_not_name_a_declared_subject() -> None:
    seed = _seed()
    seed["releases"] = [
        {
            "local_id": "release",
            "model_local_id": "missing-subject",
            "identifiers": [],
            "metadata": {},
        }
    ]

    with pytest.raises(ValueError, match="absent from the seed"):
        build_entries([seed])


def test_jsonl_input_and_bundle_writer_are_deterministic_and_refuse_overwrite(tmp_path) -> None:
    input_path = tmp_path / "seeds.jsonl"
    input_path.write_text(json.dumps(_seed()) + "\n", encoding="utf-8")

    result = build_entries(read_entry_seeds(input_path))
    output = tmp_path / "entry-bundle"
    receipt = write_entry_bundle(result, output)

    assert receipt["entry_count"] == 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "modelome-entry-corpus-v1"
    row = json.loads((output / "entries.jsonl").read_text(encoding="utf-8"))
    assert row["canonical_name"] == "ExampleNet"
    try:
        write_entry_bundle(result, output)
    except ValueError as error:
        assert "already exists" in str(error)
    else:  # pragma: no cover - failure diagnostic
        raise AssertionError("entry bundle overwrite was not rejected")
