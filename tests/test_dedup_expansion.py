from __future__ import annotations

from modelome.entries import build_entries


def _seed(record_id: str, identifiers: list[dict[str, str]]) -> dict[str, object]:
    return {
        "source": "catalog",
        "source_record_id": record_id,
        "canonical_url": f"https://catalog.example/{record_id}",
        "title": record_id,
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": "Example model",
                "identifiers": identifiers,
            }
        ],
    }


def test_versioned_arxiv_ids_reconcile_the_same_paper_across_records() -> None:
    result = build_entries(
        [
            _seed("v1", [{"namespace": "arxiv", "value": "2401.01234v1"}]),
            _seed("v3", [{"namespace": "arxiv", "value": "2401.01234V3"}]),
        ]
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert len(entry.members) == 2
    assert [identifier.key for identifier in entry.identifiers] == ["arxiv:2401.01234"]


def test_arxiv_version_namespace_remains_revision_specific() -> None:
    result = build_entries(
        [
            _seed("v1", [{"namespace": "arxiv:version", "value": "2401.01234v1"}]),
            _seed("v2", [{"namespace": "arxiv:version", "value": "2401.01234v2"}]),
        ]
    )

    assert len(result.entries) == 2


def test_doi_url_and_prefix_forms_reconcile_across_sources() -> None:
    first = _seed("crossref", [{"namespace": "doi", "value": "10.5555/Example.DOI"}])
    first["source"] = "crossref"
    second = _seed(
        "openalex",
        [{"namespace": "doi", "value": "https://doi.org/10.5555/example.doi"}],
    )
    second["source"] = "openalex"

    result = build_entries([first, second])

    assert len(result.entries) == 1
    assert {member.source for member in result.entries[0].members} == {"crossref", "openalex"}
    assert [item.key for item in result.entries[0].identifiers] == ["doi:10.5555/example.doi"]


def test_legacy_arxiv_category_case_does_not_split_identity() -> None:
    result = build_entries(
        [
            _seed("source-upper", [{"namespace": "arxiv", "value": "HEP-TH/9901001v1"}]),
            _seed("source-lower", [{"namespace": "arxiv", "value": "hep-th/9901001"}]),
        ]
    )

    assert len(result.entries) == 1
    assert [item.key for item in result.entries[0].identifiers] == ["arxiv:hep-th/9901001"]


def test_explicit_exact_mirror_identifier_joins_within_one_source() -> None:
    mirror_owner = _seed(
        "catalog-model",
        [{"namespace": "catalog:model", "value": "model-a"}],
    )
    mirror_owner["model_relations"] = [
        {
            "subject_local_id": "model",
            "predicate": "mirrors",
            "target": {
                "local_id": "model-b",
                "name": "Catalog alias",
                "identifiers": [
                    {"namespace": "catalog:model", "value": "model-b"}
                ],
            },
        }
    ]
    mirror_target = _seed(
        "catalog-model-mirror",
        [{"namespace": "catalog:model", "value": "model-b"}],
    )

    result = build_entries([mirror_owner, mirror_target])

    assert len(result.entries) == 1
    assert {member.source_record_id for member in result.entries[0].members} == {
        "catalog-model",
        "catalog-model-mirror",
    }


def test_exact_huggingface_model_card_link_joins_provider_documentation_entry() -> None:
    provider = _seed(
        "mistral-docs",
        [{"namespace": "mistral:model-documentation", "value": "mistral-large"}],
    )
    provider["source"] = "mistral-model-documentation"
    provider["links"] = [
        {
            "url": "https://huggingface.co/mistralai/Mistral-Large-2407",
            "relation": "model_card",
        }
    ]
    hub = _seed(
        "mistralai/Mistral-Large-2407",
        [{"namespace": "huggingface:model", "value": "mistralai/Mistral-Large-2407"}],
    )
    hub["source"] = "huggingface"

    result = build_entries([provider, hub])

    assert len(result.entries) == 1
    assert {member.source for member in result.entries[0].members} == {
        "mistral-model-documentation",
        "huggingface",
    }


def test_shared_huggingface_model_card_link_does_not_merge_multiple_provider_models() -> None:
    provider = _seed(
        "provider-catalog",
        [{"namespace": "provider:model", "value": "model-a"}],
    )
    provider["source"] = "provider-catalog"
    provider["models"].append(
        {"local_id": "model-b", "name": "Model B", "identifiers": []}
    )
    provider["links"] = [
        {
            "url": "https://huggingface.co/org/shared-card",
            "relation": "model_card",
        }
    ]
    hub = _seed(
        "org/shared-card",
        [{"namespace": "huggingface:model", "value": "org/shared-card"}],
    )
    hub["source"] = "huggingface"

    result = build_entries([provider, hub])

    assert len(result.entries) == 3


def test_same_direct_checkpoint_url_joins_cross_source_records_and_keeps_provenance() -> None:
    first = _seed("source-a", [])
    first["source"] = "catalog-a"
    first["models"][0]["name"] = "First catalog name"
    first["links"] = [
        {"url": "https://weights.example/model.bin", "relation": "checkpoint"}
    ]
    second = _seed("source-b", [])
    second["source"] = "catalog-b"
    second["models"][0]["name"] = "Second catalog name"
    second["links"] = [
        {"url": "https://weights.example/model.bin?utm_source=mirror", "relation": "checkpoint"}
    ]

    result = build_entries([first, second])

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert {member.source for member in entry.members} == {"catalog-a", "catalog-b"}
    assert {resource.source for resource in entry.resources} == {"catalog-a", "catalog-b"}


def test_generic_weights_link_does_not_join_checkpoint_identity() -> None:
    first = _seed("source-a", [])
    first["source"] = "catalog-a"
    first["links"] = [
        {"url": "https://weights.example/shared.bin", "relation": "checkpoint"}
    ]
    second = _seed("source-b", [])
    second["source"] = "catalog-b"
    second["links"] = [
        {"url": "https://weights.example/shared.bin", "relation": "weights"}
    ]

    result = build_entries([first, second])

    assert len(result.entries) == 2


def test_checkpoint_url_does_not_join_ambiguous_multi_model_catalog_link() -> None:
    first = _seed("source-a", [])
    first["source"] = "catalog-a"
    first["models"].append(
        {"local_id": "second", "name": "Unrelated model", "identifiers": []}
    )
    first["links"] = [
        {"url": "https://weights.example/shared.bin", "relation": "checkpoint"}
    ]
    second = _seed("source-b", [])
    second["source"] = "catalog-b"
    second["links"] = [
        {"url": "https://weights.example/shared.bin", "relation": "weights"}
    ]

    result = build_entries([first, second])

    assert len(result.entries) == 3


def test_entry_seed_attaches_evidence_only_checkpoint_by_exact_record_identifier() -> None:
    model_record = _seed(
        "paper-record",
        [{"namespace": "provider:model", "value": "example/model"}],
    )
    model_record["identifiers"] = [{"namespace": "doi", "value": "10.1000/paper"}]
    evidence_record = {
        "source": "biorxiv-jats-supplementary",
        "source_record_id": "paper-record:jats-model-resources",
        "canonical_url": "https://example.org/paper.xml",
        "title": "Paper supplementary resources",
        "kind": "paper",
        "identifiers": [{"namespace": "doi", "value": "10.1000/paper"}],
        "links": [
            {
                "url": "https://example.org/model-checkpoint.safetensors",
                "relation": "model_artifact",
                "locator": "jats.supplementary-material[0]",
                "crawl": False,
            }
        ],
        "models": [],
    }

    result = build_entries([model_record, evidence_record])

    assert len(result.entries) == 1
    resources = result.entries[0].resources
    assert any(
        resource.url == "https://example.org/model-checkpoint.safetensors"
        and resource.source == "biorxiv-jats-supplementary"
        and resource.category == "weights"
        for resource in resources
    )


def test_entry_seed_does_not_attach_evidence_without_exact_record_identifier() -> None:
    model_record = _seed(
        "paper-record",
        [{"namespace": "provider:model", "value": "example/model"}],
    )
    model_record["identifiers"] = [{"namespace": "doi", "value": "10.1000/paper-a"}]
    evidence_record = {
        "source": "zenodo-model-api",
        "source_record_id": "record:42",
        "canonical_url": "https://zenodo.org/records/42",
        "title": "Zenodo model",
        "kind": "catalog_record",
        "identifiers": [{"namespace": "doi", "value": "10.1000/paper-b"}],
        "links": [
            {
                "url": "https://zenodo.org/records/42/files/model.safetensors",
                "relation": "checkpoint",
            }
        ],
        "models": [],
    }

    result = build_entries([model_record, evidence_record])

    assert len(result.entries) == 1
    assert not any(
        resource.url == "https://zenodo.org/records/42/files/model.safetensors"
        for resource in result.entries[0].resources
    )


def test_evidence_resource_is_not_attached_to_ambiguous_models_in_one_paper() -> None:
    paper = _seed("paper-record", [])
    paper["identifiers"] = [{"namespace": "doi", "value": "10.1000/multi-model"}]
    paper["models"].extend(
        [
            {
                "local_id": "model-b",
                "name": "Second model",
                "identifiers": [{"namespace": "provider:model", "value": "second"}],
            }
        ]
    )
    evidence = {
        "source": "biorxiv-jats-supplementary",
        "source_record_id": "paper-record:jats-model-resources",
        "canonical_url": "https://example.org/paper.xml",
        "title": "Paper supplementary resources",
        "kind": "paper",
        "identifiers": [{"namespace": "doi", "value": "10.1000/multi-model"}],
        "links": [
            {
                "url": "https://example.org/shared-model-weights.safetensors",
                "relation": "model_artifact",
            }
        ],
        "models": [],
    }

    result = build_entries([paper, evidence])

    assert len(result.entries) == 2
    assert all(
        resource.url != "https://example.org/shared-model-weights.safetensors"
        for entry in result.entries
        for resource in entry.resources
    )


def test_imported_checkpoint_evidence_does_not_become_identity_join_key() -> None:
    model_from_paper = _seed(
        "paper-record",
        [{"namespace": "provider:model", "value": "paper-model"}],
    )
    model_from_paper["identifiers"] = [{"namespace": "doi", "value": "10.1000/paper"}]
    evidence = {
        "source": "zenodo-model-api",
        "source_record_id": "record:42",
        "canonical_url": "https://zenodo.org/records/42",
        "title": "Zenodo model",
        "kind": "catalog_record",
        "identifiers": [{"namespace": "doi", "value": "10.1000/paper"}],
        "links": [
            {
                "url": "https://weights.example/shared.safetensors",
                "relation": "checkpoint",
            }
        ],
        "models": [],
    }
    other_model = _seed(
        "other-source-record",
        [{"namespace": "other:model", "value": "unrelated"}],
    )
    other_model["source"] = "other-catalog"
    other_model["links"] = [
        {
            "url": "https://weights.example/shared.safetensors",
            "relation": "checkpoint",
        }
    ]

    result = build_entries([model_from_paper, evidence, other_model])

    assert len(result.entries) == 2
    assert any(
        resource.source == "zenodo-model-api"
        for entry in result.entries
        for resource in entry.resources
    )


def test_mirror_and_paper_resource_reconciliation_is_seed_order_independent() -> None:
    first = _seed(
        "source-a-record",
        [{"namespace": "provider:model", "value": "model-a"}],
    )
    first["source"] = "source-a"
    first["identifiers"] = [{"namespace": "doi", "value": "10.1000/order-check"}]
    first["model_relations"] = [
        {
            "subject_local_id": "model",
            "predicate": "mirrors",
            "target": {
                "local_id": "mirror",
                "name": "Mirrored model",
                "identifiers": [{"namespace": "provider:model", "value": "model-b"}],
            },
        }
    ]
    second = _seed(
        "source-b-record",
        [{"namespace": "provider:model", "value": "model-b"}],
    )
    second["source"] = "source-b"
    second["links"] = [
        {"url": "https://weights.example/order-checkpoint.bin", "relation": "checkpoint"}
    ]
    third = _seed(
        "source-c-record",
        [{"namespace": "provider:model", "value": "model-c"}],
    )
    third["source"] = "source-c"
    third["links"] = [
        {"url": "https://weights.example/order-checkpoint.bin", "relation": "checkpoint"}
    ]
    evidence = {
        "source": "paper-resource-index",
        "source_record_id": "paper-resource-index:10.1000/order-check",
        "canonical_url": "https://example.org/paper-resource-index",
        "title": "Paper resource index",
        "kind": "paper",
        "identifiers": [{"namespace": "doi", "value": "10.1000/order-check"}],
        "links": [
            {
                "url": "https://example.org/model-checkpoint.bin",
                "relation": "model_artifact",
            }
        ],
        "models": [],
    }

    forward = build_entries([first, second, third, evidence])
    reverse = build_entries([evidence, third, second, first])

    assert forward.manifest() == reverse.manifest()
    assert len(forward.entries) == 1
    assert len(forward.entries[0].members) == 3


def test_entry_seed_does_not_assign_paper_checkpoint_to_multiple_models() -> None:
    paper_record = _seed(
        "paper-record",
        [{"namespace": "provider:model", "value": "example/model-a"}],
    )
    paper_record["models"].append(
        {
            "local_id": "model-b",
            "name": "Example model B",
            "identifiers": [{"namespace": "provider:model", "value": "example/model-b"}],
        }
    )
    paper_record["identifiers"] = [{"namespace": "doi", "value": "10.1000/paper"}]
    evidence_record = {
        "source": "biorxiv-jats-supplementary",
        "source_record_id": "paper-record:jats-model-resources",
        "canonical_url": "https://example.org/paper.xml",
        "title": "Paper supplementary resources",
        "kind": "paper",
        "identifiers": [{"namespace": "doi", "value": "10.1000/paper"}],
        "links": [
            {
                "url": "https://example.org/model-checkpoint.safetensors",
                "relation": "model_artifact",
            }
        ],
        "models": [],
    }

    result = build_entries([paper_record, evidence_record])

    assert len(result.entries) == 2
    assert all(
        resource.url != "https://example.org/model-checkpoint.safetensors"
        for entry in result.entries
        for resource in entry.resources
    )
