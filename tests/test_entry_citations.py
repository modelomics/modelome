from __future__ import annotations

import json
from copy import deepcopy

import pytest

from modelome.entries import (
    build_entries,
    plan_entry_seed,
    read_entry_seeds,
    write_entry_bundle,
)
from modelome.entry_seed_export import export_current_entry_seeds
from modelome.models import ArtifactKind, Identifier, Link, ModelHint, SourceRecord
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.storage import Database


def seed(name, *, identifiers=(), links=()):
    return {
        "source": "fixture",
        "source_record_id": name,
        "canonical_url": f"https://papers.example/{name}",
        "title": name,
        "kind": "paper",
        "identifiers": [
            {"namespace": namespace, "value": value} for namespace, value in identifiers
        ],
        "models": [{"local_id": name, "name": name}],
        "links": list(links),
    }


def cite(url, relation="cites", **kwargs):
    return {"url": url, "relation": relation, "locator": "references:1", **kwargs}


@pytest.mark.parametrize(
    ("identifier", "url"),
    [
        (("doi", "10.1000/EXAMPLE"), "https://dx.doi.org/10.1000/example"),
        (("arxiv", "2401.12345v2"), "https://arxiv.org/pdf/2401.12345v3.pdf"),
        (("openalex", "W123"), "https://openalex.org/W123"),
    ],
)
def test_citations_resolve_exact_artifact_ids_without_merging_entries(identifier, url):
    result = build_entries(
        [
            seed("A", links=[cite(url)]),
            seed("B", identifiers=[identifier]),
        ]
    )
    a, b = result.entries
    assert len(result.entries) == 2
    assert len(a.citations) == 1
    assert a.citations[0].target_entry_id == b.id
    assert a.citations[0].evidence[0].locator == "references:1"
    assert not a.citations[0].evidence[0].crawl
    assert b.citations == ()


def test_closed_graph_ignores_external_artifact_only_ambiguous_and_self_targets(tmp_path):
    a = seed(
        "A",
        links=[
            cite("https://papers.example/B"),
            cite("https://papers.example/B"),
            cite("https://outside.example/missing"),
            cite("https://papers.example/ordinary"),
            cite("https://papers.example/A"),
            cite("https://doi.org/10.1000/shared"),
            {"url": "https://docs.example/guide", "relation": "documented_by"},
            {"url": "https://papers.example/B", "relation": "references"},
        ],
    )
    ordinary = seed("ordinary")
    ordinary["models"] = []
    seeds = [
        a,
        seed("B"),
        ordinary,
        seed("C", identifiers=[("doi", "10.1000/shared")]),
        seed("D", identifiers=[("doi", "10.1000/shared")]),
    ]
    result = build_entries(seeds)
    a, b, _, _ = result.entries
    assert [edge.target_entry_id for edge in a.citations] == [b.id]
    assert len(a.citations[0].evidence) == 1
    assert {resource.relation for resource in a.resources} == {
        "seed",
        "documented_by",
        "references",
    }
    assert all(
        edge.target_entry_id in {entry.id for entry in result.entries}
        for entry in result.entries
        for edge in entry.citations
    )
    assert result.manifest() == build_entries(reversed(seeds)).manifest()
    assert result.manifest()["citation_count"] == 1

    write_entry_bundle(result, tmp_path / "bundle")
    rows = [
        json.loads(line) for line in (tmp_path / "bundle/entries.jsonl").read_text().splitlines()
    ]
    assert rows[0]["citations"][0]["target_entry_id"] == b.id
    assert "https://outside.example/missing" not in json.dumps(rows)


def test_citations_do_not_supply_target_identity_and_keep_distinct_evidence():
    links = [cite("https://doi.org/10.1000/missing")]
    assert all(
        not entry.citations
        for entry in build_entries(
            [
                seed("A", links=links),
                seed("B", links=links),
            ]
        ).entries
    )

    a = seed("A", links=[cite("https://doi.org/10.1000/paper")])
    a["links"].append({**a["links"][0], "locator": "references:2"})
    b = seed("B", links=[{"url": "https://doi.org/10.1000/paper", "relation": "paper"}])
    first, second = build_entries([a, b]).entries
    assert first.citations[0].target_entry_id == second.id
    assert len(first.citations[0].evidence) == 2


def test_citations_respect_model_scope_and_plan_only_conditional_links():
    a = seed("A", links=[cite("https://papers.example/B", model_local_ids=["A"])])
    a["models"].append({"local_id": "sibling", "name": "Sibling"})
    result = build_entries([a, seed("B")])
    by_name = {entry.canonical_name: entry for entry in result.entries}
    assert by_name["A"].citations[0].target_entry_id == by_name["B"].id
    assert not by_name["Sibling"].citations
    actions = [
        action for action in plan_entry_seed(a)["actions"] if action.get("relation") == "cites"
    ]
    assert [action["action"] for action in actions] == ["attach_citation_if_present"]


@pytest.mark.parametrize(
    ("predicate", "direction", "reverse"),
    [
        ("cites", "outgoing", False),
        ("cites", "incoming", True),
        ("cited_by", "outgoing", True),
        ("cited_by", "incoming", False),
        ("is-cited-by", "outgoing", True),
    ],
)
def test_citation_direction_from_relation_exports(predicate, direction, reverse):
    link = cite("https://papers.example/B", predicate)
    link["resolved_artifact"] = {
        "id": "artifact-b",
        "revision_id": "revision-b",
        "kind": "paper",
        "source": "fixture",
        "source_record_id": "B",
        "canonical_url": link["url"],
    }
    link["relation_evidence"] = {
        "id": "evidence",
        "direction": direction,
        "evidence_type": "url",
        "confidence": 1.0,
    }
    a, b = build_entries([seed("A", links=[link]), seed("B")]).entries
    source, target = (b, a) if reverse else (a, b)
    assert source.citations[0].target_entry_id == target.id
    assert target.citations == ()


def test_merged_entry_self_citation_is_omitted():
    a = seed("A", links=[cite("https://papers.example/B")])
    b = seed("B")
    for item in (a, b):
        item["models"][0]["identifiers"] = [{"namespace": "model", "value": "shared"}]
    result = build_entries([a, b])
    assert len(result.entries) == 1
    assert result.entries[0].citations == ()


def test_openalex_declared_references_are_non_crawling_citations():
    record = OpenAlexSourceAdapter()._record(
        {
            "id": "https://openalex.org/W1",
            "title": "A",
            "referenced_works": [
                "https://openalex.org/W2",
                None,
                "invalid",
                {},
                "https://outside.example/W3",
                "https://openalex.org/A1",
            ],
            "related_works": ["https://openalex.org/W4"],
        }
    )
    assert [link for link in record.links if link.relation == "cites"] == [
        Link("https://openalex.org/W2", "cites", "$.referenced_works[0]", crawl=False),
    ]


@pytest.mark.parametrize("link_current_resources", [False, True])
def test_store_export_build_keeps_only_internal_citations(tmp_path, link_current_resources):
    store = Database(tmp_path / "store")
    store.initialize()
    records = []
    for name in ("A", "B", "ordinary"):
        records.append(
            SourceRecord(
                source_record_id=name,
                kind=ArtifactKind.PAPER,
                canonical_url=f"https://papers.example/{name}",
                title=name,
                raw={},
                identifiers=(Identifier("doi", f"10.1000/{name.lower()}"),),
                models=() if name == "ordinary" else (ModelHint(local_id=name, name=name),),
                links=tuple(
                    Link(url, "cites", crawl=False)
                    for url in (
                        "https://papers.example/B",
                        "https://doi.org/10.1000/ordinary",
                        "https://outside.example/missing",
                    )
                )
                if name == "A"
                else (),
            )
        )
    store.ingest_page("fixture", records, extractor="fixture")
    output = tmp_path / "seeds.jsonl"
    export_current_entry_seeds(
        tmp_path / "store",
        output,
        link_current_resources=link_current_resources,
    )
    seeds = read_entry_seeds(output)
    original = deepcopy(seeds)
    result = build_entries(seeds)
    a, b = result.entries
    assert [edge.target_entry_id for edge in a.citations] == [b.id]
    assert b.citations == ()
    assert seeds == original
