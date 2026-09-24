from __future__ import annotations

from modelome.entries import build_entries


def _seed(record_id: str, doi: str) -> dict[str, object]:
    return {
        "source": "independent-catalog",
        "source_record_id": record_id,
        "canonical_url": f"https://catalog.example/{record_id}",
        "title": record_id,
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": "Example model",
                "identifiers": [{"namespace": "doi", "value": doi}],
            }
        ],
    }


def test_doi_case_variants_merge_independently_sourced_model_records() -> None:
    result = build_entries(
        [
            _seed("catalog-a", "10.1234/Model.Card"),
            _seed("catalog-b", "10.1234/model.card"),
        ]
    )

    assert len(result.entries) == 1
    entry = result.entries[0]
    assert len(entry.members) == 2
    assert [identifier.key for identifier in entry.identifiers] == [
        "doi:10.1234/model.card"
    ]


def test_explicit_model_mirror_identifier_merges_cross_source_cards() -> None:
    opencsg = {
        "source": "opencsg",
        "source_record_id": "org/model",
        "canonical_url": "https://opencsg.example/org/model",
        "title": "OpenCSG model",
        "kind": "model_card",
        "models": [
            {
                "local_id": "opencsg-model",
                "name": "OpenCSG model",
                "identifiers": [
                    {"namespace": "opencsg:model", "value": "org/model"}
                ],
            }
        ],
        "model_relations": [
            {
                "subject_local_id": "opencsg-model",
                "predicate": "mirrors",
                "target": {
                    "local_id": "org/model#ms_path",
                    "name": "org/model",
                    "identifiers": [
                        {"namespace": "modelscope:model", "value": "org/model"}
                    ],
                },
            }
        ],
    }
    modelscope = {
        "source": "modelscope",
        "source_record_id": "org/model",
        "canonical_url": "https://modelscope.cn/models/org/model",
        "title": "ModelScope model",
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": "ModelScope model",
                "identifiers": [
                    {"namespace": "modelscope:model", "value": "org/model"}
                ],
            }
        ],
    }

    result = build_entries([opencsg, modelscope])

    assert len(result.entries) == 1
    assert {(member.source, member.source_record_id) for member in result.entries[0].members} == {
        ("opencsg", "org/model"),
        ("modelscope", "org/model"),
    }
