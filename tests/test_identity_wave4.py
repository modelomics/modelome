from __future__ import annotations

import pytest

from modelome.entries import build_entries


@pytest.mark.parametrize("relation", ["weights", "model_weights"])
def test_shared_non_checkpoint_artifacts_do_not_merge_distinct_models(
    relation: str,
) -> None:
    seeds = []
    for source, name in (("catalog-a", "Base model"), ("catalog-b", "Fine-tune")):
        seeds.append(
            {
                "source": source,
                "source_record_id": name.casefold().replace(" ", "-"),
                "canonical_url": f"https://{source}.example/{name.casefold().replace(' ', '-')}",
                "title": name,
                "kind": "model_card",
                "models": [
                    {
                        "local_id": "model",
                        "name": name,
                        "identifiers": [
                            {"namespace": f"{source}:model", "value": name}
                        ],
                    }
                ],
                "links": [
                    {
                        "url": "https://artifacts.example/shared/tokenizer.json",
                        "relation": relation,
                    }
                ],
            }
        )

    result = build_entries(seeds)

    assert len(result.entries) == 2
    assert {entry.canonical_name for entry in result.entries} == {
        "Base model",
        "Fine-tune",
    }
