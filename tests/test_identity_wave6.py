from __future__ import annotations

import pytest

from modelome.entries import build_entries


def _seed(record_id: str, value: str) -> dict[str, object]:
    return {
        "source": record_id,
        "source_record_id": record_id,
        "canonical_url": f"https://{record_id}.example/model",
        "title": record_id,
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": "Documented model",
                "identifiers": [{"namespace": "doi", "value": value}],
            }
        ],
    }


@pytest.mark.parametrize(
    "doi_value",
    [
        "doi:10.1234/Model.Reference",
        "https://doi.org/10.1234/Model.Reference",
        "http://dx.doi.org/10.1234/Model.Reference",
    ],
)
def test_doi_namespace_wrappers_reconcile_to_the_same_model_identity(
    doi_value: str,
) -> None:
    result = build_entries(
        [
            _seed("source-a", "10.1234/model.reference"),
            _seed("source-b", doi_value),
        ]
    )

    assert len(result.entries) == 1
    assert len(result.entries[0].members) == 2
    assert [identifier.key for identifier in result.entries[0].identifiers] == [
        "doi:10.1234/model.reference"
    ]
