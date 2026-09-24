from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.deepchem_checkpoint import DeepChemMol2VecCheckpointSourceAdapter

REVISION = "b" * 40
ARCHIVE = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz"
)
SOURCE = (
    "https://raw.githubusercontent.com/deepchem/deepchem/" + REVISION + "/"
    "deepchem/feat/molecule_featurizers/mol2vec_fingerprint.py"
)


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def test_emits_exact_deepchem_default_mol2vec_archive_identity() -> None:
    api = "https://api.github.com/repos/deepchem/deepchem/commits/master"
    source_text = f'DEFAULT_PRETRAINED_MODEL_URL = "{ARCHIVE}"\n'
    client = QueueClient(
        HttpResponse(200, {}, json.dumps({"sha": REVISION}).encode(), api),
        HttpResponse(200, {}, source_text.encode(), SOURCE),
    )
    source = DeepChemMol2VecCheckpointSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "checkpoint:mol2vec-model-300dim"
    assert record.models[0].identifiers == (
        Identifier("deepchem:pretrained-checkpoint", "mol2vec-model-300dim"),
    )
    assert record.raw["weight_url"] == ARCHIVE
    assert record.releases[0].metadata["archive_filename"] == "mol2vec_model_300dim.tar.gz"
    assert record.releases[0].metadata["weight_url"] == ARCHIVE
    assert record.links[1].url == ARCHIVE
    assert client.calls == [api, SOURCE]


def test_skips_source_fetch_when_upstream_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/deepchem/deepchem/commits/master"
    client = QueueClient(HttpResponse(200, {}, json.dumps({"sha": REVISION}).encode(), api))
    page = DeepChemMol2VecCheckpointSourceAdapter(client=client).fetch_page(
        {"completed_revision": REVISION, "model_count": 1}
    )
    assert page.records == ()
    assert page.upstream_count == 1
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "assignment",
    [
        'DEFAULT_PRETRAINED_MODEL_URL = build_url("mol2vec")',
        'DEFAULT_PRETRAINED_MODEL_URL = "https://example.org/weights.tar.gz"',
        'DEFAULT_PRETRAINED_MODEL_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/other.tar.gz"',
    ],
)
def test_rejects_dynamic_or_noncanonical_checkpoint_assignments(assignment: str) -> None:
    api = "https://api.github.com/repos/deepchem/deepchem/commits/master"
    client = QueueClient(
        HttpResponse(200, {}, json.dumps({"sha": REVISION}).encode(), api),
        HttpResponse(200, {}, (assignment + "\n").encode(), SOURCE),
    )
    with pytest.raises(ValueError):
        DeepChemMol2VecCheckpointSourceAdapter(client=client).fetch_page({})
