from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.orb_models_pretrained_registry import (
    OrbModelsPretrainedRegistryAdapter,
    _parse_pretrained_urls,
)

REVISION = "e" * 40
COMMIT_URL = "https://api.github.com/repos/orbital-materials/orb-models/commits/main"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/orbital-materials/orb-models/{REVISION}/"
    "orb_models/forcefield/pretrained.py"
)
SOURCE = '''\
def orb_v3_direct_20_omat(weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt"):
    pass

def orbmol_v2(weights_path: str = "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orbmol-v2-teqabfhg-20260523.ckpt"):
    pass

orbmol_v1 = orb_v3_direct_20_omat
def removed(weights_path: str | None = None):
    pass

ORB_PRETRAINED_MODELS = {
    "orb-v3-direct-20-omat": orb_v3_direct_20_omat,
    "orbmol-v2": orbmol_v2,
    "orbmol-v1": orbmol_v1,
    "orb-v1": removed,
}
'''


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def test_extracts_literal_loader_names_and_exact_weight_urls() -> None:
    assert _parse_pretrained_urls(SOURCE, "test") == (
        (
            "orb-v3-direct-20-omat",
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
            "forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt",
        ),
        (
            "orbmol-v2",
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
            "forcefields/orbmol-v2-teqabfhg-20260523.ckpt",
        ),
        (
            "orbmol-v1",
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/"
            "forcefields/orb-v3/orb-v3-direct-20-omat-20250404.ckpt",
        ),
    )


@pytest.mark.parametrize(
    "text",
    [
        SOURCE.replace("orbitalmaterials-public-models.s3.us-west-1.amazonaws.com", "evil.test"),
        SOURCE.replace("/forcefields/", "/other/"),
        SOURCE.replace(".ckpt", ".zip"),
    ],
)
def test_rejects_non_first_party_or_unrecognized_urls(text: str) -> None:
    with pytest.raises(ValueError):
        _parse_pretrained_urls(text, "test")


def test_records_are_keyed_by_loader_and_do_not_claim_binary_validation() -> None:
    client = QueueClient(
        response(COMMIT_URL, json.dumps({"sha": REVISION}).encode()),
        response(SOURCE_URL, SOURCE.encode()),
    )

    page = OrbModelsPretrainedRegistryAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert client.calls == [COMMIT_URL, SOURCE_URL]
    first = page.records[0]
    assert first.kind is ArtifactKind.WEIGHTS
    assert first.raw["model_name"] == "orb-v3-direct-20-omat"
    assert first.raw["checkpoint_filename"] == "orb-v3-direct-20-omat-20250404.ckpt"
    assert first.models[0].identifiers == (
        Identifier("orb:checkpoint", "orb-v3-direct-20-omat"),
    )
    assert first.raw["binary_reachability_checked"] is False
    assert page.records[2].raw["model_name"] == "orbmol-v1"
    assert page.records[2].raw["checkpoint_url"] == first.raw["checkpoint_url"]


def test_live_orb_source_metadata_smoke() -> None:
    """Read the official loader source only; never fetch model checkpoint bytes."""
    try:
        page = OrbModelsPretrainedRegistryAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live Orb source metadata unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 22
    assert all(
        record.raw["checkpoint_url"].startswith(
            "https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/"
        )
        for record in page.records
    )
