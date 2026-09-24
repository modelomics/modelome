from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.cgschnet_pretrained_bundle import (
    CGSchNetPretrainedBundleSourceAdapter,
)

REVISION = "f" * 40
BUNDLE_URL = "https://dx.doi.org/10.14279/depositonce-14978"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/atomistic-machine-learning/cG-SchNet/"
    f"{REVISION}/published_data/README.md"
)
README = f"""\
## Pretrained models

A zip-file containing two pretrained cG-SchNet models can be found under
[DOI 10.14279/depositonce-14978]({BUNDLE_URL}). The archive consists of two folders,
where comp_relenergy hosts the model conditioned on composition and relative
atomic energy. The other model, gap_relenergy, was conditioned on the HOMO-LUMO
gap and relative atomic energy.
"""


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: bytes, url: str = "https://api.github.com") -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def adapter(client: QueueClient) -> CGSchNetPretrainedBundleSourceAdapter:
    return CGSchNetPretrainedBundleSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )


def test_indexes_two_named_models_against_shared_doi_only() -> None:
    api = "https://api.github.com/repos/atomistic-machine-learning/cG-SchNet/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(README.encode(), SOURCE_URL),
    )

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 2
    assert client.calls == [api, SOURCE_URL]
    records = {record.models[0].name: record for record in page.records}
    assert set(records) == {
        "cG-SchNet comp_relenergy pretrained model",
        "cG-SchNet gap_relenergy pretrained model",
    }
    record = records["cG-SchNet comp_relenergy pretrained model"]
    assert record.models[0].identifiers == (
        Identifier("cgschnet:checkpoint", "comp_relenergy"),
    )
    assert record.canonical_url == BUNDLE_URL
    assert record.links[0].url == BUNDLE_URL
    assert record.links[0].relation == "bundle_reference"
    assert record.raw["per_model_artifact_url"] is None
    assert record.releases[0].metadata["binary_reachability_checked"] is False
    assert record.releases[0].metadata["per_model_artifact_url"] is None


def test_skips_readme_fetch_when_revision_is_unchanged() -> None:
    api = "https://api.github.com/repos/atomistic-machine-learning/cG-SchNet/commits/main"
    client = QueueClient(response(json.dumps({"sha": REVISION}).encode(), api))

    page = adapter(client).fetch_page({"completed_revision": REVISION, "model_count": 2})

    assert page.records == ()
    assert page.upstream_count == 2
    assert client.calls == [api]


@pytest.mark.parametrize(
    "document",
    [
        README.replace("gap_relenergy", "other_model"),
        README.replace(BUNDLE_URL, "https://example.org/archive.zip"),
        README.replace("two pretrained cG-SchNet models", "one pretrained cG-SchNet model"),
    ],
)
def test_rejects_missing_or_changed_bundle_evidence(document: str) -> None:
    api = "https://api.github.com/repos/atomistic-machine-learning/cG-SchNet/commits/main"
    client = QueueClient(
        response(json.dumps({"sha": REVISION}).encode(), api),
        response(document.encode(), SOURCE_URL),
    )

    with pytest.raises(ValueError):
        adapter(client).fetch_page({})
