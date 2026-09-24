from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.ocp_model_registry import OCPModelRegistrySourceAdapter

DOCS = "https://facebookresearch.github.io/fairchem/models-1/"
BASE = "https://dl.fbaipublicfiles.com/opencatalystproject/models/"


class QueueClient:
    def __init__(self, response: HttpResponse) -> None:
        self.response = response
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.response


def test_reads_model_names_and_exact_first_party_checkpoint_links_from_docs() -> None:
    html = (
        "<table><tr><th>Model Name</th><th>Download</th></tr>"
        "<tr><td>GemNet-OC-S2EF-OC20-All+MD</td>"
        f'<td><a href="{BASE}2022_03/s2ef/gemnet_oc_all_md.pt">checkpoint</a></td>'
        '<td><a href="https://github.com/facebookresearch/fairchem/blob/main/config.yml">'
        "config</a></td></tr>"
        "<tr><td>EquiformerV2-31M-S2EF-OC20-All+MD</td>"
        f'<td><a href="{BASE}2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt">checkpoint</a></td>'
        "</tr><tr><td>UMA</td><td>"
        '<a href="https://huggingface.co/facebook/UMA">checkpoint</a></td></tr>'
        '<tr><td>Other</td><td><a href="https://example.org/other.pt">checkpoint</a>'
        "</td></tr></table>"
    )
    client = QueueClient(HttpResponse(200, {"etag": "abc"}, html.encode(), DOCS))
    source = OCPModelRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 2
    records = {record.raw["checkpoint_handle"]: record for record in page.records}
    assert set(records) == {"GemNet-OC-S2EF-OC20-All+MD", "EquiformerV2-31M-S2EF-OC20-All+MD"}
    record = records["EquiformerV2-31M-S2EF-OC20-All+MD"]
    assert record.raw["weight_url"] == f"{BASE}2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt"
    assert record.releases[0].metadata["weight_url"] == record.raw["weight_url"]
    assert record.links[1].url == record.raw["weight_url"]
    assert all(not link.crawl for link in record.links)
    assert client.calls == [DOCS]


def test_unchanged_docs_snapshot_does_not_reemit_records() -> None:
    html = (
        "<table><tr><td>GemNet-dT-S2EFS-OC22</td><td>"
        f'<a href="{BASE}x/model.pt">checkpoint</a></td></tr></table>'
    )
    response = HttpResponse(200, {}, html.encode(), DOCS)
    digest = __import__("hashlib").sha256(html.encode()).hexdigest()
    page = OCPModelRegistrySourceAdapter(client=QueueClient(response)).fetch_page(
        {"completed_sha256": digest, "model_count": 1}
    )
    assert page.records == ()
    assert page.upstream_count == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://dl.fbaipublicfiles.com/opencatalystproject/model.pt",
        "https://example.org/model.pt",
        "https://dl.fbaipublicfiles.com.evil.example/model.pt",
        "https://dl.fbaipublicfiles.com/opencatalystproject/model.pt?download=1",
    ],
)
def test_rejects_untrusted_or_ambiguous_artifact_urls(url: str) -> None:
    html = f'<table><tr><td>Model-X</td><td><a href="{url}">checkpoint</a></td></tr></table>'
    client = QueueClient(HttpResponse(200, {}, html.encode(), DOCS))
    with pytest.raises(ValueError, match="no first-party checkpoint rows"):
        OCPModelRegistrySourceAdapter(client=client).fetch_page({})
