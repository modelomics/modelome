from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.deep_wetlands_checkpoint_registry import (
    DeepWetlandsCheckpointRegistrySourceAdapter,
    _parse_readme,
)

FILES = {
    "small-2018-2019": "1gRj98jWhvRSeLoAzcNzwM6Gi9I8K__0-",
    "small-2020-2022": "1N7ca5fKTGdazw7n8ALCYH6m3nsG2SUII",
    "large-2018-2019": "11CFnSUrKsjvTo4JqcFxKXP7RwCmwSUze",
    "large-2020-2022": "1fJeg6hPMORZoNkcUC-zh7XlaPL6Bj-FA",
}
README = "\n".join(
    (
        "# Deep Wetlands",
        "## Download Pre-trained Models",
        "### Small Models",
        "#### Years 2018-2019",
        f"https://drive.google.com/file/d/{FILES['small-2018-2019']}",
        "#### Years 2020-2022",
        f"https://drive.google.com/file/d/{FILES['small-2020-2022']}",
        "### Large Models",
        "#### Years 2018-2019",
        f"https://drive.google.com/file/d/{FILES['large-2018-2019']}",
        "#### Years 2020-2022",
        f"https://drive.google.com/file/d/{FILES['large-2020-2022']}",
        "## Tutorial",
    )
)


class Client:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.urls.append(url)
        return self.responses.pop(0)


def response(body: str, status: int = 200) -> HttpResponse:
    return HttpResponse(status, {"content-type": "text/plain"}, body.encode(), "https://fixture.test")


def test_adapter_emits_four_exact_google_drive_file_links() -> None:
    client = Client(response(README))
    page = DeepWetlandsCheckpointRegistrySourceAdapter(client=client).fetch_page({})
    expected = {
        f"https://drive.google.com/file/d/{file_id}/view"
        for file_id in FILES.values()
    }
    observed = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }

    assert page.authoritative_snapshot and page.complete and page.upstream_count == 4
    assert observed == expected
    assert client.urls == [
        "https://raw.githubusercontent.com/melqkiades/deep-wetlands/master/README.md"
    ]
    assert len({record.models[0].local_id for record in page.records}) == 4


def test_adapter_uses_readme_hash_for_freshness() -> None:
    client = Client(response(README), response(README))
    adapter = DeepWetlandsCheckpointRegistrySourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 4
    assert second.records == ()
    assert second.next_state == first.next_state


@pytest.mark.parametrize(
    "document",
    [
        README.replace(FILES["small-2018-2019"], "wrong-id"),
        README.replace("### Large Models", "### Medium Models"),
        README.replace(
            "## Tutorial",
            f"https://drive.google.com/file/d/{FILES['large-2020-2022']}\n## Tutorial",
        ),
        README + "\n## Download Pre-trained Models\n",
    ],
)
def test_parser_rejects_incomplete_or_ambiguous_inventory(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test")


def test_adapter_rejects_failed_readme_response() -> None:
    adapter = DeepWetlandsCheckpointRegistrySourceAdapter(client=Client(response(README, 404)))
    with pytest.raises(ValueError, match="HTTP 404"):
        adapter.fetch_page({})
