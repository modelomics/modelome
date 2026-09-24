from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.ftw_release_checkpoints import (
    FTWReleaseCheckpointSourceAdapter,
    _parse_readme,
)

FILES = (
    "2_Class_FULL_FTW_Pretrained.ckpt",
    "3_Class_FULL_FTW_Pretrained.ckpt",
    "2_Class_CCBY_FTW_Pretrained.ckpt",
    "3_Class_CCBY_FTW_Pretrained.ckpt",
)
PREFIX = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v1/"
README = "\n".join(
    (
        "# Fields of The World",
        "## Inference",
        "wget " + PREFIX + FILES[0],
        "wget " + PREFIX + FILES[1],
        "### CC-BY (or equivalent) trained models",
        "wget " + PREFIX + FILES[2],
        PREFIX + FILES[3],
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


def test_adapter_emits_four_exact_release_asset_links() -> None:
    client = Client(response(README))
    page = FTWReleaseCheckpointSourceAdapter(client=client).fetch_page({})
    expected = {PREFIX + filename for filename in FILES}
    observed = {
        link.url
        for record in page.records
        for link in record.links
        if link.relation == "weights"
    }

    assert page.authoritative_snapshot and page.complete and page.upstream_count == 4
    assert observed == expected
    assert client.urls == ["https://raw.githubusercontent.com/terramira/ftw-baselines/main/README.md"]
    assert len({record.models[0].local_id for record in page.records}) == 4


def test_adapter_uses_readme_hash_for_freshness() -> None:
    client = Client(response(README), response(README))
    adapter = FTWReleaseCheckpointSourceAdapter(client=client)
    first = adapter.fetch_page({})
    second = adapter.fetch_page(first.next_state)
    assert len(first.records) == 4
    assert second.records == ()
    assert second.next_state == first.next_state


@pytest.mark.parametrize(
    "document",
    [
        README.replace(FILES[0], "other.ckpt", 1),
        README.replace(PREFIX + FILES[0], "https://example.org/other.ckpt"),
        README + "\n" + PREFIX + FILES[1],
        README.replace("### CC-BY (or equivalent) trained models", "### CC-BY weights"),
    ],
)
def test_parser_rejects_incomplete_or_ambiguous_inventory(document: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(document, "test")


def test_adapter_rejects_failed_readme_response() -> None:
    adapter = FTWReleaseCheckpointSourceAdapter(client=Client(response(README, 404)))
    with pytest.raises(ValueError, match="HTTP 404"):
        adapter.fetch_page({})
