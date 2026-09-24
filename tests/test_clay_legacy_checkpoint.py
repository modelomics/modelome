from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.clay_legacy_checkpoint import ClayLegacyCheckpointSourceAdapter

_DOC_URL = "https://clay-foundation.github.io/model/clay-v0/model_embeddings.html"
_S3_URI = "s3://clay-model-ckpt/v0/clay-small-70MT-1100T-10E.ckpt"


class _Client:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.calls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(self.status, {}, self.body, url)


def test_projects_only_the_explicit_legacy_checkpoint_from_official_docs() -> None:
    body = (
        "<html><code>aws s3 cp "
        + _S3_URI
        + " checkpoints/</code><code>aws s3 cp s3://other-bucket/other.ckpt x</code></html>"
    ).encode()
    client = _Client(body)
    adapter = ClayLegacyCheckpointSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete and page.upstream_count == 1
    record = page.records[0]
    assert record.canonical_url == _DOC_URL
    assert record.links[0].url == _DOC_URL
    assert record.links[0].relation == "documentation"
    assert all(link.url.startswith("https://") for link in record.links)
    assert record.raw["s3_uri"] == _S3_URI
    assert record.kind.value == "weights"
    assert record.identifiers == (
        Identifier("clay:model", "clay-v0-small-70mt-1100t-10e"),
    )
    assert record.releases[0].revision == "v0"
    assert record.releases[0].identifiers == (Identifier("clay:checkpoint", _S3_URI),)
    assert record.releases[0].metadata == {"checkpoint_uri": _S3_URI}
    assert record.raw["documentation_sha256"] == hashlib.sha256(body).hexdigest()
    assert client.calls == [_DOC_URL]

    seed = source_record_to_entry_seed(record, source="clay-legacy-checkpoint")
    entries = build_entries([seed])
    assert len(entries.entries) == 1
    assert entries.entries[0].resources[0].url == _DOC_URL
    assert json.loads(entries.entries[0].releases[0].metadata_json)["checkpoint_uri"] == _S3_URI


def test_unchanged_document_does_not_reemit_checkpoint() -> None:
    client = _Client(f"aws s3 cp {_S3_URI} checkpoints/".encode())
    adapter = ClayLegacyCheckpointSourceAdapter(client=client)
    first = adapter.fetch_page({})

    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.complete and second.upstream_count == 1
    assert len(client.calls) == 2


@pytest.mark.parametrize("body", [b"no checkpoint here", b"aws s3 cp s3://wrong/v0/model.ckpt x"])
def test_rejects_docs_without_exact_checkpoint_command(body: bytes) -> None:
    with pytest.raises(ValueError, match="command was not found"):
        ClayLegacyCheckpointSourceAdapter(client=_Client(body)).fetch_page({})


def test_rejects_non_success_and_oversized_document() -> None:
    with pytest.raises(ValueError, match="HTTP 503"):
        ClayLegacyCheckpointSourceAdapter(client=_Client(b"", status=503)).fetch_page({})
    with pytest.raises(ValueError, match="byte limit"):
        ClayLegacyCheckpointSourceAdapter(
            max_response_bytes=1, client=_Client(b"aws s3 cp " + _S3_URI.encode())
        ).fetch_page({})


def test_disabled_proposal_matches_adapter_configuration() -> None:
    path = Path(__file__).parents[1] / "config/proposals/clay_legacy_checkpoint.toml"
    source: dict[str, Any] = tomllib.loads(path.read_text())["source"][0]

    assert source["enabled"] is False
    adapter = ClayLegacyCheckpointSourceAdapter(
        name=source["name"], url=source["url"], max_response_bytes=source["max_response_bytes"]
    )
    assert adapter.name == source["name"]
    assert adapter.url == source["url"]
    assert adapter.max_response_bytes == source["max_response_bytes"]
