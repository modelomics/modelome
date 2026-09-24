from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.torch_hub_extra import TorchHubListingSourceAdapter

_REVISION = "d" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/pytorch-hub")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        for path, text in files.items():
            package.writestr(f"hub-{_REVISION}/{path}", text)
    return output.getvalue()


def test_listing_records_are_exact_entrypoint_ref_identities_and_skip_torchvision() -> None:
    page = """\
layout | hub_detail
title | Example Vision
summary | A curated model page.
github-id | lab/example-model

    model = torch.hub.load('lab/example-model:v2.1', 'small_net')
    # Documented alternative entrypoint.
    # model = torch.hub.load('lab/example-model:v2.1', 'large_net')
    torch.hub.load('lab/example-model', 'small_net')
    torch.hub.load('pytorch/vision:v0.20.0', 'resnet50')
"""
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(_archive({"lab_example.md": page, "README.md": "not a model page"})),
    )
    adapter = TorchHubListingSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    result = adapter.fetch_page({})

    assert result.complete and result.authoritative_snapshot
    assert result.upstream_count == 3
    assert result.next_state["explicit_ref_count"] == 2
    by_identity = {record.identifiers[0].value: record for record in result.records}
    assert set(by_identity) == {
        "lab/example-model:v2.1#small_net",
        "lab/example-model:v2.1#large_net",
        "lab/example-model#small_net",
    }
    tagged = by_identity["lab/example-model:v2.1#small_net"]
    assert tagged.models[0].name == "small_net"
    assert tagged.models[0].status is ModelStatus.DOCUMENTED
    assert tagged.models[0].identifiers == (
        Identifier("pytorch:hub-load", "lab/example-model:v2.1#small_net"),
    )
    assert tagged.releases[0].version == "v2.1"
    assert tagged.releases[0].identifiers == (
        Identifier("pytorch:hub-binding", "lab/example-model:v2.1#small_net"),
    )
    assert tagged.raw["load_declarations"][0]["entrypoint"] == "small_net"
    assert "pytorch/vision" not in " ".join(by_identity)


def test_listing_skip_when_pinned_revision_is_unchanged() -> None:
    adapter = TorchHubListingSourceAdapter(
        client=_QueuedClient(_response({"sha": _REVISION})),
    )

    result = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 3})

    assert result.records == ()
    assert result.upstream_count == 3
