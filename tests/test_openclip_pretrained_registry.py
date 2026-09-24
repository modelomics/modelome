from __future__ import annotations

import json

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.openclip_pretrained_registry import (
    OpenCLIPPretrainedRegistrySourceAdapter,
    _parse_registry,
)

COMMIT = "a" * 40
COMMIT_URL = "https://api.github.com/repos/mlfoundations/open_clip/commits/main"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/mlfoundations/open_clip/{COMMIT}/"
    "src/open_clip/pretrained.py"
)

REGISTRY = '''
_RN50 = dict(
    openai=_pcfg(url="https://openaipublic.azureedge.net/clip/ViT-B-32.pt"),
    laion=_pcfg(url="https://github.com/mlfoundations/open_clip/releases/download/v1/rn50.pt"),
    hub_only=_pcfg(hf_hub="laion/CLIP-RN50/"),
)
_PRETRAINED = {"RN50": _RN50}
'''


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(url: str, body: bytes, status: int = 200) -> HttpResponse:
    return HttpResponse(status=status, headers={}, body=body, url=url)


def test_registry_parser_keeps_only_direct_non_openai_non_hub_entries() -> None:
    parsed = _parse_registry(REGISTRY)

    assert [(model, tag) for model, tag, _ in parsed] == [("RN50", "laion")]
    assert parsed[0][2]["url"].endswith("/rn50.pt")


def test_adapter_preserves_source_registry_and_exact_checkpoint_identity() -> None:
    client = QueuedClient(
        response(COMMIT_URL, json.dumps({"sha": COMMIT}).encode()),
        response(SOURCE_URL, REGISTRY.encode()),
    )
    adapter = OpenCLIPPretrainedRegistrySourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.upstream_count == 1
    assert page.next_state == {"completed_revision": COMMIT, "model_count": 1}
    assert client.calls == [COMMIT_URL, SOURCE_URL]
    record = page.records[0]
    assert record.source_record_id.endswith("RN50:laion")
    assert record.canonical_url == (
        "https://github.com/mlfoundations/open_clip/releases/download/v1/rn50.pt"
    )
    assert record.models[0].identifiers == (Identifier("openclip:pretrained-model", "RN50"),)
    assert record.releases[0].identifiers == (
        Identifier("openclip:pretrained-checkpoint", "RN50:laion"),
    )
    assert record.links[0].relation == "checkpoint"
    assert record.raw["registry"]["commit"] == COMMIT
    assert record.raw["entry"]["tag"] == "laion"


def test_unchanged_upstream_commit_skips_source_fetch() -> None:
    client = QueuedClient(response(COMMIT_URL, json.dumps({"sha": COMMIT}).encode()))
    page = OpenCLIPPretrainedRegistrySourceAdapter(client=client).fetch_page(
        {"completed_revision": COMMIT, "model_count": 17}
    )

    assert page.records == ()
    assert page.complete is True
    assert page.upstream_count == 17
    assert client.calls == [COMMIT_URL]
