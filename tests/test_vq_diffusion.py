from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.vq_diffusion import (
    MicrosoftVqDiffusionCheckpointManifestSourceAdapter,
    _parse_manifest,
)

_REVISION = "b" * 40
_SCRIPT = """\
wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_aa
wget https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ab
cat coco_pretrained_* > coco_pretrained.pth
wget https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_vqvae.pth
"""


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, Any]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/source")


def test_vq_diffusion_preserves_all_exact_urls_for_each_checkpoint() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SCRIPT))
    adapter = MicrosoftVqDiffusionCheckpointManifestSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    sharded, standalone = page.records
    assert sharded.models[0].identifiers == (
        Identifier("microsoft:vq-diffusion", "coco_pretrained"),
    )
    assert [link.url for link in sharded.links if link.relation == "weights"] == [
        "https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_aa",
        "https://github.com/tzco/storage/releases/download/vqdiffusion/coco_pretrained_ab",
    ]
    assert sharded.releases[0].metadata["weight_asset_count"] == 2
    assert standalone.releases[0].metadata["weight_urls"] == [
        "https://github.com/tzco/storage/releases/download/vqdiffusion/ithq_vqvae.pth",
    ]
    assert f"/{_REVISION}/vqdiffusion_download_checkpoints.sh" in client.calls[1]


def test_vq_diffusion_disabled_proposal_constructs_adapter() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/vq_diffusion.toml"
    import tomllib

    with proposal_path.open("rb") as handle:
        proposal = tomllib.load(handle)["source"][0]

    assert proposal["enabled"] is False
    assert proposal["adapter"] == "microsoft_vq_diffusion_checkpoint_manifest"
    adapter = MicrosoftVqDiffusionCheckpointManifestSourceAdapter(
        name=proposal["name"],
        repository=proposal["repository"],
        branch=proposal["branch"],
        source_path=proposal["source_path"],
        provider_namespace=proposal["provider_namespace"],
        max_response_bytes=proposal["max_response_bytes"],
        max_entries=proposal["max_entries"],
    )
    assert adapter.name == proposal["name"]


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            "wget https://example.com/weights/model.pth",
            "outside the approved path",
        ),
        (
            "wget https://github.com/tzco/storage/releases/download/vqdiffusion/chunk_aa",
            "has no output mapping",
        ),
        (
            "wget https://github.com/tzco/storage/releases/download/vqdiffusion/part_aa\n"
            "cat absent_* > model.pth",
            "has no literal downloads",
        ),
    ],
)
def test_vq_diffusion_parser_rejects_unsupported_or_unmapped_assets(
    document: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_manifest(document, source="fixture", path="download.sh", maximum=10)
