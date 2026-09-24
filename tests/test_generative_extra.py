from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.generative_extra import (
    CompVisLatentDiffusionDownloadsSourceAdapter,
    _parse_download_script,
)

_REVISION = "a" * 40
_SCRIPT = """\
#!/bin/bash
wget -O models/ldm/celeba256/celeba-256.zip https://ommer-lab.com/files/latent-diffusion/celeba.zip
wget -O models/ldm/text2img256/model.zip https://ommer-lab.com/files/latent-diffusion/text2img.zip
cd models/ldm/celeba256
unzip -o celeba-256.zip
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


def test_compvis_download_script_emits_exact_model_bundle_references() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_SCRIPT))
    adapter = CompVisLatentDiffusionDownloadsSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.records[0].kind is ArtifactKind.MODEL_CARD
    assert page.records[0].source_record_id == "checkpoint:celeba256"
    assert page.records[0].models[0].identifiers == (
        Identifier("compvis:latent-diffusion", "celeba256"),
    )
    assert page.records[0].releases[0].metadata["weight_url"].endswith("/celeba.zip")
    assert any(
        link.relation == "weights"
        and link.url == "https://ommer-lab.com/files/latent-diffusion/celeba.zip"
        for link in page.records[0].links
    )
    assert f"/{_REVISION}/scripts/download_models.sh" in client.calls[1]


def test_disabled_compvis_proposal_matches_adapter_constructor() -> None:
    proposal_path = Path(__file__).parents[1] / "config/proposals/generative_extra.toml"
    with proposal_path.open("rb") as handle:
        proposal = tomllib.load(handle)["source"][0]

    assert proposal["enabled"] is False
    assert proposal["adapter"] == "compvis_latent_diffusion_downloads"
    adapter = CompVisLatentDiffusionDownloadsSourceAdapter(
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
    ("row", "message"),
    [
        (
            "wget -O models/ldm/model/model.zip https://huggingface.co/model.zip",
            "non-first-party",
        ),
        (
            "wget -O models/ldm/../model/model.zip https://ommer-lab.com/files/latent-diffusion/model.zip",
            "invalid model output path",
        ),
        (
            "wget -O models/ldm/model/model.txt https://ommer-lab.com/files/latent-diffusion/model.txt",
            "non-first-party",
        ),
        (
            "wget --post-data=x -O models/ldm/model/model.zip https://ommer-lab.com/files/latent-diffusion/model.zip",
            "unsupported wget row",
        ),
    ],
)
def test_compvis_download_script_rejects_ambiguous_or_foreign_downloads(
    row: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_download_script(row, source="fixture", path="download_models.sh", maximum=10)
