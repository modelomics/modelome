from __future__ import annotations

import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter

_NOW = datetime(2026, 9, 24, tzinfo=UTC)
_BASE = "https://nixtla-public.s3.amazonaws.com/transfer/pretrained_models/"
_ARTIFACTS = {
    "nhits_m4_hourly": "Pretrained N-HiTS M4 Hourly",
    "nhits_m4_hourly_tiny": "Pretrained N-HiTS M4 Hourly (Tiny)",
    "nhits_m4_daily": "Pretrained N-HiTS M4 Daily",
    "nhits_m4_monthly": "Pretrained N-HiTS M4 Monthly",
    "nhits_m4_yearly": "Pretrained N-HiTS M4 Yearly",
    "nbeats_m4_hourly": "Pretrained N-BEATS M4 Hourly",
    "nbeats_m4_daily": "Pretrained N-BEATS M4 Daily",
    "nbeats_m4_weekly": "Pretrained N-BEATS M4 Weekly",
    "nbeats_m4_monthly": "Pretrained N-BEATS M4 Monthly",
    "nbeats_m4_yearly": "Pretrained N-BEATS M4 Yearly",
}
_HTML = (
    "<html><body>"
    + "".join(
        f'<a href="{_BASE}{artifact_id}.ckpt">{label}</a>\n'
        for artifact_id, label in _ARTIFACTS.items()
    )
    + '<a href="https://nixtla.io/">Demo</a></body></html>'
).encode()


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        return HttpResponse(200, {}, _HTML, url)


def test_official_checkpoint_inventory_extracts_exact_m4_weight_urls() -> None:
    config = tomllib.loads(
        (
            Path(__file__).parents[1]
            / "config/proposals/nixtla_transfer_learning_checkpoints.toml"
        ).read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = HtmlCatalogSourceAdapter(
        name=config["name"],
        url=config["url"],
        provider_namespace=config["provider_namespace"],
        artifact_kind=config["artifact_kind"],
        model_status=config["model_status"],
        allowed_origins=config["allowed_origins"],
        rules=config["rules"],
        max_response_bytes=config["max_response_bytes"],
        max_entries=config["max_entries"],
        client=client,
        clock=lambda: _NOW,
    )

    page = adapter.fetch_page({})

    assert page.complete and page.upstream_count == len(_ARTIFACTS)
    assert len(page.records) == 1
    record = page.records[0]
    models = {
        model.identifiers[0].value: model
        for model in record.models
    }
    assert set(models) == set(_ARTIFACTS)
    for artifact_id, label in _ARTIFACTS.items():
        model = models[artifact_id]
        assert model.name == label
        assert any(
            link.url == f"{_BASE}{artifact_id}.ckpt"
            and link.model_local_ids == (model.local_id,)
            for link in record.links
            if link.relation == "weights"
        )
    assert len(client.calls) == 1
