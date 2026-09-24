from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.apt_vla_checkpoint_registry import (
    APTVLACheckpointRegistryAdapter,
    _parse_inventory,
)

_SHA = "e" * 40
_FOLDERS = ("apt_vla", "apt_vla_ftlibero", "apt_vla_ftpp")
_README = "\n".join(
    [
        "# APT",
        "## Pretrained Checkpoints",
        "Stage | Config | Datasets | Hugging Face",
        "--- | --- | --- | ---",
        *[
            f"{stage} | `{config}` | `{dataset}` | "
            f"[checkpoint](https://huggingface.co/KechunXu1/apt_models/tree/main/{folder})"
            for stage, config, dataset, folder in (
                ("Pretrained VLA policy", "pretrain", "Droid + AgiBotWorld", _FOLDERS[0]),
                ("LIBERO fine-tuned", "finetune_libero", "LIBERO suites", _FOLDERS[1]),
                ("Pick-Place fine-tuned", "finetune_pp", "PickPlaceCan", _FOLDERS[2]),
            )
        ],
        "## Citation",
    ]
)


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        del params, headers
        self.calls.append(url)
        if url.endswith("/README.md"):
            body: Any = _README
        elif url.endswith("/api/models/KechunXu1/apt_models"):
            body = {"sha": _SHA}
        elif "/tree/" in url and "/b0efa7a0892cc5ad776c7fac1ca21c8500576207/apt_va" in url:
            body = [
                {
                    "type": "file",
                    "path": "apt_va/ckpt_best.pt",
                    "oid": "71d232cc580cc6e9cfc63c397d1d6242fb20aa5d",
                    "size": 676_848_250,
                },
                {
                    "type": "file",
                    "path": "apt_va/ckpt_latest.pt",
                    "oid": "oid-apt_va",
                    "size": 676_849_222,
                },
                {
                    "type": "file",
                    "path": "apt_va/202605111256.json",
                    "oid": "config-apt_va",
                    "size": 1_138,
                },
            ]
        elif "/tree/" in url:
            folder = url.rsplit("/", 1)[-1]
            body = [
                {
                    "type": "file",
                    "path": f"{folder}/ckpt_latest.pt",
                    "oid": f"oid-{folder}",
                    "size": 1024,
                },
                {
                    "type": "file",
                    "path": f"{folder}/config.json",
                    "oid": f"config-{folder}",
                    "size": 128,
                },
            ]
            if folder == "apt_va":
                body[0]["size"] = 676_849_222
                body[1] = {
                    "type": "file",
                    "path": "apt_va/config.json",
                    "oid": "config-apt_va",
                    "size": 1_138,
                }
        else:
            raise AssertionError(f"unexpected URL: {url}")
        encoded = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        return HttpResponse(200, {}, encoded, url)


def test_parser_keeps_exact_three_official_model_ids() -> None:
    entries = _parse_inventory(_README, source="test")
    assert [entry[0] for entry in entries] == sorted(_FOLDERS)
    assert {entry[1] for entry in entries} == {
        "Pretrained VLA policy",
        "LIBERO fine-tuned",
        "Pick-Place fine-tuned",
    }


def test_adapter_pins_each_exact_checkpoint_and_config_asset() -> None:
    client = _Client()
    adapter = APTVLACheckpointRegistryAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 5
    assert [record.source_record_id for record in page.records] == [
        *[f"apt:{folder}" for folder in sorted(_FOLDERS)],
        "apt:apt_va:latest",
        "apt:apt_va:best-archived",
    ]
    assert client.calls[1] == "https://huggingface.co/api/models/KechunXu1/apt_models"
    for record in page.records:
        folder = record.source_record_id.removeprefix("apt:").split(":", 1)[0]
        filename = record.releases[0].metadata["checkpoint_path"].rsplit("/", 1)[-1]
        revision = record.releases[0].revision
        weights_url = (
            f"https://huggingface.co/KechunXu1/apt_models/resolve/{revision}/{folder}/{filename}"
        )
        assert record.kind is ArtifactKind.WEIGHTS
        assert record.canonical_url == weights_url
        assert record.identifiers == (Identifier("apt:policy", folder),)
        assert (
            next(link for link in record.links if link.relation == "model_artifact").url
            == weights_url
        )
        assert next(
            link for link in record.links if link.relation == "model_configuration"
        ).url == (
            f"https://huggingface.co/KechunXu1/apt_models/resolve/{revision}/"
            f"{record.releases[0].metadata['config_path']}"
        )
        assert record.releases[0].metadata["checkpoint_oid"] == (
            "71d232cc580cc6e9cfc63c397d1d6242fb20aa5d"
            if filename == "ckpt_best.pt"
            else f"oid-{folder}"
        )


def test_adapter_fails_closed_when_checkpoint_component_is_missing() -> None:
    class MissingWeightsClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            response = super().get(url, params=params, headers=headers)
            if "/tree/" in url:
                body = [
                    {
                        "type": "file",
                        "path": f"{url.rsplit('/', 1)[-1]}/config.json",
                        "oid": "x",
                        "size": 1,
                    }
                ]
                return HttpResponse(200, {}, json.dumps(body).encode(), url)
            return response

    with pytest.raises(ValueError, match="missing checkpoint/config assets"):
        APTVLACheckpointRegistryAdapter(client=MissingWeightsClient()).fetch_page({})
