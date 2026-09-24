from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.allennlp_model_archives import AllenNLPModelArchiveSourceAdapter

_REVISION = "a" * 40
_REPOSITORY = "allenai/allennlp-models"
_CARDS = "allennlp_models/modelcards"


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
        if "/commits/" in url:
            body = json.dumps({"sha": _REVISION}).encode()
        elif "/contents/" in url:
            body = json.dumps(
                [
                    {"type": "file", "name": "rc-bidaf-elmo.json"},
                    {"type": "file", "name": "modelcard-template.json"},
                    {"type": "dir", "name": "ignored"},
                ]
            ).encode()
        elif url.endswith("/rc-bidaf-elmo.json"):
            body = json.dumps(
                {
                    "id": "rc-bidaf-elmo",
                    "display_name": "ELMo-BiDAF",
                    "model_details": {"short_description": "BiDAF with ELMo embeddings."},
                    "model_usage": {"archive_file": "bidaf-elmo.2021-02-11.tar.gz"},
                }
            ).encode()
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {"content-type": "application/json"}, body, url)


def test_allennlp_model_cards_produce_exact_archive_entries() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/allennlp_model_archives.toml").read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = AllenNLPModelArchiveSourceAdapter(
        name=config["name"],
        repository=config["repository"],
        branch=config["branch"],
        modelcards_path=config["modelcards_path"],
        provider_namespace=config["provider_namespace"],
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.title == "ELMo-BiDAF"
    assert record.canonical_url == (
        "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo.2021-02-11.tar.gz"
    )
    assert record.models[0].identifiers[0].value == "rc-bidaf-elmo"
    assert record.releases[0].identifiers[0].value == "bidaf-elmo.2021-02-11.tar.gz"
    assert any(link.relation == "model_card" for link in record.links)
    assert not any(url.endswith(".tar.gz") for url in client.calls)
