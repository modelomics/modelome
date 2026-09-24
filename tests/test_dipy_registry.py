from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.dipy_registry import DipyPretrainedRegistrySourceAdapter, _parse_fetchers

SHA = "a" * 40


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(value: Any, url: str, *, status: int = 200) -> HttpResponse:
    body = value if isinstance(value, bytes) else json.dumps(value).encode()
    return HttpResponse(status, {}, body, url)


def test_dipy_fetcher_parser_keeps_only_literal_model_weight_figshare_entries() -> None:
    code = """
fetch_synb0_weights = _make_fetcher(
    "fetch_synb0_weights", dipy_home / "synb0", "https://ndownloader.figshare.com/files/",
    ["36379914", "36379917"], [Path("synb0_1.h5"), Path("synb0_2.h5")],
    md5_list=["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"],
    doc=("Download Synb0 model weights for Schilling et al. 2019"))
fetch_synb0_test = _make_fetcher(
    "fetch_synb0_test", dipy_home / "synb0", "https://ndownloader.figshare.com/files/",
    ["36379911"], [Path("test_input.npz")], doc="Download Synb0 test data")
fetch_other_weights = _make_fetcher(
    "fetch_other_weights", dipy_home / "other", "https://example.org/",
    ["12345678"], [Path("weights.pt")], doc="Download other model weights")
"""
    entries = _parse_fetchers(code, "dipy", "dipy/data/fetcher.py", 10)
    assert len(entries) == 1
    fetcher, doc, assets, md5s, locator = entries[0]
    assert fetcher == "fetch_synb0_weights"
    assert "Synb0" in doc
    assert assets == (("36379914", "synb0_1.h5"), ("36379917", "synb0_2.h5"))
    assert md5s == ("a" * 32, "b" * 32)
    assert locator == "dipy/data/fetcher.py:L2"


def test_dipy_adapter_emits_exact_artifact_ids_without_downloading_weights() -> None:
    source_url = f"https://raw.githubusercontent.com/dipy/dipy/{SHA}/dipy/data/fetcher.py"
    source_text = """fetch_deepn4_torch_weights = _make_fetcher(
"fetch_deepn4_torch_weights", dipy_home / "deepn4", "https://ndownloader.figshare.com/files/",
["52285805"], [Path("deepn4_torch_weights")],
md5_list=["97c5a5f8356a3d0eeca1c6bb7949c8b8"],
doc="Download DeepN4 model weights for Kanakaraj et. al 2024")\n"""
    client = QueuedClient(
        response({"sha": SHA}, "https://api.github.com/repos/dipy/dipy/commits/master"),
        response(source_text.encode(), source_url),
    )
    source = DipyPretrainedRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 23, tzinfo=UTC)
    )
    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == "dipy-pretrained-models:fetch_deepn4_torch_weights"
    assert record.models[0].identifiers == (Identifier("dipy:model", "fetch_deepn4_torch_weights"),)
    assert record.links[0].url == "https://ndownloader.figshare.com/files/52285805"
    assert record.links[0].relation == "weights"
    assert record.releases[0].metadata["artifacts"] == [
        {
            "file_id": "52285805",
            "filename": "deepn4_torch_weights",
            "url": "https://ndownloader.figshare.com/files/52285805",
            "md5": "97c5a5f8356a3d0eeca1c6bb7949c8b8",
        }
    ]
    assert len(client.calls) == 2
    assert client.calls[1] == source_url


def test_dipy_parser_rejects_unbounded_asset_forms() -> None:
    code = """fetch_model_weights = _make_fetcher(
"fetch_model_weights", None, "https://ndownloader.figshare.com/files/",
[make_id()], [Path("weights.pt")], doc="Download model weights")\n"""
    assert _parse_fetchers(code, "dipy", "fetcher.py", 10) == ()
    with pytest.raises(ValueError, match="eligible model fetchers"):
        _parse_fetchers(
            """fetch_one_weights = _make_fetcher("fetch_one_weights", None,
"https://ndownloader.figshare.com/files/", ["12345678"], ["weights.pt"],
doc="Download model weights")\n""",
            "dipy",
            "fetcher.py",
            0,
        )
