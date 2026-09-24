from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.ddlp_video_checkpoints import (
    DDLPVideoCheckpointSourceAdapter,
    _parse_model_zoo,
)

_SHA = "a" * 40
_URLS = {
    ("DLPv2", "OBJ3D"): "https://mega.nz/file/wdMUxaQJ#75L0jqofo4Gj1EyjIJN2zBzt6XFN9s2jgU83XjjAqXQ",
    ("DLPv2", "Traffic"): "https://mega.nz/file/MNljnLCZ#3d6U6zP_FCDOBpxPWOJdOOnmjiq8Cyl9ND2u8qjXlsE",
    ("DDLP", "OBJ3D"): "https://mega.nz/file/QcsRSQRD#-jCBXhIIKs__6Zys8eBLo8f75WQfDhP0LcPLuRgy5p8",
    ("DDLP", "Traffic"): "https://mega.nz/file/9clHVLTI#6_HQXzHWLYal36HnNnynqzQwnmzVY4FIXbnVmQ5eXa8",
    ("DDLP", "PHYRE"): "https://mega.nz/file/UBcl2LgQ#cA2mS1mfSoAbNa8VmjeSkrRHDlXlupjVNNXnhO8zoPs",
    ("DDLP", "CLEVRER"): "https://mega.nz/file/oUUjXY4B#LLxCs1h3h3v4pXbLjl2Tplxwoq41DWaLJxyRHTgOWnw",
    ("DiffuseDDLP", "OBJ3D"): "https://mega.nz/file/kJkgAa6a#owpT2vTvPWLEG5KRh-JqZXt3yL1Y1wSH7R3lyZT9uRs",
    ("DiffuseDDLP", "Traffic"): "https://mega.nz/file/4J0DHBBA#EwETJSsL_Utzn8L7GBL9E63MNolKKqEgBd40HCBnoac",
}
_README = """## Model Zoo - Pretrained Models

| Model Type | Dataset | Link |
|---|---|---|
""" + "\n".join(
    f"| {model} | {dataset} (128x128) | [MEGA.nz]({url}) |"
    for (model, dataset), url in _URLS.items()
) + """

## Interactive Graphical User Interface (GUI)
"""


class _Client:
    def __init__(self, readme: str = _README) -> None:
        self.readme = readme
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = json.dumps({"sha": _SHA}).encode() if "/commits/" in url else self.readme.encode()
        return HttpResponse(200, {}, body, url)


def test_parser_extracts_all_eight_exact_first_party_bundle_urls() -> None:
    rows = _parse_model_zoo(_README, maximum=8)
    assert len(rows) == 8
    assert {(row["model"], row["dataset"]): row["url"] for row in rows} == _URLS


def test_parser_rejects_changed_ids_duplicates_missing_rows_and_overflow() -> None:
    changed_asset = _README.replace("wdMUxaQJ", "evilID0")
    with pytest.raises(ValueError, match="unexpected DDLP checkpoint"):
        _parse_model_zoo(changed_asset, maximum=8)

    first_row = next(
        line for line in _README.splitlines() if line.startswith("| DLPv2 | OBJ3D")
    )
    with pytest.raises(ValueError, match="duplicate DDLP"):
        changed = _README.replace("## Interactive", f"{first_row}\n## Interactive")
        _parse_model_zoo(changed, maximum=8)

    with pytest.raises(ValueError, match="missing expected"):
        _parse_model_zoo(_README.replace(first_row + "\n", ""), maximum=8)
    with pytest.raises(ValueError, match="count exceeds"):
        _parse_model_zoo(_README, maximum=7)


def test_adapter_emits_metadata_only_bundle_records_and_pins_source_revision() -> None:
    client = _Client()
    adapter = DDLPVideoCheckpointSourceAdapter(client=client)
    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot and page.upstream_count == 8
    assert len(client.calls) == 2
    assert client.calls[0] == adapter.commit_url
    assert client.calls[1] == adapter.raw_url(_SHA)
    by_type_dataset = {(r.raw["model_type"], r.raw["dataset"]): r for r in page.records}
    assert set(by_type_dataset) == set(_URLS)
    for key, record in by_type_dataset.items():
        url = _URLS[key]
        assert record.canonical_url == url.split("#", 1)[0]
        assert record.raw["checkpoint_url"] == url
        assert record.raw["bundle_id"] == url.split("/file/", 1)[1].split("#", 1)[0]
        assert record.raw["checkpoint_bytes_fetched"] is False
        assert record.links[0].url == url and record.links[0].crawl is False
        assert record.models[0].identifiers[0].value == f"{key[0]}/{key[1]}/128x128"
        assert record.releases[0].metadata["checkpoint_url"] == url


def test_adapter_fails_closed_when_expected_entries_are_absent() -> None:
    readme = _README.replace("## Model Zoo - Pretrained Models", "## Changed heading")
    with pytest.raises(ValueError, match="expected all eight"):
        DDLPVideoCheckpointSourceAdapter(client=_Client(readme)).fetch_page({})
