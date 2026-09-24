from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.yolox_model_zoo import YOLOXModelZooSourceAdapter, _parse_model_zoo

_REVISION = "d" * 40
_CURRENT = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
_LEGACY = "https://github.com/Megvii-BaseDetection/storage/releases/download/0.0.1"
_README = (
    b"# YOLOX\n#### Standard Models.\n"
    b"|Model |size |mAP |weights |\n| ------ |:---: |:---: |:----: |\n"
    + (
        f"|[YOLOX-s](./exps/default/yolox_s.py) |640 |40.5 | "
        f"[github]({_CURRENT}/yolox_s.pth) |\n"
    ).encode()
    + b"<details>\n<summary>Legacy models</summary>\n"
    b"|Model |size |mAP |weights |\n| ------ |:---: |:---: |:----: |\n"
    + (
        f"|[YOLOX-s](./exps/default/yolox_s.py) |640 |39.6 |"
        f"[onedrive](https://example.org/download?id=private)/"
        f"[github]({_LEGACY}/yolox_s.pth) |\n"
    ).encode()
    + b"</details>\n#### Light Models.\n"
    b"|Model |size |mAP |weights |\n| ------ |:---: |:---: |:----: |\n"
    + (
        f"|[YOLOX-Nano](./exps/default/yolox_nano.py) |416 |25.8 | "
        f"[github]({_CURRENT}/yolox_nano.pth) |\n"
    ).encode()
)


class _Client:
    def __init__(self, readme: bytes = _README) -> None:
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
        body = f'{{"sha":"{_REVISION}"}}'.encode() if "/commits/" in url else self.readme
        return HttpResponse(200, {"content-type": "text/plain"}, body, url)


def test_yolox_source_preserves_exact_current_and_legacy_checkpoint_versions() -> None:
    client = _Client()
    page = YOLOXModelZooSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert len(client.calls) == 2
    records = {record.raw["weight_url"]: record for record in page.records}
    assert set(records) == {
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_s.pth",
        "https://github.com/Megvii-BaseDetection/storage/releases/download/"
        "0.0.1/yolox_s.pth",
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_nano.pth",
    }

    standard = records[
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_s.pth"
    ]
    legacy = records[
        "https://github.com/Megvii-BaseDetection/storage/releases/download/"
        "0.0.1/yolox_s.pth"
    ]
    assert standard.models[0].name == legacy.models[0].name == "YOLOX-s"
    assert standard.models[0].identifiers == legacy.models[0].identifiers
    assert standard.releases[0].version == "0.1.1rc0"
    assert legacy.releases[0].version == "0.0.1"
    assert standard.releases[0].identifiers != legacy.releases[0].identifiers
    assert legacy.releases[0].metadata["table_section"] == "legacy-standard models."

    nano = records[
        "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        "0.1.1rc0/yolox_nano.pth"
    ]
    assert nano.models[0].name == "YOLOX-Nano"
    assert nano.releases[0].metadata["config_url"].endswith("exps/default/yolox_nano.py")


def test_yolox_parser_ignores_non_release_rows_but_rejects_unknown_release_rows() -> None:
    no_release = _README.replace(
        b"https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
        b"0.1.1rc0/yolox_s.pth",
        b"https://example.org/yolox_s.pth",
        1,
    )
    assert len(_parse_model_zoo(no_release.decode(), maximum=10)) == 2

    unknown = _README.replace(b"YOLOX-s](./", b"YOLOX-z](./", 1)
    with pytest.raises(ValueError, match="unexpected model/checkpoint mapping"):
        _parse_model_zoo(unknown.decode(), maximum=10)


def test_yolox_parser_enforces_limit_and_rejects_duplicate_release_urls() -> None:
    with pytest.raises(ValueError, match="entry limit"):
        _parse_model_zoo(_README.decode(), maximum=2)

    duplicate = _README.replace(
        b"</details>",
        b"|[YOLOX-s](./exps/default/yolox_s.py) |640 |39.6 |[github]("
        b"https://github.com/Megvii-BaseDetection/storage/releases/download/"
        b"0.0.1/yolox_s.pth) |\n</details>",
    )
    with pytest.raises(ValueError, match="duplicate checkpoint URL"):
        _parse_model_zoo(duplicate.decode(), maximum=10)


def test_yolox_source_reuses_unchanged_revision_checkpoint() -> None:
    page = YOLOXModelZooSourceAdapter(client=_Client()).fetch_page(
        {"completed_revision": _REVISION, "checkpoint_count": 3}
    )
    assert page.complete and not page.records
    assert page.upstream_count == 3
    assert not page.authoritative_snapshot
