from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.groundingdino_checkpoints import (
    GroundingDINOCheckpointSourceAdapter,
    _parse_checkpoint_rows,
)

_REVISION = "c" * 40
_T_GH = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha/groundingdino_swint_ogc.pth"
)
_T_HF = (
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
    "groundingdino_swint_ogc.pth"
)
_T_CONFIG = (
    "https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)
_B_GH = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
    "v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
)
_B_HF = (
    "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
    "groundingdino_swinb_cogcoor.pth"
)
_B_CONFIG = (
    "https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/config/"
    "GroundingDINO_SwinB_cfg.py"
)
_README = (
    b"# Grounding DINO\n<table><thead><tr>"
    b"<th></th><th>name</th><th>backbone</th><th>Data</th>"
    b"<th>box AP on COCO</th><th>Checkpoint</th><th>Config</th>"
    b"</tr></thead><tbody>"
    + (
        '<tr><th>1</th><td>GroundingDINO-T</td><td>Swin-T</td><td>O365,GoldG,Cap4M</td>'
        '<td>48.4 (zero-shot) / 57.2 (fine-tune)</td><td>'
        f'<a href="{_T_GH}">GitHub link</a> | <a href="{_T_HF}">HF link</a>'
        f'</td><td><a href="{_T_CONFIG}">link</a></td></tr>'
    ).encode()
    + (
        '<tr><th>2</th><td>GroundingDINO-B</td><td>Swin-B</td>'
        '<td>COCO,O365,GoldG,Cap4M,OpenImage,ODinW-35,RefCOCO</td><td>56.7</td><td>'
        f'<a href="{_B_GH}">GitHub link</a> | <a href="{_B_HF}">HF link</a>'
        f'</td><td><a href="{_B_CONFIG}">link</a></td></tr>'
    ).encode()
    + b"</tbody></table>"
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


def test_groundingdino_catalog_pairs_each_variant_with_exact_release_and_mirror() -> None:
    client = _Client()
    page = GroundingDINOCheckpointSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert len(client.calls) == 2
    by_name = {record.models[0].name: record for record in page.records}
    assert set(by_name) == {"GroundingDINO-T", "GroundingDINO-B"}

    tiny = by_name["GroundingDINO-T"]
    assert tiny.models[0].identifiers[0].value == "GroundingDINO-T"
    assert tiny.releases[0].version == "v0.1.0-alpha"
    assert tiny.releases[0].metadata["github_release_url"].endswith(
        "/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )
    assert tiny.releases[0].metadata["huggingface_mirror_url"].endswith(
        "/groundingdino_swint_ogc.pth"
    )
    assert tiny.releases[0].metadata["config_url"].endswith("GroundingDINO_SwinT_OGC.py")
    assert tiny.releases[0].metadata["reported_coco_box_ap"] == (
        "48.4 (zero-shot) / 57.2 (fine-tune)"
    )
    assert [link.relation for link in tiny.links] == [
        "model_card",
        "weights",
        "weights_mirror",
        "model_config",
    ]

    base = by_name["GroundingDINO-B"]
    assert base.releases[0].version == "v0.1.0-alpha2"
    assert base.raw["checkpoint_filename"] == "groundingdino_swinb_cogcoor.pth"


def test_groundingdino_checkpoint_parser_fails_closed_on_unmapped_model_row() -> None:
    readme = _README.replace(
        b"GroundingDINO-B</td>", b"GroundingDINO-L</td>"
    )
    with pytest.raises(ValueError, match="unknown checkpoint row"):
        _parse_checkpoint_rows(readme.decode(), maximum=10)


def test_groundingdino_checkpoint_parser_enforces_limit_and_duplicate_rows() -> None:
    with pytest.raises(ValueError, match="entry limit"):
        _parse_checkpoint_rows(_README.decode(), maximum=1)

    duplicate = _README.replace(
        b"</tbody>",
        b"<tr><td></td><td>GroundingDINO-T</td><td>T</td><td>data</td><td>48.4</td>"
        b"<td><a href=\"https://github.com/IDEA-Research/GroundingDINO/releases/download/"
        b"v0.1.0-alpha/groundingdino_swint_ogc.pth\">GitHub</a> | "
        b"<a href=\"https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/"
        b"groundingdino_swint_ogc.pth\">HF</a></td><td><a href=\"https://github.com/"
        b"IDEA-Research/GroundingDINO/blob/main/groundingdino/config/"
        b"GroundingDINO_SwinT_OGC.py\">config</a></td></tr></tbody>",
    )
    with pytest.raises(ValueError, match="duplicate checkpoint row"):
        _parse_checkpoint_rows(duplicate.decode(), maximum=10)


def test_groundingdino_checkpoint_source_reuses_unchanged_revision() -> None:
    page = GroundingDINOCheckpointSourceAdapter(client=_Client()).fetch_page(
        {"completed_revision": _REVISION, "model_count": 2}
    )
    assert page.complete and not page.records
    assert page.upstream_count == 2
    assert not page.authoritative_snapshot
