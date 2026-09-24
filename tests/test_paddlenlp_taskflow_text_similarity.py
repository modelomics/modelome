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
from modelome.sources.paddlenlp_taskflow_text_similarity import (
    PaddleNlpTaskflowTextSimilaritySourceAdapter,
    _parse_resource_map,
)

_REVISION = "f" * 40
_MODELS = {
    "simbert-base-chinese": (
        "https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/simbert-base-chinese/model_state.pdparams",
        "27d9ef240c2e8e736bdfefea52af2542",
    ),
    "rocketqa-zh-dureader-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-zh-dureader-cross-encoder/model_state.pdparams",
        "88bc3e1a64992a1bdfe4044ecba13bc7",
    ),
    "rocketqa-base-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-base-cross-encoder/model_state.pdparams",
        "6d845a492a2695e62f2be79f8017be92",
    ),
    "rocketqa-medium-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-medium-cross-encoder/model_state.pdparams",
        "4b929f4fc11a1df8f59fdf2784e23fa7",
    ),
    "rocketqa-mini-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-mini-cross-encoder/model_state.pdparams",
        "c411111df990132fb88c070d8b8cf3f7",
    ),
    "rocketqa-micro-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-micro-cross-encoder/model_state.pdparams",
        "3d643ff7d6029c8ceab5653680167dc0",
    ),
    "rocketqa-nano-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqa-nano-cross-encoder/model_state.pdparams",
        "4c1d36e5e94f5af09f665fc7ad0be140",
    ),
    "rocketqav2-en-marco-cross-encoder": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/rocketqav2-en-marco-cross-encoder/model_state.pdparams",
        "a5afc77b6a63fc32a1beca3010f40f32",
    ),
    "ernie-search-large-cross-encoder-marco-en": (
        "https://paddlenlp.bj.bcebos.com/taskflow/text_similarity/ernie-search-large-cross-encoder-marco-en/model_state.pdparams",
        "fdf29f7de0f7fe570740d343c96165e5",
    ),
}
_MAP_ITEMS = ",\n".join(
    f'        "{name}": {{"model_state": ["{url}", "{md5}"]}}'
    for name, (url, md5) in _MODELS.items()
)
_INTERNAL_FIXTURE = (
    '"__internal_testing__/tiny-random-bert": '
    '{"model_state": ["https://bj.bcebos.com/paddlenlp/taskflow/text_similarity/test/'
    'model_state.pdparams", "0123456789abcdef0123456789abcdef"]}'
)
_SOURCE = f"""\
class TextSimilarityTask:
    resource_files_urls = {{
{_MAP_ITEMS},
        {_INTERNAL_FIXTURE},
    }}
"""


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
        self.calls.append(url)
        body = json.dumps({"sha": _REVISION}).encode() if "/commits/" in url else _SOURCE.encode()
        return HttpResponse(200, {}, body, url)


def _adapter(client: _Client, **kwargs: Any) -> PaddleNlpTaskflowTextSimilaritySourceAdapter:
    return PaddleNlpTaskflowTextSimilaritySourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
        **kwargs,
    )


def test_parser_and_adapter_emit_all_public_exact_taskflow_checkpoint_rows() -> None:
    parsed = _parse_resource_map(_SOURCE, max_entries=20)
    assert parsed == tuple((name, *details) for name, details in sorted(_MODELS.items()))

    page = _adapter(_Client()).fetch_page({})
    assert page.complete is True and page.authoritative_snapshot is True
    assert page.upstream_count == len(_MODELS)
    row = next(
        record
        for record in page.records
        if record.identifiers[0].value == "rocketqa-base-cross-encoder"
    )
    assert row.kind is ArtifactKind.WEIGHTS
    assert row.identifiers == (
        Identifier("paddlenlp:taskflow-text-similarity", "rocketqa-base-cross-encoder"),
    )
    assert row.canonical_url == _MODELS["rocketqa-base-cross-encoder"][0]
    assert row.releases[0].metadata["md5"] == _MODELS["rocketqa-base-cross-encoder"][1]
    assert row.raw["revision"] == _REVISION


def test_parser_rejects_nonliteral_maps_and_untrusted_artifact_urls() -> None:
    with pytest.raises(ValueError, match="not literal"):
        _parse_resource_map(
            "class TextSimilarityTask:\n resource_files_urls = load_map()",
            max_entries=20,
        )
    unsafe = _SOURCE.replace(
        _MODELS["rocketqa-base-cross-encoder"][0], "https://example.test/model.pdparams"
    )
    parsed = _parse_resource_map(unsafe, max_entries=20)
    assert all(name != "rocketqa-base-cross-encoder" for name, *_ in parsed)


def test_disabled_proposal_matches_adapter() -> None:
    path = Path(__file__).parents[1] / "config/proposals/paddlenlp_taskflow_text_similarity.toml"
    source = tomllib.loads(path.read_text())["source"][0]
    assert source["enabled"] is False
    adapter = PaddleNlpTaskflowTextSimilaritySourceAdapter(
        name=source["name"],
        repository=source["repository"],
        branch=source["branch"],
        source_path=source["source_path"],
        max_response_bytes=source["max_response_bytes"],
        max_entries=source["max_entries"],
    )
    assert adapter.name == source["name"]
    assert adapter.source_path == source["source_path"]
