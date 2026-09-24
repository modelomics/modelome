from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.wenet_model_zoo import WenetPretrainedModelSourceAdapter

_REVISION = "c" * 40


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
    raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, raw, "https://fixtures.test/wenet")


def test_wenet_model_table_preserves_checkpoint_and_runtime_artifacts() -> None:
    document = "\n".join(
        (
            "# Pretrained Models",
            "## Model Types",
            "Checkpoint Model uses .pt; Runtime Model uses .zip.",
            "## Model List",
            "| Datasets | Languages | Checkpoint Model | Runtime Model | Contributor |",
            "| --- | --- | --- | --- | --- |",
            "| [aishell](../examples/aishell/s0/README.md) | CN | "
            "[Conformer](https://wenet.org.cn/downloads?models=wenet&version="
            "aishell_u2pp_conformer_exp.tar.gz) | [Conformer](https://wenet.org.cn/downloads?"
            "models=wenet&version=aishell_u2pp_conformer_libtorch.tar.gz) | Example Org |",
            "| paraformer | CN&EN | "
            "[Model](https://wenet.org.cn/downloads?models=wenet&version="
            "paraformer.tar.gz) | NA | Alibaba |",
        )
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(document))
    adapter = WenetPretrainedModelSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["artifact_count"] == 3
    assert len(client.calls) == 2
    aishell, paraformer = page.records
    assert aishell.identifiers == (
        Identifier("wenet:pretrained-model", "aishell/Conformer"),
    )
    assert [release.metadata["artifact_type"] for release in aishell.releases] == [
        "checkpoint",
        "runtime",
    ]
    artifact_relations = [
        link.relation
        for link in aishell.links
        if link.url.startswith("https://wenet.org.cn/")
    ]
    assert artifact_relations == ["weights", "inference_artifact"]
    assert paraformer.models[0].name == "paraformer"
    assert paraformer.releases[0].metadata["archive_filename"] == "paraformer.tar.gz"


def test_wenet_model_table_rejects_unexpected_artifact_hosts() -> None:
    document = "\n".join(
        (
            "## Model List",
            "| Datasets | Languages | Checkpoint Model | Runtime Model | Contributor |",
            "| --- | --- | --- | --- | --- |",
            "| aishell | CN | [Conformer](https://example.org/a.tar.gz) | NA | A |",
        )
    )
    adapter = WenetPretrainedModelSourceAdapter(
        client=_QueuedClient(_response({"sha": _REVISION}), _response(document))
    )

    with pytest.raises(ValueError, match="invalid WeNet URL"):
        adapter.fetch_page({})


@pytest.mark.parametrize(
    "document_path",
    ("README.md", "docs/other.md", "../pretrained_models.md"),
)
def test_wenet_adapter_is_bound_to_its_model_index(document_path: str) -> None:
    with pytest.raises(ValueError, match="document_path"):
        WenetPretrainedModelSourceAdapter(document_path=document_path)


def test_wenet_proposal_is_disabled_and_exactly_scoped() -> None:
    import tomllib

    proposal = tomllib.loads(Path("config/proposals/wenet_models.toml").read_text())
    source = proposal["source"][0]
    assert source["enabled"] is False
    assert source["adapter"] == "wenet_pretrained_models"
    assert source["document_path"] == "docs/pretrained_models.md"
