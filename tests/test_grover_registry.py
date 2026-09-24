from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.grover_registry import GroverCheckpointRegistrySourceAdapter

_REVISION = "d" * 40
_BASE = "https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_base_ft_refine"
_LARGE = "https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/grover_large_ft_refine"
_BBBP_LINE = f"- BBBP: [BASE]({_BASE}/bbbp.tar.gz), [LARGE]({_LARGE}/bbbp.tar.gz)"
_README = f"""# GROVER

## The Reproducibility Issue

{_BBBP_LINE}
- FreeSolv: [BASE]({_BASE}/freesolv.tar.gz), [LARGE]({_LARGE}/freesolv.tar.gz)
- QM8 [BASE]({_BASE}/qm8.tar.gz), [LARGE]({_LARGE}/qm8.tar.gz)

## Next section
"""


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Mapping[str, object] | None = None,
            headers: Mapping[str, str] | None = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: str | Mapping[str, object]) -> HttpResponse:
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/registry")


def _adapter(client: _QueuedClient) -> GroverCheckpointRegistrySourceAdapter:
    return GroverCheckpointRegistrySourceAdapter(
        name="tencent-grover-finetuned-checkpoints",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_grover_registry_expands_each_explicit_dataset_row_into_base_and_large() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_README))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 6
    assert client.calls[1].endswith(f"/{_REVISION}/README.md")
    by_name = {record.title: record for record in page.records}
    assert set(by_name) == {
        "grover_base_ft_refine/bbbp", "grover_large_ft_refine/bbbp",
        "grover_base_ft_refine/freesolv", "grover_large_ft_refine/freesolv",
        "grover_base_ft_refine/qm8", "grover_large_ft_refine/qm8",
    }
    base = by_name["grover_base_ft_refine/bbbp"]
    assert base.models[0].identifiers == (
        Identifier("tencent-grover:finetuned-checkpoint", "grover_base_ft_refine/bbbp"),
    )
    assert base.releases[0].metadata["weight_url"] == (
        "https://ai.tencent.com/ailab/ml/ml-data/grover-models/finetune/"
        "grover_base_ft_refine/bbbp.tar.gz"
    )


def test_grover_registry_rejects_malformed_or_identity_mismatched_rows() -> None:
    bad_rows = (
        "- BBBP: [BASE](https://evil.test/model.tar.gz), "
        f"[LARGE]({_LARGE}/bbbp.tar.gz)",
        f"- BBBP: [BASE]({_LARGE}/bbbp.tar.gz), "
        f"[LARGE]({_LARGE}/bbbp.tar.gz)",
    )
    for bad_row in bad_rows:
        document = _README.replace(_BBBP_LINE, bad_row)
        client = _QueuedClient(_response({"sha": _REVISION}), _response(document))

        with pytest.raises(ValueError):
            _adapter(client).fetch_page({})
