from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind
from modelome.sources.falcon_vla_checkpoint_zoo import FalconVLACheckpointZooAdapter

_COMMIT = "a" * 40
_REVISION = "b" * 40
_README = """## 🤗 Model Zoo <a name="model-zoo"></a>
<table>
<tr><td>FALCON-FC-CALVIN-ABC</td><td><a href="https://huggingface.co/FALCON-VLA/FALCON-series/tree/main/falcon-esm-fc-calvin-abc/ckpts">model</a></td></tr>
<tr><td>FALCON-FC-OXE</td><td><a href="https://huggingface.co/FALCON-VLA/FALCON-series/tree/main/falcon-oxe-magic-soup-pretrain/ckpts">model</a></td></tr>
</table>
## 🏋️ Training
"""


class _Client:
    def __init__(self) -> None:
        self.extra_weights = False

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        del params, headers
        if url.endswith("/commits/main"):
            payload: Any = {"sha": _COMMIT}
        elif url.endswith(f"/{_COMMIT}/README.md"):
            payload = _README
        elif url.endswith("/api/models/FALCON-VLA/FALCON-series"):
            payload = {"sha": _REVISION}
        elif "/tree/" in url:
            payload = [
                {
                    "type": "file",
                    "path": "falcon-esm-fc-calvin-abc/ckpts/falcon_esm_fc_calvin_abc.pt",
                    "oid": "1" * 40,
                    "size": 2048,
                },
                {
                    "type": "file",
                    "path": (
                        "falcon-oxe-magic-soup-pretrain/ckpts/"
                        "falcon_oxe_magic_soup_pretrain.pt"
                    ),
                    "oid": "2" * 40,
                    "size": 4096,
                },
            ]
            if self.extra_weights:
                payload.append(
                    {
                        "type": "file",
                        "path": "falcon-esm-fc-calvin-abc/ckpts/extra.pt",
                        "oid": "3" * 40,
                        "size": 1,
                    }
                )
        else:
            raise AssertionError(f"unexpected URL: {url}")
        body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        return HttpResponse(200, {}, body, url)


def test_indexes_model_zoo_rows_with_pinned_weight_files() -> None:
    adapter = FalconVLACheckpointZooAdapter(
        client=_Client(), clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )
    page = adapter.fetch_page({})
    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert [record.title for record in page.records] == [
        "FALCON-FC-CALVIN-ABC",
        "FALCON-FC-OXE",
    ]
    record = page.records[0]
    expected = (
        "https://huggingface.co/FALCON-VLA/FALCON-series/resolve/"
        f"{_REVISION}/falcon-esm-fc-calvin-abc/ckpts/falcon_esm_fc_calvin_abc.pt"
    )
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url == expected
    assert record.releases[0].metadata["weight_oid"] == "1" * 40
    assert record.releases[0].metadata["weight_size_bytes"] == 2048
    assert record.releases[0].metadata["model_zoo_document_revision"] == _COMMIT
    assert FalconVLACheckpointZooAdapter(client=_Client()).checkpoint_signature == (
        adapter.checkpoint_signature
    )


def test_rejects_ambiguous_checkpoint_path() -> None:
    client = _Client()
    client.extra_weights = True
    with pytest.raises(ValueError, match="expected one checkpoint .pt file"):
        FalconVLACheckpointZooAdapter(client=client).fetch_page({})
