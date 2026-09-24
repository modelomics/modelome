from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter

NOW = datetime(2026, 9, 21, 20, 0, tzinfo=UTC)
REVISION = "c" * 40


class QueuedClient:
    def __init__(self, *payloads: dict[str, Any]) -> None:
        self.payloads = list(payloads)

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        payload = self.payloads.pop(0)
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_monai_bundle_keys_preserve_semver_build_metadata_and_exact_identity() -> None:
    key = "swin_unetr_3d_v1.2.0+cuda.12"
    archive = "https://example.test/bundles/swin_unetr_3d-v1.2.0%2Bcuda.12.zip"
    client = QueuedClient(
        {"sha": REVISION},
        {key: {"source": archive, "checksum": "A" * 40}},
    )
    source = MonaiModelZooSourceAdapter(client=client, clock=lambda: NOW)

    page = source.fetch_page({})

    assert page.upstream_count == 1
    record = page.records[0]
    assert record.source_record_id == f"bundle:{key}"
    assert record.identifiers == (Identifier("monai:bundle-record", key),)
    assert record.models[0].name == "swin_unetr_3d"
    assert record.models[0].identifiers == (
        Identifier("monai:model", "swin_unetr_3d"),
    )
    assert record.releases[0].version == "1.2.0+cuda.12"
    assert record.releases[0].identifiers[0] == Identifier("monai:bundle", key)
    assert record.releases[0].metadata["archive_sha1"] == "a" * 40
    assert record.releases[0].metadata["archive_url"] == archive
