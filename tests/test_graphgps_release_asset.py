from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.graphgps_release_asset import GraphGPSReleaseAssetSourceAdapter

_REVISION = "b" * 40
_README = """# GraphGPS

## Inference and submission files for OGB-LSC leaderboard
You can download our pretrained GPS-deep (151 MB).

```bash
wget https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip
unzip pcqm4m-GPS+RWSE.deep.zip -d pretrained/
```

## Benchmarking GPS on 11 datasets
Other section.
"""


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
    body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/README.md")


def _adapter(client: _QueuedClient) -> GraphGPSReleaseAssetSourceAdapter:
    return GraphGPSReleaseAssetSourceAdapter(
        name="graphgps-ogblsc-pretrained-asset",
        repository="rampasek/GraphGPS",
        branch="main",
        provider_namespace="graphgps:checkpoint",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )


def test_graphgps_release_asset_emits_documented_gps_deep_archive() -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(_README))

    page = _adapter(client).fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 1
    assert client.calls[1].endswith(f"/{_REVISION}/README.md")
    record = page.records[0]
    assert record.title == "GPS-deep"
    assert record.models[0].identifiers == (Identifier("graphgps:checkpoint", "GPS-deep"),)
    assert record.releases[0].metadata["weight_url"] == (
        "https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip"
    )


@pytest.mark.parametrize(
    "document",
    [
        _README.replace(
            "## Inference and submission files for OGB-LSC leaderboard",
            "## Other",
        ),
        _README.replace("You can download our pretrained GPS-deep (151 MB).\n", ""),
        _README.replace(
            "wget https://www.dropbox.com/s/aomimvak4gb6et3/pcqm4m-GPS%2BRWSE.deep.zip",
            "wget https://example.test/one.zip\nwget https://example.test/two.zip",
        ),
    ],
)
def test_graphgps_release_asset_rejects_missing_or_ambiguous_inventory(
    document: str,
) -> None:
    client = _QueuedClient(_response({"sha": _REVISION}), _response(document))

    with pytest.raises(ValueError):
        _adapter(client).fetch_page({})
