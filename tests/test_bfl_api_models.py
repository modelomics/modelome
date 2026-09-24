from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.bfl_api_models import BFLAPIModelsSourceAdapter, _parse_endpoints

_REVISION = "b" * 40
_GUIDE = b"""# BFL API Endpoints

### FLUX.2 [klein] 4B
```
POST /v1/flux-2-klein-4b
```

### FLUX.2 [klein] 9B
```
POST /v1/flux-2-klein-9b
```

### FLUX.2 [max]
```
POST /v1/flux-2-max
```

### FLUX.2 [pro]
```
POST /v1/flux-2-pro
```

### FLUX.2 [flex]
```
POST /v1/flux-2-flex
```

### FLUX1.1 [pro]
```
POST /v1/flux-pro-1.1
```

### FLUX.1 Kontext
```
POST /v1/flux-kontext
```

### FLUX.1 Kontext Max
```
POST /v1/flux-kontext-max
```

### FLUX.1 Fill
```
POST /v1/flux-fill
```
"""


class _Client:
    def __init__(self, guide: bytes = _GUIDE) -> None:
        self.guide = guide
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        body = f'{{"sha":"{_REVISION}"}}'.encode() if "/commits/" in url else self.guide
        return HttpResponse(200, {"content-type": "text/plain"}, body, url)


def test_bfl_adapter_emits_only_hosted_api_ids_not_open_klein_weight_ids() -> None:
    client = _Client()
    page = BFLAPIModelsSourceAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 7
    assert {record.raw["api_model_id"] for record in page.records} == {
        "flux-2-max",
        "flux-2-pro",
        "flux-2-flex",
        "flux-pro-1.1",
        "flux-kontext",
        "flux-kontext-max",
        "flux-fill",
    }
    max_model = next(
        record for record in page.records if record.raw["api_model_id"] == "flux-2-max"
    )
    assert max_model.models[0].identifiers[0].namespace == "bfl:api-model"
    assert max_model.links[1].url == "https://api.bfl.ai/v1/flux-2-max"
    assert max_model.links[1].relation == "inference_endpoint"
    assert max_model.raw["source_path"] == "skills/bfl-api/references/endpoints.md"
    assert len(client.calls) == 2


def test_bfl_parser_rejects_duplicate_exact_api_ids() -> None:
    guide = """### FLUX.2 [max]
```
POST /v1/flux-2-max
```
### FLUX.2 [max alternate]
```
POST /v1/flux-2-max
```
"""
    with pytest.raises(ValueError, match="duplicate API model IDs"):
        _parse_endpoints(guide, maximum=10)


def test_bfl_parser_enforces_entry_limit() -> None:
    guide = """### FLUX.2 [max]
```
POST /v1/flux-2-max
```
### FLUX.2 [pro]
```
POST /v1/flux-2-pro
```
"""
    with pytest.raises(ValueError, match="entry limit"):
        _parse_endpoints(guide, maximum=1)


def test_bfl_adapter_does_not_claim_an_unchanged_snapshot_is_authoritative() -> None:
    page = BFLAPIModelsSourceAdapter(client=_Client()).fetch_page(
        {"completed_revision": _REVISION, "model_count": 7}
    )
    assert page.complete and not page.records
    assert page.upstream_count == 7
    assert not page.authoritative_snapshot
