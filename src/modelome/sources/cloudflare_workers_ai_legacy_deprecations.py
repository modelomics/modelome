"""Older Workers AI model retirements recorded in Cloudflare's changelog."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_URL = "https://developers.cloudflare.com/workers-ai/changelog"
_ID = r"@(?:cf|hf)/[A-Za-z0-9][A-Za-z0-9._/-]{1,180}"
_EXPECTED_RETIREMENTS = 19
_BLOCK = re.compile(
    r"Some older Workers AI models are being deprecated on October 1st, 2025\.[^\n]*"
    r"(?P<body>.*?)(?=^##\s+\d{4}-\d{2}-\d{2}\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_ID_BULLET = re.compile(rf"^\s*[-*]\s+`?(?P<id>{_ID})`?\s*$", re.MULTILINE)


class CloudflareWorkersAILegacyDeprecations:
    """Record the 19 exact model IDs Cloudflare retired on 2025-10-01."""

    def __init__(
        self,
        *,
        name: str = "cloudflare-workers-ai-deprecations-2025-10",
        url: str = _URL,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 200,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if self.url != _URL:
            raise ValueError("URL must use Cloudflare's Workers AI changelog")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "cloudflare-workers-ai-legacy-deprecations-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/markdown"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: changelog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: changelog exceeds {self.max_response_bytes} bytes")
        text = response.text()
        block = _BLOCK.search(text)
        if block is None:
            raise ValueError(f"{self.name}: changelog lacks the October 2025 deprecation notice")
        matches = tuple(_ID_BULLET.finditer(block.group("body")))
        ids = [match.group("id") for match in matches]
        if not ids:
            raise ValueError(f"{self.name}: deprecation notice contains no recognized model IDs")
        if len(ids) > self.max_entries:
            raise ValueError(
                f"{self.name}: deprecation notice exceeds {self.max_entries} model entries"
            )
        if len(ids) != _EXPECTED_RETIREMENTS:
            raise ValueError(
                f"{self.name}: expected {_EXPECTED_RETIREMENTS} retirement IDs, found {len(ids)}"
            )
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: deprecation notice contains duplicate model IDs")
        snapshot = content_hash(response.body)
        records = tuple(self._record(model_id, snapshot) for model_id in ids)
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": snapshot},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(self, model_id: str, snapshot: str) -> SourceRecord:
        identifier = Identifier("cloudflare:workers-ai", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=model_id,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
        )
        return SourceRecord(
            source_record_id=f"deprecation:{model_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title=f"Deprecated Workers AI model: {model_id}",
            raw={
                "model_id": model_id,
                "provider_lifecycle_status": "deprecated",
                "deprecated_on": "October 1, 2025",
                "announcement_date": "2025-09-18",
                "announcement_sha256": snapshot,
            },
            text=f"Cloudflare announced {model_id} as deprecated on October 1, 2025.",
            identifiers=(identifier,),
            links=(Link(self.url, relation="deprecation_announcement"),),
            models=(model,),
        )
