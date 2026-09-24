"""Historical Workers AI model deprecations from Cloudflare's changelog.

This source records provider-declared retirement events. It does not imply that
an ID is currently callable, nor that the referenced model weights are hosted
by Cloudflare beyond the stated retirement date.
"""

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

_ID = r"@(?:cf|hf)/[A-Za-z0-9][A-Za-z0-9._/-]{1,180}"
_BULLET = re.compile(
    rf"^\s*[-*]\s+`(?P<id>{_ID})`(?:\s+-->\s+`(?P<replacement>{_ID})`)?\s*$",
    re.MULTILINE,
)
_SECTION = re.compile(
    r"(?m)^#{1,6}\s+Models deprecated on ([A-Z][a-z]+ \d{1,2}, \d{4})\s*$"
    r"(?P<body>.*?)(?=^#{1,6}\s|\Z)",
    re.DOTALL,
)


class CloudflareWorkersAIDeprecations:
    """Capture exact model IDs from a first-party deprecation announcement."""

    def __init__(
        self,
        *,
        name: str = "cloudflare-workers-ai-deprecations-2026-05",
        url: str = "https://developers.cloudflare.com/changelog/post/2026-05-08-planned-model-deprecations/",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 200,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if not self.url.startswith(
            "https://developers.cloudflare.com/changelog/post/2026-05-08-planned-model-deprecations"
        ):
            raise ValueError("URL must use Cloudflare's May 2026 deprecation announcement")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "cloudflare-workers-ai-deprecations-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/markdown"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: announcement returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: announcement exceeds {self.max_response_bytes} bytes")
        text = response.text()
        section_match = _SECTION.search(text)
        if section_match is None:
            raise ValueError(f"{self.name}: announcement lacks a dated deprecation section")
        retired_on = section_match.group(1)
        matches = tuple(_BULLET.finditer(section_match.group("body")))
        if not matches:
            raise ValueError(f"{self.name}: announcement contains no recognized model IDs")
        if len(matches) > self.max_entries:
            raise ValueError(f"{self.name}: announcement exceeds {self.max_entries} model entries")
        ids = [match.group("id") for match in matches]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.name}: announcement contains duplicate model IDs")
        snapshot = content_hash(response.body)
        records = tuple(
            self._record(match.group("id"), match.group("replacement"), retired_on, snapshot)
            for match in matches
        )
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": snapshot},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(
        self, model_id: str, replacement: str | None, retired_on: str, snapshot: str
    ) -> SourceRecord:
        identifier = Identifier("cloudflare:workers-ai", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=model_id,
            identifiers=(identifier,),
            status=ModelStatus.DOCUMENTED,
        )
        raw: dict[str, Any] = {
            "model_id": model_id,
            "provider_lifecycle_status": "deprecated",
            "deprecated_on": retired_on,
            "announcement_sha256": snapshot,
        }
        if replacement:
            raw["replacement_model_id"] = replacement
        return SourceRecord(
            source_record_id=f"deprecation:{model_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.url,
            title=f"Deprecated Workers AI model: {model_id}",
            raw=raw,
            text=f"Cloudflare announced {model_id} as deprecated on {retired_on}.",
            identifiers=(identifier,),
            links=(Link(self.url, relation="deprecation_announcement"),),
            models=(model,),
        )
