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

# Workers AI publishes its model catalog as first-party documentation. The
# provider-scoped IDs are stable public identifiers (for example @cf/meta/...).
_MODEL_ID = re.compile(r"(?<![A-Za-z0-9._/-])@cf/[A-Za-z0-9][A-Za-z0-9._/-]{1,180}")


class CloudflareWorkersAIModelCatalog:
    """Read the public Cloudflare Workers AI model list, without account APIs."""

    def __init__(
        self,
        *,
        name: str = "cloudflare-workers-ai-models",
        url: str = "https://developers.cloudflare.com/workers-ai/models/",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 500,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        if not self.url.startswith("https://developers.cloudflare.com/workers-ai/models"):
            raise ValueError("catalog URL must use Cloudflare's Workers AI model catalog")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "cloudflare-workers-ai-public-catalog-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(self.url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_response_bytes} bytes")
        body = response.text()
        ids = tuple(dict.fromkeys(_MODEL_ID.findall(body)))
        if not ids:
            raise ValueError(f"{self.name}: public catalog contained no Workers AI model IDs")
        if len(ids) > self.max_entries:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_entries} model entries")
        catalog_hash = content_hash(response.body)
        records = tuple(self._record(model_id, catalog_hash) for model_id in ids)
        return SourcePage(
            records=records,
            next_state={"entry_count": len(records), "content_hash": catalog_hash},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, model_id: str, catalog_hash: str) -> SourceRecord:
        url = self.url
        identifier = Identifier("cloudflare:workers-ai", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=model_id,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
        )
        return SourceRecord(
            source_record_id=f"model:{model_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=url,
            title=model_id,
            raw={
                "catalog_url": self.url,
                "catalog_revision_sha256": catalog_hash,
                "model_id": model_id,
            },
            text=model_id,
            identifiers=(identifier,),
            links=(Link(self.url, relation="model_catalog"),),
            models=(model,),
        )
