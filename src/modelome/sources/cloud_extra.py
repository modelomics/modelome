from __future__ import annotations

import re
from collections.abc import Mapping
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

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

DEFAULT_URL = "https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm"
_DETAIL_PATH = re.compile(r"^/en-us/iaas/Content/generative-ai/[A-Za-z0-9_.-]+\.htm$")
_MODEL_NAME = re.compile(
    r"(?:OCI Model Name|Model Name in OCI Generative AI)\s*:\s*"
    r"`?(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*)`?",
    re.IGNORECASE,
)
_BACKTICK = re.compile(r"`([A-Za-z][A-Za-z0-9_.-]{2,})`")
_OPENING_MODEL_ID = re.compile(r"`?([a-z][a-z0-9_-]*\.[a-z0-9][a-z0-9_.-]*)`?", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


class _CatalogLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[tuple[str, str]] = []
        self._href: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            label = " ".join(" ".join(self._parts).split())
            self.anchors.append((self._href, label))
            self._href = None
            self._parts = []


class OciGenerativeAIModelCatalog:
    """Enumerate OCI's public offered-model list and its linked model cards."""

    def __init__(
        self,
        *,
        name: str = "oci-generative-ai-pretrained-models",
        url: str = DEFAULT_URL,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = name
        self.url = canonicalize_url(url)
        parsed = urlsplit(self.url)
        if parsed.hostname != "docs.oracle.com" or parsed.path != urlsplit(DEFAULT_URL).path:
            raise ValueError("catalog URL must use Oracle's offered pretrained model catalog")
        self.max_response_bytes = int(max_response_bytes)
        self.max_entries = int(max_entries)
        if self.max_response_bytes < 1 or self.max_entries < 1:
            raise ValueError("response and entry limits must be positive")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "oci-generative-ai-public-model-catalog-v1",
            "url": self.url,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        catalog_response = self.client.get(self.url, headers={"Accept": "text/html"})
        catalog_html = self._body(catalog_response, "catalog")
        parser = _CatalogLinks()
        parser.feed(catalog_html)
        entries: dict[str, str] = {}
        for href, label in parser.anchors:
            absolute = canonicalize_url(urljoin(self.url, href))
            parts = urlsplit(absolute)
            if (
                parts.hostname != "docs.oracle.com"
                or not _DETAIL_PATH.fullmatch(parts.path)
                or not label
            ):
                continue
            entries.setdefault(absolute, label)
        if not entries:
            raise ValueError(f"{self.name}: catalog contained no linked OCI model cards")
        if len(entries) > self.max_entries:
            raise ValueError(f"{self.name}: catalog exceeds {self.max_entries} model entries")

        catalog_hash = content_hash(catalog_response.body)
        records: list[SourceRecord] = []
        for card_url, label in entries.items():
            card_response = self.client.get(card_url, headers={"Accept": "text/html"})
            card_html = self._body(card_response, "model card")
            model_id = _model_id(card_html)
            if model_id is None:
                raise ValueError(f"{self.name}: model card did not expose an OCI model name")
            records.append(self._record(card_url, label, model_id, catalog_hash))

        return SourcePage(
            records=tuple(records),
            next_state={"entry_count": len(records), "content_hash": catalog_hash},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _body(self, response: HttpResponse, description: str) -> str:
        if response.status != 200:
            raise ValueError(f"{self.name}: {description} returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: {description} exceeds {self.max_response_bytes} bytes"
            )
        return response.text()

    def _record(
        self, card_url: str, label: str, model_id: str, catalog_hash: str
    ) -> SourceRecord:
        identifier = Identifier("oci:generative-ai-model", model_id)
        model = ModelHint(
            local_id=f"{model_id}#model",
            name=label,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=card_url,
        )
        return SourceRecord(
            source_record_id=f"model:{model_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=card_url,
            title=label,
            raw={
                "catalog_url": self.url,
                "catalog_revision_sha256": catalog_hash,
                "oci_model_name": model_id,
            },
            text=f"{label}\nOCI model name: {model_id}",
            identifiers=(identifier,),
            links=(Link(self.url, relation="model_catalog"),),
            models=(model,),
        )


def _model_id(document: str) -> str | None:
    text = " ".join(unescape(_TAG.sub(" ", document)).split())
    if match := _MODEL_NAME.search(text):
        return match.group("id")
    # Some Oracle cards identify the model in the opening sentence instead of
    # the explicit key/value section. Keep this fallback scoped to the first
    # sentence; the later page contains unrelated model IDs in examples.
    opening = text[:1500]
    if match := _OPENING_MODEL_ID.search(opening):
        return match.group(1)
    candidates = _BACKTICK.findall(opening)
    return candidates[0] if candidates else None
