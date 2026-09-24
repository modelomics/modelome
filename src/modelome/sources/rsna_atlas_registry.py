"""RSNA ATLAS's published radiology AI model-card catalog."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_GRAPHQL_URL = (
    "https://inrmbpna7vbw7kkmerf25hy7hu.appsync-api.us-east-1.amazonaws.com/graphql"
)
# AppSync API keys are public client identifiers. This value is served by the
# first-party ATLAS web client; it grants read access to published cards only.
_PUBLIC_API_KEY = "da2-iabc3kghlvbhfnzoxy3gnh2sia"
_CARDS_URL = "https://atlas.rsna.org/cards/"
_QUERY = """query AtlasPublishedCards($limit: Int, $nextToken: String) {
  listIndexCards(
    filter: {publishingStatus: {eq: "published"}}
    limit: $limit
    nextToken: $nextToken
  ) {
    items { id indexCardId title schemaVersion publishingStatus roadmapObject }
    nextToken
  }
}"""
_URL_RE = re.compile(r"https?://[^\s<>\]\[{}\"']+")
_MAX_PAGE_SIZE = 200
_MAX_ENTRIES = 5000


class RsnaAtlasRegistrySourceAdapter:
    """Page published ATLAS model cards and preserve only card-declared facts.

    ATLAS is a model-card registry rather than a checkpoint file manifest.
    Availability is free text; URLs mentioned there remain untyped references
    and are never presented as direct weight files.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers published RSNA ATLAS records whose schemaVersion identifies a "
        "model card. Model availability is free text: URLs are recorded only as "
        "availability references, not asserted to be checkpoint files. Dataset "
        "cards and unpublished cards are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "rsna-atlas-medical-ai-model-cards",
        page_size: int = 100,
        max_response_bytes: int = 32 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not name.strip() or not 1 <= page_size <= _MAX_PAGE_SIZE:
            raise ValueError("name and page_size from 1 to 200 are required")
        if max_response_bytes <= 0 or not 1 <= max_entries <= _MAX_ENTRIES:
            raise ValueError("positive response limit and max_entries up to 5000 are required")
        self.name = name
        self.page_size = page_size
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash({
            "adapter": "rsna-atlas-published-model-cards-v1",
            "endpoint": _GRAPHQL_URL,
            "query": _QUERY,
            "page_size": page_size,
            "max_entries": max_entries,
            "max_response_bytes": max_response_bytes,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        variables: dict[str, Any] = {"limit": self.page_size}
        next_token = state.get("next_token")
        if next_token is not None:
            if not isinstance(next_token, str) or not next_token:
                raise ValueError(f"{self.name}: invalid pagination token")
            variables["nextToken"] = next_token
        response = self.client.get(
            _GRAPHQL_URL,
            params={"query": _QUERY, "variables": json.dumps(variables, separators=(",", ":"))},
            headers={"Accept": "application/json", "x-api-key": _PUBLIC_API_KEY},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: GraphQL returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: response exceeds configured byte limit")
        payload = response.json()
        if not isinstance(payload, Mapping) or payload.get("errors"):
            raise ValueError(f"{self.name}: GraphQL returned errors")
        data = payload.get("data")
        listing = data.get("listIndexCards") if isinstance(data, Mapping) else None
        if not isinstance(listing, Mapping) or not isinstance(listing.get("items"), list):
            raise ValueError(f"{self.name}: malformed card listing")
        items = listing["items"]
        if len(items) > self.page_size:
            raise ValueError(f"{self.name}: upstream page exceeded requested page size")
        cursor = listing.get("nextToken")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ValueError(f"{self.name}: malformed next token")
        page_index = state.get("page_index", 0)
        if not isinstance(page_index, int) or page_index < 0:
            raise ValueError(f"{self.name}: invalid page index")
        if page_index * self.page_size + len(items) > self.max_entries:
            raise ValueError(f"{self.name}: listing exceeds configured entry limit")
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for item in items:
            if not _is_published_model(item):
                continue
            try:
                records.append(_record(item))
            except ValueError as error:
                card_id = item.get("id") if isinstance(item, Mapping) else None
                issues.append(SourceIssue(
                    source_record_id=(
                        f"rsna-atlas:model-card:{card_id}"
                        if isinstance(card_id, str) and card_id
                        else f"rsna-atlas:model-card:invalid-page-{page_index}"
                    ),
                    stage="normalize",
                    error=str(error),
                    summary={"title": item.get("title") if isinstance(item, Mapping) else None},
                ))
        return SourcePage(
            tuple(records),
            {"next_token": cursor, "page_index": page_index + 1},
            complete=cursor is None,
            upstream_count=None,
            authoritative_snapshot=False,
            issues=tuple(issues),
            advance_on_source_issues=bool(issues),
        )


def _is_published_model(item: Any) -> bool:
    if (
        not isinstance(item, Mapping)
        or item.get("publishingStatus") != "published"
        or not isinstance(item.get("schemaVersion"), str)
        or not item["schemaVersion"].endswith("/model.json")
    ):
        return False
    roadmap = item.get("roadmapObject")
    if isinstance(roadmap, str):
        try:
            roadmap = json.loads(roadmap)
        except json.JSONDecodeError:
            # Keep malformed model rows in scope so _record quarantines them.
            return True
    # ATLAS currently has cards with a model schemaVersion but a legitimate
    # Dataset payload. They are provider schema mislabels, not malformed model
    # records, so exclude them from this model-only source.
    return not (
        isinstance(roadmap, Mapping)
        and "Dataset" in roadmap
        and "Model" not in roadmap
    )


def _record(item: Mapping[str, Any]) -> SourceRecord:
    if not _is_published_model(item):
        raise ValueError("ATLAS row is not a published model card")
    card_id = item.get("id")
    provider_id = item.get("indexCardId")
    schema_version = item.get("schemaVersion")
    if not isinstance(card_id, str) or not card_id.strip():
        raise ValueError("ATLAS model card lacks its card UUID")
    if not isinstance(provider_id, str) or not provider_id.strip():
        raise ValueError("ATLAS model card lacks its provider index ID")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ValueError("ATLAS model card lacks schema version")
    title = item.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("ATLAS model card lacks title")
    roadmap = item.get("roadmapObject")
    if isinstance(roadmap, str):
        try:
            roadmap = json.loads(roadmap)
        except json.JSONDecodeError as error:
            raise ValueError("ATLAS model card has invalid ROADMAP JSON") from error
    if not isinstance(roadmap, Mapping) or not isinstance(roadmap.get("Model"), Mapping):
        raise ValueError("ATLAS model card lacks its Model object")
    model_data = roadmap["Model"]
    name = model_data.get("Name")
    model_url = model_data.get("Link")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("ATLAS model card lacks Model.Name")
    if not _http_url(model_url):
        raise ValueError("ATLAS model card has invalid Model.Link")
    properties = model_data.get("Model properties", {})
    properties = properties if isinstance(properties, Mapping) else {}
    availability = properties.get("Availability")
    availability_text = availability if isinstance(availability, str) else ""
    descriptor = model_data.get("Descriptors", {})
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    version = descriptor.get("Version")
    local_id = f"model:{card_id}"
    card_url = _CARDS_URL + card_id
    links = [
        Link(card_url, "provider_model_card", crawl=False),
        Link(model_url, "model_source", crawl=False, model_local_ids=(local_id,)),
    ]
    for url in _availability_urls(availability_text):
        if url != model_url:
            links.append(Link(
                url, "availability_reference", locator="ATLAS Model properties.Availability",
                crawl=False, model_local_ids=(local_id,),
            ))
    model = ModelHint(
        local_id,
        name.strip(),
        identifiers=(
            Identifier("rsna-atlas:card-id", card_id),
            Identifier("rsna-atlas:index-card-id", provider_id),
        ),
        aliases=(title.strip(),) if title.strip() != name.strip() else (),
        # Card publication does not imply the model weights are released.
        status=ModelStatus.DOCUMENTED,
        locator=f"RSNA ATLAS card {card_id}: Model.Name",
    )
    releases: tuple[ReleaseHint, ...] = ()
    if isinstance(version, str) and version.strip():
        releases = (ReleaseHint(
            f"card-version:{card_id}", local_id,
            version=version.strip(),
            identifiers=(Identifier("rsna-atlas:card-version", f"{card_id}:{version.strip()}"),),
            metadata={"availability_statement": availability_text,
                      "model_source_url": model_url, "schema_version": schema_version},
            locator=f"RSNA ATLAS card {card_id}: Model.Descriptors.Version",
        ),)
    return SourceRecord(
        source_record_id=f"rsna-atlas:model-card:{card_id}",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=canonicalize_url(card_url),
        title=name.strip(),
        raw={"card_id": card_id, "index_card_id": provider_id,
             "schema_version": schema_version, "roadmap_object": roadmap,
             "availability_statement": availability_text},
        text="\n".join((name.strip(), title.strip(), availability_text)).strip(),
        identifiers=(Identifier("rsna-atlas:card-id", card_id),),
        links=tuple(links),
        models=(model,),
        releases=releases,
    )


def _availability_urls(text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.findall(text):
        url = match.rstrip(".,;:)")
        if _http_url(url) and url not in seen:
            seen.add(url)
            urls.append(url)
    return tuple(urls)


def _http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parts = urlsplit(value)
    return (
        parts.scheme == "https"
        and bool(parts.hostname)
        and not parts.username
        and not parts.password
    )


__all__ = ["RsnaAtlasRegistrySourceAdapter"]
