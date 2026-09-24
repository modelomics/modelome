from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.http import HttpClient
from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.normalize import content_hash

_GRAPH_API = "https://api.semanticscholar.org/graph/v1"
_DIRECTIONS = {"citations": ("citingPaper", "cited_by"), "references": ("citedPaper", "cites")}


class SemanticScholarCitationGraphAdapter:
    """Page exact citation/reference edges for one Semantic Scholar paper.

    The adapter is intentionally scoped to one anchor paper. Its checkpoint is
    an integer offset from the Graph API `next` field; it does not infer edges
    from titles, authors, or other approximate metadata.
    """

    def __init__(
        self,
        *,
        paper_id: str,
        paper_url: str,
        paper_title: str,
        direction: str = "references",
        page_size: int = 1000,
        max_records: int = 9_999,
        api_key: str | None = None,
        client: HttpClient | Any | None = None,
        name: str = "semantic-scholar-citation-edges",
        url: str = _GRAPH_API,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.paper_id = _required_text(paper_id, "paper ID")
        self.paper_url = _https_url(paper_url, "anchor paper URL")
        self.paper_title = _required_text(paper_title, "anchor paper title")
        if direction not in _DIRECTIONS:
            raise ValueError("direction must be 'citations' or 'references'")
        self.direction = direction
        self.child_field, self.relation = _DIRECTIONS[direction]
        self.page_size = _positive_int(page_size, "page size")
        if self.page_size > 1000:
            raise ValueError("Semantic Scholar Graph API page size cannot exceed 1000")
        self.max_records = _positive_int(max_records, "maximum records")
        self.url = _api_base_url(url)
        self._api_key = _optional_text(api_key)
        self.client = client or HttpClient()
        self.checkpoint_signature = content_hash(
            {
                "adapter": "semantic-scholar-citation-graph-v1",
                "paper_id": self.paper_id,
                "paper_url": self.paper_url,
                "direction": self.direction,
                "page_size": self.page_size,
                "max_records": self.max_records,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        """Fetch one page and return edge records plus a resumable offset."""

        signature = state.get("signature")
        if signature is not None and signature != self.checkpoint_signature:
            raise ValueError(f"{self.name}: checkpoint belongs to another query")
        offset = _nonnegative_int(state.get("offset", 0), "checkpoint offset")
        if offset > self.max_records:
            raise ValueError(f"{self.name}: checkpoint exceeds maximum record limit")
        if state.get("done") is True:
            return SourcePage(records=(), next_state=dict(state), complete=True, upstream_count=0)

        endpoint = f"{self.url}/paper/{quote(self.paper_id, safe='')}/{self.direction}"
        fields = ",".join(
            (
                f"{self.child_field}.paperId",
                f"{self.child_field}.corpusId",
                f"{self.child_field}.url",
                f"{self.child_field}.title",
                "contexts",
                "intents",
                "isInfluential",
            )
        )
        headers = {"x-api-key": self._api_key} if self._api_key else None
        response = self.client.get(
            endpoint,
            params={"offset": offset, "limit": self.page_size, "fields": fields},
            headers=headers,
        )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: Graph API page must be an object")
        response_offset = _nonnegative_int(payload.get("offset"), "response offset")
        if response_offset != offset:
            raise ValueError(f"{self.name}: Graph API returned a nonmatching offset")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError(f"{self.name}: Graph API page data must be an array")
        if offset + len(rows) > self.max_records:
            raise ValueError(f"{self.name}: page exceeds configured maximum record limit")

        records = tuple(self._edge_record(row) for row in rows)
        next_value = payload.get("next")
        if next_value is None or (
            isinstance(next_value, int)
            and not isinstance(next_value, bool)
            and next_value == 0
        ):
            next_offset = offset + len(rows)
            done = True
        else:
            next_offset = _positive_int(next_value, "response next offset")
            if next_offset != offset + len(rows):
                raise ValueError(f"{self.name}: Graph API next offset skips or repeats edges")
            if not rows:
                raise ValueError(f"{self.name}: Graph API returned an empty nonfinal page")
            done = False
        if next_offset > self.max_records:
            raise ValueError(f"{self.name}: next offset exceeds configured maximum record limit")
        return SourcePage(
            records=records,
            next_state={
                "signature": self.checkpoint_signature,
                "offset": next_offset,
                "done": done,
            },
            complete=done,
            upstream_count=None,
        )

    def _edge_record(self, row: Any) -> SourceRecord:
        if not isinstance(row, Mapping):
            raise ValueError(f"{self.name}: citation edge must be an object")
        paper = row.get(self.child_field)
        if not isinstance(paper, Mapping):
            raise ValueError(f"{self.name}: edge is missing {self.child_field}")
        child_id = _required_text(paper.get("paperId"), f"{self.child_field}.paperId")
        child_url = _https_url(paper.get("url"), f"{self.child_field}.url")
        _required_text(paper.get("title"), f"{self.child_field}.title")
        corpus_id = _optional_exact_id(paper.get("corpusId"), f"{self.child_field}.corpusId")

        if self.direction == "references":
            citing_id, cited_id = self.paper_id, child_id
            link_url = child_url
            link_locator = f"$.{self.child_field}.url"
        else:
            citing_id, cited_id = child_id, self.paper_id
            link_url = child_url
            link_locator = f"$.{self.child_field}.url"
        edge_identity = {"citing_paper_id": citing_id, "cited_paper_id": cited_id}
        record_id = f"{self.name}:{self.direction}:{content_hash(edge_identity)}"
        identifiers = [
            Identifier("semantic-scholar:query-id", self.paper_id),
            Identifier("semantic-scholar:paper-id", child_id),
        ]
        if corpus_id is not None:
            identifiers.append(Identifier("semantic-scholar:corpus-id", corpus_id))
        return SourceRecord(
            source_record_id=record_id,
            kind=ArtifactKind.PAPER,
            canonical_url=self.paper_url,
            title=self.paper_title,
            raw={
                "record_type": "citation_edge",
                "direction": self.direction,
                "anchor_paper_id": self.paper_id,
                "citing_paper_id": citing_id,
                "cited_paper_id": cited_id,
                "edge": dict(row),
            },
            identifiers=tuple(identifiers),
            links=(Link(url=link_url, relation=self.relation, locator=link_locator, crawl=False),),
        )


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value.strip()


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a positive integer") from error
    if parsed < 1 or str(parsed) != str(value):
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a nonnegative integer") from error
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"{label} must be a nonnegative integer")
    return parsed


def _optional_exact_id(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a string or integer")
    return str(value)


def _https_url(value: Any, label: str) -> str:
    url = _required_text(value, label)
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError(f"{label} must be an absolute HTTPS URL")
    try:
        ip = ipaddress.ip_address(parts.hostname)
    except ValueError:
        ip = None
    if ip is not None and not ip.is_global:
        raise ValueError(f"{label} must not use a non-public IP address")
    return url


def _api_base_url(value: Any) -> str:
    url = _https_url(value, "Graph API URL").rstrip("/")
    parts = urlsplit(url)
    if parts.query or parts.fragment:
        raise ValueError("Graph API URL must not include query or fragment")
    return url
