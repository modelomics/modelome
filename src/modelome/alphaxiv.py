from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, build_opener

from modelome.fetchers import PublicUrlPolicy
from modelome.http import (
    HttpClient,
    HttpFailure,
    HttpResponse,
    _redacted_url,
    _require_public_redirect,
    _retry_delay,
    _SafeRedirectHandler,
)
from modelome.models import ArtifactKind, Identifier, Link, SourceRecord
from modelome.normalize import (
    canonical_json,
    canonicalize_url,
    extract_url_mentions,
    extract_urls,
    infer_url_relation,
)

ALPHAXIV_MCP_ENDPOINT = "https://api.alphaxiv.org/mcp/v1"
ALPHAXIV_MCP_TOOL = "get_paper_content"
MCP_PROTOCOL_VERSION = "2026-07-28"
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

_MODERN_ARXIV_ID_RE = re.compile(r"\d{4}\.\d{4,5}")
_LEGACY_ARXIV_ID_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9.-]*(?:\.[A-Za-z][A-Za-z0-9.-]*)?/\d{7}"
)


class AlphaXivError(RuntimeError):
    """An alphaXiv exact-identifier enrichment could not be materialized."""


class AlphaXivSafetyError(AlphaXivError, ValueError):
    """The request would leave the documented alphaXiv interface or exact-ID scope."""


class AlphaXivResponseError(AlphaXivError):
    """The documented interface returned an invalid, failed, or oversized result."""


@dataclass(frozen=True, slots=True)
class AlphaXivEnrichmentPlan:
    """A non-enumerating alphaXiv request derived from one known arXiv identity."""

    arxiv_id: str
    arxiv_url: str
    alphaxiv_url: str
    endpoint: str
    tool: str
    arguments: Mapping[str, Any]


class McpToolCaller(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


def normalize_discovered_arxiv_id(value: str) -> str:
    """Validate an already-normalized, versionless arXiv identifier.

    This deliberately does not accept URLs, ``arXiv:`` prefixes, or version
    suffixes. Those forms must first be resolved by a primary discovery source;
    alphaXiv is enrichment evidence and never an identity-discovery mechanism.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("arXiv ID must be a non-empty string")
    if value != value.strip():
        raise ValueError("arXiv ID must already be normalized")
    if not (_MODERN_ARXIV_ID_RE.fullmatch(value) or _LEGACY_ARXIV_ID_RE.fullmatch(value)):
        raise ValueError(f"invalid normalized arXiv ID: {value!r}")
    return value


def plan_alphaxiv_enrichment(
    arxiv_id: str,
    *,
    full_text: bool = True,
) -> AlphaXivEnrichmentPlan:
    """Plan alphaXiv enrichment without issuing a search or enumeration request."""

    normalized = normalize_discovered_arxiv_id(arxiv_id)
    encoded = quote(normalized, safe="/._-")
    arxiv_url = canonicalize_url(f"https://arxiv.org/abs/{encoded}")
    alphaxiv_url = canonicalize_url(f"https://www.alphaxiv.org/abs/{encoded}")
    return AlphaXivEnrichmentPlan(
        arxiv_id=normalized,
        arxiv_url=arxiv_url,
        alphaxiv_url=alphaxiv_url,
        endpoint=ALPHAXIV_MCP_ENDPOINT,
        tool=ALPHAXIV_MCP_TOOL,
        arguments={"url": arxiv_url, "fullText": bool(full_text)},
    )


class AlphaXivExactIdEnricher:
    """Read alphaXiv evidence for an arXiv ID already discovered elsewhere."""

    def __init__(
        self,
        client: McpToolCaller,
        *,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        url_policy: PublicUrlPolicy | None = None,
    ) -> None:
        if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.client = client
        self.max_response_bytes = max_response_bytes
        self.url_policy = url_policy or PublicUrlPolicy()

    def enrich(self, arxiv_id: str) -> SourceRecord:
        plan = plan_alphaxiv_enrichment(arxiv_id, full_text=True)
        response = self.client.call_tool(plan.tool, plan.arguments)
        result = _normalize_mcp_result(response, self.max_response_bytes)
        if result.get("isError") is True:
            detail = _tool_text(result) or "alphaXiv tool reported an error"
            raise AlphaXivResponseError(detail[:500])

        text = _tool_text(result)
        if not text:
            raise AlphaXivResponseError("alphaXiv get_paper_content returned no text")

        links = _evidence_links(
            text=text,
            result=result,
            plan=plan,
            policy=self.url_policy,
        )
        raw = {
            "provider": "alphaXiv",
            "interface": "mcp",
            "endpoint": plan.endpoint,
            "tool": plan.tool,
            "arguments": dict(plan.arguments),
            "alphaxiv_url": plan.alphaxiv_url,
            "mcp_result": result,
        }
        _bounded_json(raw, self.max_response_bytes)
        return SourceRecord(
            source_record_id=plan.arxiv_id,
            kind=ArtifactKind.PAPER,
            canonical_url=plan.arxiv_url,
            title=_result_title(result) or f"arXiv {plan.arxiv_id}",
            raw=raw,
            text=text,
            identifiers=(Identifier("arxiv", plan.arxiv_id),),
            links=links,
        )


class AlphaXivMcpClient:
    """Bounded client for alphaXiv's documented, current Streamable HTTP MCP API.

    It intentionally exposes only ``get_paper_content``. In particular, the
    alphaXiv search and recommendation tools cannot be used to seed discovery.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = ALPHAXIV_MCP_ENDPOINT,
        http: Any | None = None,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.endpoint = _official_endpoint(endpoint)
        candidate_key = api_key if api_key is not None else os.environ.get("ALPHAXIV_API_KEY")
        if not isinstance(candidate_key, str) or not candidate_key.strip():
            raise AlphaXivSafetyError("alphaXiv MCP enrichment requires an API key")
        if candidate_key != candidate_key.strip() or any(
            character in candidate_key for character in "\r\n"
        ):
            raise AlphaXivSafetyError("invalid alphaXiv API key")
        if not isinstance(max_response_bytes, int) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self._api_key = candidate_key
        self.max_response_bytes = max_response_bytes
        self.http = http or _PostHttpClient(max_response_bytes=max_response_bytes)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if name != ALPHAXIV_MCP_TOOL:
            raise AlphaXivSafetyError(
                "alphaXiv enrichment permits only the exact-ID get_paper_content tool"
            )
        expected = plan_alphaxiv_enrichment(_arxiv_id_from_arguments(arguments))
        if dict(arguments) != dict(expected.arguments):
            raise AlphaXivSafetyError(
                "alphaXiv tool arguments must be the deterministic exact-ID enrichment plan"
            )

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": ALPHAXIV_MCP_TOOL,
                "arguments": dict(expected.arguments),
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "modelome",
                        "version": "0.1.0",
                    }
                },
            },
        }
        body = canonical_json(payload).encode("utf-8")
        if len(body) > self.max_response_bytes:
            raise AlphaXivResponseError("alphaXiv MCP request exceeds the configured byte limit")
        headers = {
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            "Mcp-Method": "tools/call",
            "Mcp-Name": ALPHAXIV_MCP_TOOL,
        }
        response = self.http.post(
            self.endpoint,
            body=body,
            headers=headers,
            redirect_validator=_require_official_endpoint_redirect,
        )
        return _parse_mcp_response(response, self.max_response_bytes)


class _PostHttpClient(HttpClient):
    """POST support with the shared HTTP client's bounds and retry policy."""

    def post(
        self,
        url: str,
        *,
        body: bytes,
        headers: Mapping[str, str],
        redirect_validator: Callable[[str], None] | None = None,
    ) -> HttpResponse:
        display_url = _redacted_url(url)
        request_headers = {"User-Agent": self.user_agent, **dict(headers)}
        opener = self._opener
        if redirect_validator is not None:

            def validate_redirect(target: str) -> None:
                _require_public_redirect(target)
                redirect_validator(target)

            opener = build_opener(_SafeRedirectHandler(validate_redirect))

        last_error: Exception | None = None
        for attempt in range(self.attempts):
            request = Request(url, data=body, headers=request_headers, method="POST")
            try:
                with opener.open(request, timeout=self.timeout) as response:  # noqa: S310
                    response_body = response.read(self.max_response_bytes + 1)
                    if len(response_body) > self.max_response_bytes:
                        raise HttpFailure(
                            f"POST {display_url} exceeded {self.max_response_bytes} bytes"
                        )
                    return HttpResponse(
                        status=response.status,
                        headers={
                            key.casefold(): value for key, value in response.headers.items()
                        },
                        body=response_body,
                        url=_redacted_url(response.url),
                    )
            except HTTPError as error:
                last_error = error
                if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                    break
                delay = _retry_delay(error.headers, attempt)
            except (TimeoutError, URLError) as error:
                last_error = error
                delay = min(2**attempt, 30)
            if attempt + 1 < self.attempts:
                self._sleep(delay)
        raise HttpFailure(f"POST {display_url} failed: {last_error}") from last_error


def _official_endpoint(value: str) -> str:
    if not isinstance(value, str):
        raise AlphaXivSafetyError("alphaXiv MCP endpoint must be a string")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as error:
        raise AlphaXivSafetyError("invalid alphaXiv MCP endpoint") from error
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold().rstrip(".") != "api.alphaxiv.org"
        or port not in {None, 443}
        or parts.username is not None
        or parts.password is not None
        or parts.path.rstrip("/") != "/mcp/v1"
        or parts.query
        or parts.fragment
    ):
        raise AlphaXivSafetyError(
            "alphaXiv credentials may only be sent to https://api.alphaxiv.org/mcp/v1"
        )
    return ALPHAXIV_MCP_ENDPOINT


def _require_official_endpoint_redirect(value: str) -> None:
    _official_endpoint(value)


def _arxiv_id_from_arguments(arguments: Mapping[str, Any]) -> str:
    if not isinstance(arguments, Mapping):
        raise AlphaXivSafetyError("alphaXiv tool arguments must be a mapping")
    if set(arguments) != {"url", "fullText"} or arguments.get("fullText") is not True:
        raise AlphaXivSafetyError("alphaXiv enrichment requires exact URL and fullText=true")
    url = arguments.get("url")
    if not isinstance(url, str):
        raise AlphaXivSafetyError("alphaXiv enrichment URL must be a string")
    prefix = "https://arxiv.org/abs/"
    if not url.startswith(prefix):
        raise AlphaXivSafetyError("alphaXiv enrichment URL must be an arXiv abstract URL")
    return normalize_discovered_arxiv_id(url[len(prefix) :])


def _parse_mcp_response(response: HttpResponse, max_response_bytes: int) -> Mapping[str, Any]:
    if not isinstance(response, HttpResponse):
        raise AlphaXivResponseError("alphaXiv MCP transport returned no HTTP response")
    if response.status < 200 or response.status >= 300:
        raise AlphaXivResponseError(f"alphaXiv MCP returned HTTP {response.status}")
    if len(response.body) > max_response_bytes:
        raise AlphaXivResponseError("alphaXiv MCP response exceeds the configured byte limit")

    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
    try:
        if content_type == "text/event-stream":
            envelope = _last_sse_message(response.body)
        elif content_type in {"", "application/json"} or content_type.endswith("+json"):
            envelope = json.loads(response.body)
        else:
            raise AlphaXivResponseError(
                f"alphaXiv MCP returned unsupported content type {content_type!r}"
            )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AlphaXivResponseError("alphaXiv MCP returned invalid JSON") from error
    if not isinstance(envelope, Mapping):
        raise AlphaXivResponseError("alphaXiv MCP response must be a JSON object")
    if envelope.get("jsonrpc") != "2.0" or envelope.get("id") != 1:
        raise AlphaXivResponseError("alphaXiv MCP returned an unmatched JSON-RPC response")
    error = envelope.get("error")
    if error is not None:
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str) and message:
                raise AlphaXivResponseError(f"alphaXiv MCP error: {message[:500]}")
        raise AlphaXivResponseError("alphaXiv MCP returned a JSON-RPC error")
    result = envelope.get("result")
    return _normalize_mcp_result(result, max_response_bytes)


def _last_sse_message(body: bytes) -> Any:
    text = body.decode("utf-8")
    messages: list[Any] = []
    data_lines: list[str] = []
    for line in (*text.splitlines(), ""):
        if not line:
            if data_lines:
                messages.append(json.loads("\n".join(data_lines)))
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if not messages:
        raise AlphaXivResponseError("alphaXiv MCP event stream contained no JSON-RPC response")
    return messages[-1]


def _normalize_mcp_result(value: Any, max_response_bytes: int) -> Mapping[str, Any]:
    if isinstance(value, Mapping) and value.get("jsonrpc") == "2.0" and "result" in value:
        value = value["result"]
    normalized = _json_value(value)
    if not isinstance(normalized, Mapping):
        raise AlphaXivResponseError("alphaXiv MCP tool result must be a JSON object")
    _bounded_json(normalized, max_response_bytes)
    content = normalized.get("content")
    if not isinstance(content, list):
        raise AlphaXivResponseError("alphaXiv MCP tool result has no content list")
    return normalized


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AlphaXivResponseError("alphaXiv MCP result contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise AlphaXivResponseError("alphaXiv MCP result contains a non-string key")
            result[key] = _json_value(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(child) for child in value]
    raise AlphaXivResponseError(
        f"alphaXiv MCP result contains unsupported value {type(value).__name__}"
    )


def _bounded_json(value: Mapping[str, Any], max_response_bytes: int) -> bytes:
    payload = canonical_json(value).encode("utf-8")
    if len(payload) > max_response_bytes:
        raise AlphaXivResponseError("alphaXiv MCP result exceeds the configured byte limit")
    return payload


def _tool_text(result: Mapping[str, Any]) -> str:
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return ""
    text_parts = [
        block["text"].strip()
        for block in blocks
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and block["text"].strip()
    ]
    return "\n\n".join(text_parts)


def _result_title(result: Mapping[str, Any]) -> str | None:
    structured = result.get("structuredContent")
    if not isinstance(structured, Mapping):
        return None
    title = structured.get("title")
    if not isinstance(title, str):
        paper = structured.get("paper")
        title = paper.get("title") if isinstance(paper, Mapping) else None
    if not isinstance(title, str):
        return None
    title = " ".join(title.split())
    return title if 0 < len(title) <= 1_000 else None


def _evidence_links(
    *,
    text: str,
    result: Mapping[str, Any],
    plan: AlphaXivEnrichmentPlan,
    policy: PublicUrlPolicy,
) -> tuple[Link, ...]:
    links: dict[str, Link] = {
        plan.alphaxiv_url: Link(
            url=plan.alphaxiv_url,
            relation="enriched_by",
            locator="alphaxiv:mcp:get_paper_content",
            crawl=False,
        )
    }
    for url, locator in extract_url_mentions(text):
        _add_evidence_link(
            links,
            url,
            relation=infer_url_relation(text, locator),
            locator=locator,
            policy=policy,
        )
    for url in sorted(extract_urls(result)):
        _add_evidence_link(
            links,
            url,
            relation="references",
            locator="alphaxiv:mcp:result",
            policy=policy,
        )
    return tuple(links.values())


def _add_evidence_link(
    links: dict[str, Link],
    url: str,
    *,
    relation: str,
    locator: str,
    policy: PublicUrlPolicy,
) -> None:
    if not policy.allows(url, resolve=False):
        return
    safe_url = policy.validate(url, resolve=False)
    if safe_url in links:
        return
    own_paper = safe_url.startswith(("https://arxiv.org/", "https://www.alphaxiv.org/"))
    links[safe_url] = Link(
        url=safe_url,
        relation=relation,
        locator=locator,
        crawl=not own_paper,
    )
