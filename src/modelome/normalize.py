from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from modelome.models import Identifier

_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_ARXIV_RE = re.compile(
    r"^(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z0-9.-]*/\d{7}(?:v\d+)?)(?:\.pdf)?$",
    re.I,
)
_TEXT_LOCATOR_RE = re.compile(r"^text:(\d+)-(\d+)$")
_IMPLEMENTATION_NOUN = (
    r"(?:source\s+code|implementation|codebase|code\s+repository|repository|repo)"
)
_IMPLEMENTATION_ACTION = (
    r"(?:available|released|hosted|provided|published|found|downloaded)"
)
_OFFICIAL_IMPLEMENTATION_RE = re.compile(
    rf"\bofficial\s+{_IMPLEMENTATION_NOUN}\b[^.!?\n]{{0,100}}$",
    re.IGNORECASE,
)
_IMPLEMENTATION_BEFORE_RE = re.compile(
    rf"(?:"
    rf"\b{_IMPLEMENTATION_NOUN}\b[^.!?\n]{{0,100}}\b{_IMPLEMENTATION_ACTION}\b"
    rf"(?:\s+(?:at|on|from|via|here))?"
    rf"|\b{_IMPLEMENTATION_ACTION}\b[^.!?\n]{{0,100}}\b{_IMPLEMENTATION_NOUN}\b"
    rf"(?:\s+(?:at|on|from|via|here))?"
    rf"|\b{_IMPLEMENTATION_NOUN}\b\s*[:=\-–—\[\]()]*"
    rf")\s*$",
    re.IGNORECASE,
)
_IMPLEMENTATION_AFTER_RE = re.compile(
    rf"^\s*[\])}}.,;:\-–—]*\s*(?:the\s+)?(?:official\s+)?"
    rf"{_IMPLEMENTATION_NOUN}\b",
    re.IGNORECASE,
)
_WEIGHTS_RE = re.compile(
    r"\b(?:trained\s+weights?|model\s+weights?|checkpoints?)\b[^.!?\n]{0,100}$",
    re.IGNORECASE,
)
_MODEL_CARD_RE = re.compile(
    r"\bmodel\s+card\b[^.!?\n]{0,100}$",
    re.IGNORECASE,
)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return _SPACE_RE.sub(" ", value).strip()


def canonical_json(value: Mapping[str, Any] | list[Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Mapping[str, Any] | list[Any] | str | bytes) -> str:
    if isinstance(value, (dict, list)):
        payload = canonical_json(value).encode()
    elif isinstance(value, str):
        payload = value.encode()
    else:
        payload = value
    return hashlib.sha256(payload).hexdigest()


def canonicalize_url(value: str) -> str:
    value = value.strip().rstrip(".,;:!?)\"]}")
    parts = urlsplit(value)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return value
    host = parts.hostname.casefold()
    port = parts.port
    netloc = host
    default_port = (parts.scheme.lower() == "http" and port == 80) or (
        parts.scheme.lower() == "https" and port == 443
    )
    if port and not default_port:
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path).rstrip("/") or "/"
    query = urlencode(
        sorted(
            (key, item)
            for key, item in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
        )
    )
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def extract_urls(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        for match in _URL_RE.finditer(value):
            try:
                found.append(canonicalize_url(match.group(0)))
            except ValueError:
                # Free text and provider metadata can contain URL-shaped typos.
                # They are not evidence locators and must not block the enclosing
                # artifact from being preserved.
                continue
    elif isinstance(value, Mapping):
        for child in value.values():
            found.extend(extract_urls(child))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        for child in value:
            found.extend(extract_urls(child))
    return list(dict.fromkeys(found))


def extract_url_mentions(value: str) -> list[tuple[str, str]]:
    """Return canonical URLs with deterministic character-span evidence locators."""

    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for match in _URL_RE.finditer(value):
        try:
            url = canonicalize_url(match.group(0))
        except ValueError:
            continue
        if url in seen:
            continue
        seen.add(url)
        found.append((url, f"text:{match.start()}-{match.end()}"))
    return found


def infer_url_relation(value: str, locator: str) -> str:
    """Classify a URL mention from local prose without using model vocabularies."""

    match = _TEXT_LOCATOR_RE.fullmatch(locator)
    if match is None:
        raise ValueError("URL mention locator must use text:start-end")
    start, end = (int(item) for item in match.groups())
    if start < 0 or end <= start or end > len(value):
        raise ValueError("URL mention locator is outside the supplied text")
    before = value[max(0, start - 240) : start]
    after = value[end : min(len(value), end + 120)]
    if _OFFICIAL_IMPLEMENTATION_RE.search(before):
        return "official_implementation"
    if _IMPLEMENTATION_BEFORE_RE.search(before) or _IMPLEMENTATION_AFTER_RE.search(after):
        return "implementation"
    if _WEIGHTS_RE.search(before):
        return "weights"
    if _MODEL_CARD_RE.search(before):
        return "model_card"
    return "embedded"


def identifier_from_url(value: str) -> Identifier | None:
    url = canonicalize_url(value)
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold()
    path = parts.path.strip("/")
    segments = path.split("/") if path else []

    if (
        host in {"huggingface.co", "www.huggingface.co"}
        and len(segments) >= 2
        and segments[0] not in {"datasets", "spaces", "papers", "docs", "blog"}
    ):
        return Identifier("huggingface:model", "/".join(segments[:2]))
    if host in {"github.com", "www.github.com"} and len(segments) >= 2:
        return Identifier("github:repository", f"{segments[0]}/{segments[1].removesuffix('.git')}")
    if host in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        match = _ARXIV_RE.match(path)
        if match:
            return Identifier("arxiv", re.sub(r"v\d+$", "", match.group(1)))
    if host in {"doi.org", "dx.doi.org"} and path:
        return Identifier("doi", path.casefold())
    return None
