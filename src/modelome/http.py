from __future__ import annotations

import ipaddress
import json
import os
import socket
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_CROSS_ORIGIN_SAFE_HEADERS = {
    "accept",
    "accept-encoding",
    "accept-language",
    # A byte range conveys no credential or source-private validator. Retaining
    # it across a public resolver-to-CDN redirect lets immutable large archives
    # be read in bounded pieces instead of forcing a whole-object download.
    "range",
    "user-agent",
}
_GITHUB_COMMIT_CACHE_MAX_ENTRIES = 512


class HttpFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str

    def json(self) -> Any:
        return json.loads(self.body)

    def text(self, encoding: str = "utf-8") -> str:
        return self.body.decode(encoding, errors="replace")


class HttpClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        attempts: int = 4,
        max_response_bytes: int = 64 * 1024 * 1024,
        user_agent: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        contact = os.environ.get("MODELOME_CONTACT_EMAIL", "").strip()
        default_agent = "modelome/0.1"
        if contact:
            default_agent = f"{default_agent} (mailto:{contact})"
        self.timeout = timeout
        self.attempts = attempts
        self.max_response_bytes = max_response_bytes
        self.user_agent = user_agent or default_agent
        self._sleep = sleep
        self._opener = build_opener(_SafeRedirectHandler(_require_public_redirect))
        self._github_commit_cache: dict[tuple[str, str, str], HttpResponse] = {}
        self._github_commit_cache_lock = threading.Lock()

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
        redirect_validator: Callable[[str], None] | None = None,
    ) -> HttpResponse:
        request_url = _with_params(url, params or {})
        request_headers = {"Accept": "application/json", "User-Agent": self.user_agent}
        request_headers.update(headers or {})
        cache_key = (
            _github_commit_cache_key(request_url, request_headers)
            if redirect_validator is None
            else None
        )
        if cache_key is None:
            return self._get_uncached(request_url, request_headers, redirect_validator)

        # Serialize only matching commit lookups. This prevents simultaneous
        # adapters from issuing the same anonymous API request before the first
        # response reaches the in-memory cache.
        with self._github_commit_cache_lock:
            cached = self._github_commit_cache.get(cache_key)
            if cached is not None:
                return cached
            response = self._get_uncached(request_url, request_headers, None)
            if response.status == 200 and _origin(response.url) == _origin(request_url):
                if len(self._github_commit_cache) >= _GITHUB_COMMIT_CACHE_MAX_ENTRIES:
                    self._github_commit_cache.pop(next(iter(self._github_commit_cache)))
                self._github_commit_cache[cache_key] = response
            return response

    def clear_github_commit_cache(self) -> None:
        """Discard successful commit lookups, for example at a sync-run boundary."""

        with self._github_commit_cache_lock:
            self._github_commit_cache.clear()

    def _get_uncached(
        self,
        request_url: str,
        request_headers: Mapping[str, str],
        redirect_validator: Callable[[str], None] | None,
    ) -> HttpResponse:
        display_url = _redacted_url(request_url)

        last_error: Exception | None = None
        opener = self._opener
        if redirect_validator is not None:
            def validate_redirect(target: str) -> None:
                _require_public_redirect(target)
                redirect_validator(target)

            opener = build_opener(_SafeRedirectHandler(validate_redirect))
        for attempt in range(self.attempts):
            request = Request(request_url, headers=request_headers, method="GET")
            try:
                with opener.open(request, timeout=self.timeout) as response:  # noqa: S310
                    body = response.read(self.max_response_bytes + 1)
                    if len(body) > self.max_response_bytes:
                        raise HttpFailure(
                            f"GET {display_url} exceeded {self.max_response_bytes} bytes"
                        )
                    return HttpResponse(
                        status=response.status,
                        headers={key.casefold(): value for key, value in response.headers.items()},
                        body=body,
                        url=_redacted_url(response.url),
                    )
            except HTTPError as error:
                if error.code == 304:
                    return HttpResponse(
                        status=304,
                        headers={key.casefold(): value for key, value in error.headers.items()},
                        body=b"",
                        url=display_url,
                    )
                last_error = error
                if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                    break
                delay = _retry_delay(error.headers, attempt)
            except (TimeoutError, URLError) as error:
                last_error = error
                delay = min(2**attempt, 30)
            if attempt + 1 < self.attempts:
                self._sleep(delay)
        raise HttpFailure(f"GET {display_url} failed: {last_error}") from last_error


def _github_commit_cache_key(
    url: str,
    headers: Mapping[str, str],
) -> tuple[str, str, str] | None:
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or (parts.hostname or "").casefold() != "api.github.com"
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        return None
    path_parts = parts.path.split("/")
    if (
        len(path_parts) != 6
        or path_parts[0] != ""
        or path_parts[1] != "repos"
        or not path_parts[2]
        or not path_parts[3]
        or path_parts[4] != "commits"
        or not all(path_parts[5:])
    ):
        return None
    normalized_headers = {key.casefold(): value for key, value in headers.items()}
    safe_headers = {"accept", "user-agent", "x-github-api-version"}
    if set(normalized_headers) - safe_headers:
        return None
    if "if-none-match" in normalized_headers or "if-modified-since" in normalized_headers:
        return None
    return (
        url,
        normalized_headers.get("accept", ""),
        "\n".join(
            (
                normalized_headers.get("user-agent", ""),
                normalized_headers.get("x-github-api-version", ""),
            )
        ),
    )


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Never forward credentials or source validators to another origin."""

    def __init__(self, validator: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self.validator = validator

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> Request | None:
        if self.validator is not None:
            self.validator(newurl)
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None or _origin(req.full_url) == _origin(newurl):
            return redirected
        for name, _ in redirected.header_items():
            if name.casefold() not in _CROSS_ORIGIN_SAFE_HEADERS:
                redirected.remove_header(name)
        return redirected


def _with_params(url: str, params: Mapping[str, str | int]) -> str:
    if not params:
        return url
    parts = urlsplit(url)
    query = "&".join(item for item in (parts.query, urlencode(params)) if item)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _redacted_url(url: str) -> str:
    parts = urlsplit(url)
    sensitive = {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "key",
        "secret",
        "sig",
        "signature",
        "token",
    }
    query = urlencode(
        [
            (key, "REDACTED" if key.casefold() in sensitive else value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ]
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _origin(url: str) -> tuple[str, str, int | None]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port
    if port is None:
        port = 443 if scheme == "https" else 80 if scheme == "http" else None
    return scheme, (parts.hostname or "").casefold(), port


def _require_public_redirect(url: str) -> None:
    """Reject redirects that could make a public fetch reach a private network."""

    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").casefold().rstrip(".")
        if (
            parts.scheme.casefold() not in {"http", "https"}
            or not host
            or parts.username is not None
            or parts.password is not None
            or host == "localhost"
            or host.endswith(".localhost")
        ):
            raise ValueError
        try:
            addresses = (ipaddress.ip_address(host),)
        except ValueError:
            addresses = tuple(
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            )
        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError
    except (OSError, ValueError):
        raise HttpFailure("redirect target is not a public HTTP(S) endpoint") from None


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return min(float(retry_after), 120.0)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(retry_after).timestamp() - time.time())
                return max(0.0, min(seconds, 120.0))
            except (TypeError, ValueError):
                pass
    return min(2**attempt, 30)
