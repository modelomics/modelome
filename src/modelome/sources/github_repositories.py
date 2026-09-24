"""Direct, cursor-safe enumeration of GitHub's public repository catalog."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, Link, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash, identifier_from_url

Clock = Callable[[], datetime]
_LINK_RE = re.compile(r'<([^>]+)>\s*((?:;\s*[^,]+)*)')
_REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;\s,]+))', re.IGNORECASE)
_MAX_PAGE_SIZE = 100
_GITHUB_API_VERSION = "2026-03-10"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GitHubPublicRepositoriesSourceAdapter:
    """Enumerate public GitHub repositories directly through ``GET /repositories``.

    GitHub documents this endpoint as its created-order public repository listing,
    whose cursor is exclusively the numeric ``since`` repository ID supplied in
    the response's RFC 8288 ``Link`` header.  This is a repository census plane,
    not a search: it contains no topic, language, owner, activity, or model-term
    selection.  It therefore complements GH Archive's activity stream and
    Software Heritage's historical-origin snapshot.

    Each catalog row stores GitHub's immutable numeric ID and adds the canonical
    HTML repository URL to the ordinary frontier.  The existing GitHub fetcher
    then retrieves current metadata and the README before a repository can
    document a neural-model claim.  The catalog response itself never asserts
    that a repository contains a model.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public repositories returned after the durable GitHub numeric-ID "
        "cursor. Private, deleted, unavailable, and not-yet-reached repositories "
        "are outside the observed catalog. The endpoint has no deletion feed, so "
        "it is not proof that a previously observed repository remains public."
    )

    def __init__(
        self,
        *,
        name: str = "github-public-repositories",
        url: str = "https://api.github.com/repositories",
        page_size: int = _MAX_PAGE_SIZE,
        initial_since: int = 0,
        max_repository_id: int | None = None,
        token: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _catalog_url(url, self.name)
        self.page_size = _positive_int(page_size, "page_size", self.name)
        if self.page_size > _MAX_PAGE_SIZE:
            raise ValueError(f"{self.name}: page_size must not exceed {_MAX_PAGE_SIZE}")
        self.initial_since = _nonnegative_int(initial_since, "initial_since", self.name)
        self.max_repository_id = (
            None
            if max_repository_id is None
            else _positive_int(max_repository_id, "max_repository_id", self.name)
        )
        if (
            self.max_repository_id is not None
            and self.max_repository_id < self.initial_since
        ):
            raise ValueError(f"{self.name}: max_repository_id must be >= initial_since")
        self.token = _optional_text(token)
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "github-public-repositories-v1",
                "url": self.url,
                "page_size": self.page_size,
                "initial_since": self.initial_since,
                "max_repository_id": self.max_repository_id,
                "authenticated": bool(self.token),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError(f"{self.name}: checkpoint state must be a mapping")
        next_url = _optional_text(state.get("next_url"))
        last_repository_id = _state_id(
            state.get("last_repository_id"),
            "last_repository_id",
            self.name,
        )
        if (
            self.max_repository_id is not None
            and last_repository_id is not None
            and last_repository_id >= self.max_repository_id
        ):
            return self._completed_page(last_repository_id, ())
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if next_url:
            request_url = self._safe_next_url(next_url, self.url)
            response: HttpResponse = self.client.get(request_url, headers=headers)
        else:
            request_url = self.url
            params: dict[str, int] = {"per_page": self.page_size}
            since = self.initial_since if last_repository_id is None else last_repository_id
            if since:
                params["since"] = since
            response = self.client.get(request_url, params=params, headers=headers)
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")

        payload = response.json()
        if not _is_sequence(payload):
            raise ValueError(f"{self.name}: catalog response is not a JSON array")
        if len(payload) > self.page_size:
            raise ValueError(
                f"{self.name}: response contained {len(payload)} repositories, "
                f"above page_size {self.page_size}"
            )

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        observed_ids: list[int] = []
        reached_range_end = False
        for index, item in enumerate(payload):
            try:
                record, repository_id = self._record(item, index)
                if observed_ids and repository_id <= observed_ids[-1]:
                    raise ValueError("repository IDs are not strictly increasing")
                if last_repository_id is not None and repository_id <= last_repository_id:
                    raise ValueError("repository ID did not advance past checkpoint")
                if (
                    self.max_repository_id is not None
                    and repository_id > self.max_repository_id
                ):
                    reached_range_end = True
                    break
                records.append(record)
                observed_ids.append(repository_id)
            except (TypeError, ValueError) as error:
                raw = dict(item) if isinstance(item, Mapping) else {"value": repr(item)[:1_000]}
                record_id = _identifier_text(item.get("id")) if isinstance(item, Mapping) else ""
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            f"github-repository-id:{record_id}"
                            if record_id
                            else f"{self.name}:malformed:{content_hash(raw)[:32]}"
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"index": index, "raw": raw},
                    )
                )

        if issues:
            retry_state = dict(state)
            return SourcePage(
                records=tuple(records),
                next_state=retry_state,
                complete=False,
                upstream_count=None,
                issues=tuple(issues),
                retry_state=retry_state,
            )

        if reached_range_end:
            final_id = observed_ids[-1] if observed_ids else last_repository_id
            return self._completed_page(final_id, records)

        link_next = _link_relation(_header(response.headers, "link"), "next")
        if link_next:
            if not observed_ids:
                raise ValueError(f"{self.name}: empty page supplied a next cursor")
            safe_next = self._safe_next_url(link_next, response.url or request_url)
            self._validate_next_cursor(safe_next, observed_ids[-1])
            next_state = {
                "next_url": safe_next,
                "last_repository_id": observed_ids[-1],
                "raw_items_seen": _state_count(state, "raw_items_seen") + len(records),
            }
            return SourcePage(
                records=tuple(records),
                next_state=next_state,
                complete=False,
                upstream_count=None,
            )
        if len(payload) == self.page_size:
            raise ValueError(f"{self.name}: full page omitted GitHub's next cursor")

        final_id = observed_ids[-1] if observed_ids else last_repository_id
        return self._completed_page(final_id, records)

    def _completed_page(
        self,
        final_id: int | None,
        records: Sequence[SourceRecord],
    ) -> SourcePage:
        next_state: dict[str, Any] = {"completed_at": _isoformat(self.clock())}
        if final_id is not None:
            next_state["last_repository_id"] = final_id
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
        )

    def _record(self, item: Any, index: int) -> tuple[SourceRecord, int]:
        if not isinstance(item, Mapping):
            raise ValueError(f"catalog item {index} is not an object")
        if item.get("private") is True:
            raise ValueError("public repository catalog returned a private repository")
        repository_id = _positive_int(item.get("id"), f"catalog item {index} id", self.name)
        full_name = _repository_name(item.get("full_name"), f"catalog item {index} full_name")
        repository_url = _repository_url(item.get("html_url"), full_name, self.name)
        api_url = _repository_api_url(item.get("url"), full_name, self.url, self.name)
        locator = f"repositories[{index}]"
        return (
            SourceRecord(
                source_record_id=f"github-repository-id:{repository_id}",
                kind=ArtifactKind.CODE_REPOSITORY,
                canonical_url=api_url,
                title=full_name,
                raw={"repository": dict(item), "locator": locator},
                published_at=_optional_text(item.get("created_at")) or None,
                modified_at=_optional_text(item.get("updated_at")) or None,
                identifiers=(
                    Identifier("github:repository-id", str(repository_id)),
                    Identifier("github:repository", full_name),
                ),
                links=(
                    Link(
                        repository_url,
                        relation="repository_catalog",
                        locator=f"{locator}.html_url",
                    ),
                ),
            ),
            repository_id,
        )

    def _safe_next_url(self, value: str, base_url: str) -> str:
        candidate = canonicalize_url(urljoin(base_url, value))
        candidate_parts = urlsplit(candidate)
        expected_parts = urlsplit(self.url)
        if (
            candidate_parts.scheme.casefold() != expected_parts.scheme.casefold()
            or (candidate_parts.hostname or "").casefold()
            != (expected_parts.hostname or "").casefold()
            or candidate_parts.path.rstrip("/") != expected_parts.path.rstrip("/")
            or candidate_parts.username is not None
            or candidate_parts.password is not None
            or candidate_parts.fragment
        ):
            raise ValueError(f"{self.name}: GitHub pagination URL escaped the catalog endpoint")
        return candidate

    def _validate_next_cursor(self, url: str, last_repository_id: int) -> None:
        query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
        since = _positive_int(query.get("since"), "GitHub next cursor since", self.name)
        if since != last_repository_id:
            raise ValueError(
                f"{self.name}: GitHub next cursor since {since} does not match "
                f"the final repository ID {last_repository_id}"
            )
        per_page = _positive_int(
            query.get("per_page", self.page_size), "GitHub next cursor per_page", self.name
        )
        if per_page != self.page_size:
            raise ValueError(f"{self.name}: GitHub next cursor changed page_size")
        if set(query) - {"since", "per_page"}:
            raise ValueError(f"{self.name}: GitHub next cursor contains unknown parameters")


def _catalog_url(value: Any, source: str) -> str:
    url = canonicalize_url(_required_text(value, "catalog URL"))
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or (parts.hostname or "").casefold() != "api.github.com"
        or parts.path.rstrip("/") != "/repositories"
        or parts.query
        or parts.fragment
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError(f"{source}: catalog URL must be https://api.github.com/repositories")
    return url


def _repository_name(value: Any, field: str) -> str:
    name = _required_text(value, field)
    parts = name.split("/")
    if len(parts) != 2 or not all(part and part.strip() == part for part in parts):
        raise ValueError(f"{field} must be an owner/name repository")
    return name


def _repository_url(value: Any, full_name: str, source: str) -> str:
    url = canonicalize_url(_required_text(value, "repository HTML URL"))
    identifier = identifier_from_url(url)
    if identifier is None or identifier.namespace != "github:repository":
        raise ValueError(f"{source}: repository HTML URL is not a GitHub repository")
    if identifier.value.casefold() != full_name.casefold():
        raise ValueError(f"{source}: repository HTML URL does not match full_name")
    return url


def _repository_api_url(value: Any, full_name: str, catalog_url: str, source: str) -> str:
    url = canonicalize_url(_required_text(value, "repository API URL"))
    parts = urlsplit(url)
    catalog_parts = urlsplit(catalog_url)
    expected_path = "/repos/" + "/".join(quote(part, safe="") for part in full_name.split("/"))
    if (
        parts.scheme.casefold() != catalog_parts.scheme.casefold()
        or (parts.hostname or "").casefold() != (catalog_parts.hostname or "").casefold()
        or parts.path != expected_path
        or parts.query
        or parts.fragment
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError(f"{source}: repository API URL does not match full_name")
    return url


def _link_relation(value: str, relation: str) -> str | None:
    for target, parameters in _LINK_RE.findall(value):
        match = _REL_RE.search(parameters)
        if match is None:
            continue
        values = (match.group(1) or match.group(2) or "").casefold().split()
        if relation.casefold() in values:
            return target
    return None


def _header(headers: Mapping[str, Any], name: str) -> str:
    for key, value in headers.items():
        if str(key).casefold() == name.casefold():
            return _optional_text(value)
    return ""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def _state_id(value: Any, field: str, source: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field, source)


def _state_count(state: Mapping[str, Any], field: str) -> int:
    value = state.get(field, 0)
    result = _integer_value(value, field)
    if result < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _required_text(value: Any, field: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _identifier_text(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, int):
        return str(value) if value > 0 else ""
    return _optional_text(value)


def _positive_int(value: Any, field: str, source: str) -> int:
    try:
        result = _integer_value(value, field)
    except ValueError as error:
        raise ValueError(f"{source}: {field} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {field} must be a positive integer")
    return result


def _nonnegative_int(value: Any, field: str, source: str) -> int:
    try:
        result = _integer_value(value, field)
    except ValueError as error:
        raise ValueError(f"{source}: {field} must be a nonnegative integer") from error
    if result < 0:
        raise ValueError(f"{source}: {field} must be a nonnegative integer")
    return result


def _integer_value(value: Any, field: str) -> int:
    """Accept JSON integers and decimal cursor strings without lossy coercion."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value.strip())
    raise ValueError(f"{field} must be an integer")


def _isoformat(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["GitHubPublicRepositoriesSourceAdapter"]
