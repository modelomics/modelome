"""Authenticated ingestion of Replicate's public, displayworthy model catalog.

Replicate's documented list endpoint is cursor-paginated and intentionally exposes
only public models that meet its displayworthy criteria.  This adapter retains that
useful hosted-model plane without claiming it is a census of every model ever created
on Replicate, and never fetches model outputs or artifact bytes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, quote, urljoin, urlsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, extract_urls

_SENSITIVE_QUERY_KEYS = {
    "accesstoken",
    "apikey",
    "auth",
    "authorization",
    "key",
    "secret",
    "sig",
    "signature",
    "token",
}


class ReplicateModelsSourceAdapter:
    """Follow Replicate's documented cursor URLs for public model cards.

    The endpoint requires a token but returns public models only.  A cursor scan has
    no provider-reported total and model eligibility can change while it runs, so this
    deliberately remains an availability observation rather than an authoritative
    tombstone-producing snapshot.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public Replicate models returned by its authenticated List Models API. "
        "Replicate documents that the API returns only displayworthy public models, so "
        "private, unpublished, and non-displayworthy public models are outside this "
        "source. Current versions for enumerated models are fetched from the public "
        "cursor-paginated model-versions endpoint; downloadable weight bytes are not "
        "fetched."
    )

    def __init__(
        self,
        *,
        name: str = "replicate-models",
        url: str = "https://api.replicate.com/v1/models",
        model_page_base_url: str = "https://replicate.com",
        token: str,
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.model_page_base_url = _web_url(
            model_page_base_url,
            self.name,
            "model-page base URL",
        ).rstrip("/")
        self._token = _required_text(token, "Replicate API token")
        self.max_response_bytes = _positive_int(
            max_response_bytes,
            "max_response_bytes",
            self.name,
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "replicate-models-v2",
                "url": self.url,
                "model_page_base_url": self.model_page_base_url,
                "max_response_bytes": self.max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        queue = _version_queue(state.get("version_queue", []), self.name)
        if queue:
            return self._fetch_version_page(state, queue)

        next_url = _text(state.get("next_url"))
        catalog_complete = state.get("catalog_complete") is True
        if catalog_complete:
            return SourcePage(records=(), next_state={}, complete=True)
        request_url = self._safe_next_url(next_url, self.url) if next_url else self.url
        try:
            response: HttpResponse = self.client.get(
                request_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
        except Exception as error:
            if self._token in str(error):
                raise RuntimeError(f"{self.name}: catalog request failed") from None
            raise
        if response.status != 200:
            raise ValueError(f"{self.name}: catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: catalog response is not a JSON object")
        items = payload.get("results")
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: catalog results is not a JSON list")

        records = tuple(self._record(item, index) for index, item in enumerate(items))
        version_queue = []
        for item in items:
            if isinstance(item, Mapping):
                owner = _text(item.get("owner"))
                model_name = _text(item.get("name"))
                if owner and model_name:
                    version_queue.append({"model_id": f"{owner}/{model_name}"})
        raw_items_seen = (_state_count(state, "raw_items_seen") if next_url else 0) + len(items)
        raw_next = _text(payload.get("next"))
        safe_next = None
        if raw_next:
            safe_next = self._safe_next_url(raw_next, response.url or request_url)
            if safe_next == request_url:
                raise ValueError(f"{self.name}: pagination URL did not advance")
        next_state: dict[str, Any] = {}
        if safe_next:
            next_state["next_url"] = safe_next
            next_state["raw_items_seen"] = raw_items_seen
        elif raw_items_seen:
            next_state["raw_items_seen"] = raw_items_seen
        if version_queue:
            next_state["version_queue"] = version_queue
        if not safe_next:
            next_state["catalog_complete"] = True
        complete = not version_queue and not safe_next
        if complete:
            next_state = {}
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
        )

    def _fetch_version_page(
        self,
        state: Mapping[str, Any],
        queue: list[dict[str, str]],
    ) -> SourcePage:
        current = queue[0]
        model_id = current["model_id"]
        owner, model_name = model_id.split("/", maxsplit=1)
        versions_url = (
            f"{self.url.rstrip('/')}/{quote(owner, safe='')}/"
            f"{quote(model_name, safe='')}/versions"
        )
        queued_url = current.get("next_url")
        request_url = self._safe_next_url(queued_url, versions_url) if queued_url else versions_url
        try:
            response: HttpResponse = self.client.get(
                request_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
        except Exception as error:
            if self._token in str(error):
                raise RuntimeError(f"{self.name}: versions request failed") from None
            raise
        if response.status != 200:
            raise ValueError(f"{self.name}: versions returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: versions response exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping) or not _is_sequence(payload.get("results")):
            raise ValueError(f"{self.name}: versions response has no results list")

        records = tuple(
            self._version_record(model_id, item, index)
            for index, item in enumerate(payload["results"])
        )
        raw_next = _text(payload.get("next"))
        if raw_next:
            safe_next = self._safe_next_url(raw_next, response.url or request_url)
            if safe_next == request_url:
                raise ValueError(f"{self.name}: versions pagination URL did not advance")
            current["next_url"] = safe_next
        else:
            queue.pop(0)

        next_state = dict(state)
        if queue:
            next_state["version_queue"] = queue
        else:
            next_state.pop("version_queue", None)
        complete = not queue and next_state.get("catalog_complete") is True
        if complete:
            next_state = {}
        return SourcePage(records=records, next_state=next_state, complete=complete)

    def _version_record(self, model_id: str, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping) or not _text(item.get("id")):
            raise ValueError(f"{self.name}: version {index} is missing its id")
        _, model_name = model_id.split("/", maxsplit=1)
        version_id = _text(item.get("id"))
        model_local_id = f"{model_id}#model"
        model_page = canonicalize_url(
            f"{self.model_page_base_url}/{quote(model_id, safe='/._-')}"
        )
        release = ReleaseHint(
            local_id=f"{model_local_id}#version:{version_id}",
            model_local_id=model_local_id,
            version=version_id,
            identifiers=(Identifier("replicate:model-version", version_id),),
            released_at=_text(item.get("created_at")) or None,
            metadata={
                key: value
                for key in ("cog_version", "openapi_schema")
                if (value := item.get(key)) is not None
            },
            locator="$.id",
        )
        model = ModelHint(
            local_id=model_local_id,
            name=model_id,
            identifiers=(Identifier("replicate:model", model_id),),
            aliases=(model_name,),
            status=ModelStatus.RELEASED,
            locator="$.model_id",
        )
        return SourceRecord(
            source_record_id=f"{model_id}@{version_id}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_page,
            title=f"{model_id} version {version_id}",
            raw=_redact_secret(dict(item), self._token),
            modified_at=release.released_at,
            identifiers=(Identifier("replicate:model", model_id),),
            links=(Link(model_page, relation="model_page", locator="$.model_id", crawl=False),),
            models=(model,),
            releases=(release,),
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"{self.name}: catalog item {index} is not an object")
        owner = _required_text(item.get("owner"), f"catalog item {index} owner")
        name = _required_text(item.get("name"), f"catalog item {index} name")
        model_id = f"{owner}/{name}"
        visibility = _text(item.get("visibility"))
        if visibility and visibility.casefold() != "public":
            raise ValueError(f"{self.name}: catalog returned a non-public model")
        model_page = _optional_web_url(item.get("url"), self.model_page_base_url)
        if not model_page:
            model_page = canonicalize_url(
                f"{self.model_page_base_url}/{quote(model_id, safe='/._-')}"
            )
        identifier = Identifier("replicate:model", model_id)
        local_id = f"{model_id}#model"
        model = ModelHint(
            local_id=local_id,
            name=model_id,
            identifiers=(identifier,),
            aliases=(name,) if name != model_id else (),
            status=ModelStatus.RELEASED,
            locator="$.owner,$.name",
        )
        description = _text(item.get("description"))
        readme = _text(item.get("readme"))
        text_parts = [value for value in (description, readme) if value]
        links: list[Link] = [
            Link(model_page, relation="model_page", locator="$.url", crawl=False),
        ]
        for locator, value in (("$.description", description), ("$.readme", readme)):
            links.extend(
                Link(url, relation="documentation_reference", locator=locator)
                for url in extract_urls(value)
            )
        for field, relation, crawl in (
            ("github_url", "source_repository", True),
            ("paper_url", "paper", True),
            ("weights_url", "weights", False),
            ("license_url", "license", False),
        ):
            if url := _optional_web_url(item.get(field), model_page):
                links.append(Link(url, relation=relation, locator=f"$.{field}", crawl=crawl))

        latest_version = item.get("latest_version")
        if latest_version is not None and not isinstance(latest_version, Mapping):
            raise ValueError(f"{self.name}: model {model_id!r} latest_version is not an object")
        releases = ()
        modified_at = _text(item.get("updated_at")) or None
        if isinstance(latest_version, Mapping):
            release = self._latest_release(latest_version, local_id, model_id)
            if release is not None:
                releases = (release,)
                modified_at = modified_at or release.released_at

        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=model_page,
            title=model_id,
            raw=_redact_secret(dict(item), self._token),
            text="\n".join(text_parts),
            published_at=_text(item.get("created_at")) or None,
            modified_at=modified_at,
            identifiers=(identifier,),
            links=_unique_links(links),
            models=(model,),
            releases=releases,
        )

    def _latest_release(
        self,
        value: Mapping[str, Any],
        model_local_id: str,
        model_id: str,
    ) -> ReleaseHint | None:
        version_id = _text(value.get("id"))
        if not version_id:
            return None
        metadata = {
            key: text
            for key in ("cog_version", "openapi_schema")
            if (text := value.get(key)) is not None
        }
        return ReleaseHint(
            local_id=f"{model_local_id}#version:{version_id}",
            model_local_id=model_local_id,
            version=version_id,
            identifiers=(Identifier("replicate:model-version", version_id),),
            released_at=_text(value.get("created_at")) or None,
            metadata=metadata,
            locator="$.latest_version.id",
        )

    def _safe_next_url(self, value: str, base_url: str) -> str:
        candidate = _web_url(urljoin(base_url, value), self.name, "pagination URL")
        if _origin(candidate) != _origin(self.url):
            raise ValueError(f"{self.name}: pagination URL must remain on the catalog origin")
        for key, _ in parse_qsl(urlsplit(candidate).query, keep_blank_values=True):
            normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
            if normalized in _SENSITIVE_QUERY_KEYS:
                raise ValueError(
                    f"{self.name}: pagination URL contains a sensitive query parameter"
                )
        return candidate



def _redact_secret(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, Mapping):
        return {key: _redact_secret(item, secret) for key, item in value.items()}
    if _is_sequence(value):
        return [_redact_secret(item, secret) for item in value]
    return value


def _unique_links(values: Iterable[Link]) -> tuple[Link, ...]:
    return tuple(
        sorted(
            set(values),
            key=lambda link: (link.url, link.relation, link.locator or "", link.crawl),
        )
    )


def _optional_web_url(value: Any, base_url: str) -> str | None:
    text = _text(value)
    if not text:
        return None
    result = canonicalize_url(urljoin(base_url, text))
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Replicate model link is not an HTTP(S) URL")
    return result


def _web_url(value: str, source: str, label: str) -> str:
    result = canonicalize_url(_required_text(value, label))
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: {label} is not an HTTP(S) URL")
    return result


def _origin(value: str) -> tuple[str, str, int | None]:
    parts = urlsplit(value)
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.scheme.casefold(), (parts.hostname or "").casefold(), port


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} is required")
    if any(character in result for character in "\r\n\x00"):
        raise ValueError(f"{label} contains control characters")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _state_count(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key, 0)
    if isinstance(value, bool):
        raise ValueError(f"Replicate sync state {key} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Replicate sync state {key} must be a non-negative integer"
        ) from error
    if result < 0 or str(result) != str(value).strip():
        raise ValueError(f"Replicate sync state {key} must be a non-negative integer")
    return result


def _version_queue(value: Any, source: str) -> list[dict[str, str]]:
    if not _is_sequence(value):
        raise ValueError(f"{source}: version checkpoint queue must be a list")
    result: list[dict[str, str]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{source}: version checkpoint {index} must be an object")
        model_id = _required_text(entry.get("model_id"), "checkpoint model id")
        if model_id.count("/") != 1 or any(character in model_id for character in "?#\\"):
            raise ValueError(f"{source}: invalid model id in version checkpoint")
        queued = {"model_id": model_id}
        if next_url := _text(entry.get("next_url")):
            queued["next_url"] = next_url
        result.append(queued)
    return result


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be positive") from error
    if result < 1 or str(result) != str(value).strip():
        raise ValueError(f"{source}: {label} must be positive")
    return result


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["ReplicateModelsSourceAdapter"]
