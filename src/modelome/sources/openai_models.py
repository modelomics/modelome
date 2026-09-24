"""Authenticated, public-only observations from OpenAI's Models API.

The API's model list is account-scoped.  It can therefore contain a caller's
fine-tuned or otherwise private models alongside OpenAI-owned public offerings.
This adapter admits only configured public owners and deliberately records those
rows as provider-availability evidence, not as a global catalog or a claim that
weights are available.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

Clock = Callable[[], datetime]

_MODEL_ID = re.compile(r"^[^\s\x00-\x1f\x7f]{1,512}$")
_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class OpenAIModelsSourceAdapter:
    """Record public OpenAI-owned models visible through ``GET /v1/models``.

    A caller's API key is used only to list its accessible models.  Rows owned by
    another organization are ignored before persistence so a normal source run
    cannot turn private fine-tunes into registry observations.  The source is
    intentionally re-read every run: the response is an account availability
    view, not an immutable provider snapshot and never authorizes tombstones.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only configured public-owner rows returned by an operator's "
        "authenticated OpenAI Models API response at its observation time. It "
        "excludes private, organization-owned, unavailable, deleted, and "
        "never-API-listed models; it does not enumerate model artifacts, "
        "weights, papers, or a historical provider catalog."
    )

    def __init__(
        self,
        *,
        name: str = "openai-models",
        url: str = "https://api.openai.com/v1/models",
        token: str,
        public_owners: Sequence[str] = ("openai", "system"),
        max_response_bytes: int = 4 * 1024 * 1024,
        max_models: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, "catalog URL")
        self._token = _required_text(token, "OpenAI API key")
        self.public_owners = _owners(public_owners, self.name)
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes", self.name
        )
        self.max_models = _positive_int(max_models, "max_models", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openai-models-v1",
                "url": self.url,
                "public_owners": self.public_owners,
                "max_response_bytes": self.max_response_bytes,
                "max_models": self.max_models,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        """Fetch a fresh account-availability observation without retaining a key."""

        try:
            response: HttpResponse = self.client.get(
                self.url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
        except Exception as error:
            if self._token in str(error):
                raise RuntimeError(f"{self.name}: model-list request failed") from None
            raise
        if response.status != 200:
            raise ValueError(f"{self.name}: model-list endpoint returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: model-list response exceeds {self.max_response_bytes} bytes"
            )
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: model-list response is not an object")
        rows = payload.get("data")
        if not _sequence(rows):
            raise ValueError(f"{self.name}: model-list response data is not an array")
        if len(rows) > self.max_models:
            raise ValueError(f"{self.name}: model-list response exceeds {self.max_models} rows")

        records: list[SourceRecord] = []
        seen_ids: set[str] = set()
        skipped_nonpublic = 0
        observed_at = _isoformat(self.clock())
        for index, item in enumerate(rows):
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: model-list row {index} is not an object")
            model_id = _required_text(item.get("id"), f"model-list row {index} id")
            owner = _required_text(item.get("owned_by"), f"model-list row {index} owner")
            if not _MODEL_ID.fullmatch(model_id):
                raise ValueError(f"{self.name}: model-list row {index} has invalid model ID")
            if not _OWNER.fullmatch(owner):
                raise ValueError(f"{self.name}: model-list row {index} has invalid owner")
            if model_id in seen_ids:
                raise ValueError(f"{self.name}: model-list response contains duplicate model IDs")
            seen_ids.add(model_id)
            if owner.casefold() not in self.public_owners:
                skipped_nonpublic += 1
                continue
            records.append(self._record(item, model_id, owner, observed_at, response.url))

        next_state = {
            "checked_at": observed_at,
            "returned_model_count": len(rows),
            "public_model_count": len(records),
            "skipped_nonpublic_model_count": skipped_nonpublic,
            "catalog_sha256": content_hash(response.body),
        }
        if etag := _header(response.headers, "etag"):
            next_state["catalog_etag"] = etag
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )

    def _record(
        self,
        item: Mapping[str, Any],
        model_id: str,
        owner: str,
        observed_at: str,
        response_url: str,
    ) -> SourceRecord:
        identifier = Identifier("openai:model", model_id)
        local_id = f"model:{model_id}"
        created_at = _unix_timestamp(item.get("created"), self.name, model_id)
        endpoint_url = canonicalize_url(
            f"{self.url.rstrip('/')}/{quote(model_id, safe='')}"
        )
        model = ModelHint(
            local_id=local_id,
            name=model_id,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator="$.data[].id",
        )
        return SourceRecord(
            source_record_id=f"model:{model_id}",
            kind=ArtifactKind.PROVIDER_PAGE,
            canonical_url=endpoint_url,
            title=model_id,
            raw={
                "endpoint": self.url,
                "response_url": response_url or self.url,
                "observed_at": observed_at,
                "model": {
                    "id": model_id,
                    "object": _optional_text(item.get("object")),
                    "owned_by": owner,
                    "created": item.get("created"),
                    "shutdown_date": _optional_text(item.get("shutdown_date")),
                },
            },
            text="\n".join(
                part
                for part in (
                    f"owner: {owner}",
                    f"created: {created_at}" if created_at else "",
                    (
                        "shutdown date: " + _optional_text(item.get("shutdown_date"))
                        if _optional_text(item.get("shutdown_date"))
                        else ""
                    ),
                )
                if part
            ),
            published_at=created_at,
            modified_at=observed_at,
            identifiers=(identifier,),
            links=(
                Link(endpoint_url, relation="provider_api_resource", crawl=False),
                Link(
                    "https://developers.openai.com/api/docs/models",
                    relation="provider_documentation",
                    crawl=False,
                ),
            ),
            models=(model,),
        )


def _required_text(value: Any, label: str) -> str:
    result = _optional_text(value)
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _optional_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be positive") from error
    if result < 1:
        raise ValueError(f"{source}: {label} must be positive")
    return result


def _owners(value: Sequence[str], source: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{source}: public_owners must be an array")
    owners = tuple(sorted({_required_text(item, "public owner").casefold() for item in value}))
    if not owners or any(not _OWNER.fullmatch(owner) for owner in owners):
        raise ValueError(f"{source}: public_owners contains an invalid owner")
    return owners


def _web_url(value: str, label: str) -> str:
    result = _required_text(value, label)
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials or query text")
    return canonicalize_url(result)


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _unix_timestamp(value: Any, source: str, model_id: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{source}: model {model_id!r} has invalid created timestamp")
    if value < 0:
        raise ValueError(f"{source}: model {model_id!r} has invalid created timestamp")
    try:
        return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError) as error:
        raise ValueError(f"{source}: model {model_id!r} has invalid created timestamp") from error


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _header(headers: Mapping[str, str], name: str) -> str:
    target = name.casefold()
    return next(
        (str(value).strip() for key, value in headers.items() if key.casefold() == target),
        "",
    )


__all__ = ["OpenAIModelsSourceAdapter"]
