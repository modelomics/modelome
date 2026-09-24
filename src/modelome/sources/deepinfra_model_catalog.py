from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html.parser import HTMLParser
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
from modelome.normalize import canonicalize_url

_ORIGIN = "https://deepinfra.com"
_NEXT_DATA_ID = "__NEXT_DATA__"
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,150}/[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")


class _NextDataParser(HTMLParser):
    def __init__(self, *, max_script_chars: int) -> None:
        super().__init__(convert_charrefs=False)
        self.max_script_chars = max_script_chars
        self.matches: list[str] = []
        self._in_target = False
        self._parts: list[str] = []
        self._script_chars = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "script":
            return
        attributes = {key.casefold(): value or "" for key, value in attrs}
        if attributes.get("id") == _NEXT_DATA_ID:
            if self._in_target:
                raise ValueError("DeepInfra page has nested Next.js data scripts")
            if attributes.get("type") != "application/json":
                raise ValueError("DeepInfra Next.js data script has an unexpected type")
            self._in_target = True
            self._parts = []
            self._script_chars = 0

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._in_target:
            self.matches.append("".join(self._parts))
            self._parts = []
            self._in_target = False

    def handle_data(self, data: str) -> None:
        if not self._in_target:
            return
        self._script_chars += len(data)
        if self._script_chars > self.max_script_chars:
            raise ValueError("DeepInfra Next.js data script exceeds character limit")
        self._parts.append(data)


class DeepInfraModelCatalogAdapter:
    """Read DeepInfra's public model list embedded in the first-party catalog page.

    The SSR payload contains the complete catalog array. The adapter uses its
    exact ``full_name`` serving IDs and makes no reusable-weight claim.
    """

    def __init__(
        self,
        *,
        name: str = "deepinfra-model-catalog",
        url: str = "https://deepinfra.com/models",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_script_chars: int = 8 * 1024 * 1024,
        max_entries: int = 10_000,
    ) -> None:
        if not name.strip():
            raise ValueError("DeepInfra source name must be non-empty")
        if max_response_bytes < 1 or max_script_chars < 1 or max_entries < 1:
            raise ValueError("DeepInfra bounds must be positive")
        self.name = name
        self.url = url
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.max_script_chars = max_script_chars
        self.max_entries = max_entries

    def fetch_page(self, state: Mapping[str, Any] | None = None) -> SourcePage:
        if state:
            raise ValueError(f"{self.name}: this endpoint returns one complete catalog page")
        response: HttpResponse = self.client.get(
            self.url,
            headers={"Accept": "text/html"},
        )
        if not 200 <= response.status < 300:
            raise ValueError(f"{self.name}: model catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model catalog exceeds byte limit")
        final_url = response.url or self.url
        if final_url != self.url:
            raise ValueError(f"{self.name}: model catalog redirected to an unexpected URL")

        parser = _NextDataParser(max_script_chars=self.max_script_chars)
        parser.feed(response.text())
        parser.close()
        if len(parser.matches) != 1:
            raise ValueError(f"{self.name}: expected exactly one Next.js data payload")
        try:
            payload = json.loads(parser.matches[0])
        except json.JSONDecodeError as error:
            raise ValueError(f"{self.name}: malformed Next.js data payload") from error
        page_props = _mapping_at(payload, "props", "pageProps", source=self.name)
        if page_props.get("page") != 1:
            raise ValueError(f"{self.name}: catalog root did not return its first page")
        items = page_props.get("models")
        if not isinstance(items, list) or not items:
            raise ValueError(f"{self.name}: Next.js data has no complete model list")
        if len(items) > self.max_entries:
            raise ValueError(f"{self.name}: model list exceeds {self.max_entries} entries")

        records: list[SourceRecord] = []
        seen: set[str] = set()
        for index, raw_item in enumerate(items):
            item = _as_mapping(raw_item, source=self.name, index=index)
            if item.get("private") is True:
                continue
            if not isinstance(item.get("private"), bool):
                raise ValueError(f"{self.name}: model at index {index} lacks public/private flag")
            endpoint_id = item.get("full_name")
            owner = item.get("owner")
            model_name = item.get("name")
            if (
                not isinstance(endpoint_id, str)
                or _MODEL_ID.fullmatch(endpoint_id) is None
                or not isinstance(owner, str)
                or not isinstance(model_name, str)
                or endpoint_id != f"{owner}/{model_name}"
            ):
                raise ValueError(f"{self.name}: model at index {index} has invalid exact ID")
            if endpoint_id in seen:
                raise ValueError(f"{self.name}: duplicate model ID {endpoint_id!r}")
            seen.add(endpoint_id)
            model_type = item.get("type")
            if not isinstance(model_type, str) or not model_type:
                raise ValueError(f"{self.name}: model {endpoint_id!r} has no type")
            description = item.get("description")
            if not isinstance(description, str):
                raise ValueError(f"{self.name}: model {endpoint_id!r} has no description field")

            item_url = f"{_ORIGIN}/{endpoint_id}"
            metadata = {
                key: item[key]
                for key in (
                    "type",
                    "tags",
                    "deprecated",
                    "replaced_by",
                    "quantization",
                    "max_tokens",
                    "expected",
                    "is_partner",
                )
                if key in item
            }
            records.append(
                SourceRecord(
                    source_record_id=f"deepinfra:model:{endpoint_id}",
                    kind=ArtifactKind.PROVIDER_PAGE,
                    canonical_url=canonicalize_url(item_url),
                    title=endpoint_id,
                    raw={
                        "provider": "deepinfra",
                        "endpoint_id": endpoint_id,
                        "description": description,
                        **metadata,
                    },
                    text=description,
                    identifiers=(Identifier("deepinfra:model", endpoint_id),),
                    links=(Link(item_url, relation="model_documentation", crawl=False),),
                    models=(
                        ModelHint(
                            local_id=endpoint_id,
                            name=endpoint_id,
                            identifiers=(Identifier("deepinfra:model", endpoint_id),),
                            status=ModelStatus.DOCUMENTED,
                            locator=f"next_data:props.pageProps.models[{index}]",
                        ),
                    ),
                )
            )
        return SourcePage(
            records=tuple(records),
            next_state={"catalog_count": len(records)},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=False,
        )


def _mapping_at(value: Any, *keys: str, source: str) -> Mapping[str, Any]:
    current = value
    for key in keys:
        current = _as_mapping(current, source=source, label=key).get(key)
    return _as_mapping(current, source=source, label="pageProps")


def _as_mapping(
    value: Any,
    *,
    source: str,
    index: int | None = None,
    label: str = "model",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        where = f" at index {index}" if index is not None else ""
        raise ValueError(f"{source}: {label}{where} is not an object")
    return value


__all__ = ["DeepInfraModelCatalogAdapter"]
