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
    ModelRelationHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_MISSING = object()
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_QUERY_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")
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
_SENSITIVE_HEADER_NAMES = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}


class JsonCatalogSourceAdapter:
    """Enumerate a provider model catalog using declarative JSON field paths.

    Paths use dotted object keys, optionally prefixed by ``$.``. Numeric path
    segments address array elements. The adapter knows no provider or model
    vocabulary: identifiers, names, timestamps, lineage, and pagination all come
    from configuration.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        provider_namespace: str,
        model_card_url_template: str,
        mapping: Mapping[str, Any],
        client: HttpClient | Any | None = None,
        cursor_param: str = "cursor",
        page_size: int = 100,
        page_size_param: str | None = None,
        auth_header_name: str | None = None,
        auth_query_param: str | None = None,
        auth_token: str | None = None,
        auth_scheme: str = "Bearer",
        static_headers: Mapping[str, str] | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        provider_id_is_release: bool = False,
        model_status: str | ModelStatus = ModelStatus.RELEASED,
        model_page_crawl: bool = True,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _web_url(url, self.name, "catalog URL")
        self.provider_namespace = _namespace(provider_namespace, self.name)
        self.model_card_url_template = _required_text(
            model_card_url_template, "model-card URL template"
        )
        self.mapping = dict(mapping)
        self.items_path = _required_text(self.mapping.get("items_path"), "mapping.items_path")
        self.id_path = _required_text(self.mapping.get("id_path"), "mapping.id_path")
        self.name_path = _required_text(self.mapping.get("name_path"), "mapping.name_path")
        self.created_path = _text(self.mapping.get("created_path"))
        self.updated_path = _text(self.mapping.get("updated_path"))
        self.base_model_path = _text(self.mapping.get("base_model_path"))
        self.next_cursor_path = _text(self.mapping.get("next_cursor_path"))
        self.next_cursor_condition_path = _text(
            self.mapping.get("next_cursor_condition_path")
        )
        self.next_url_path = _text(self.mapping.get("next_url_path"))
        self.total_path = _text(self.mapping.get("total_path"))
        self.cursor_param = _required_text(cursor_param, "cursor parameter")
        self.page_size = int(page_size)
        if self.page_size < 1:
            raise ValueError(f"{self.name}: page size must be positive")
        self.page_size_param = _text(page_size_param)
        self.provider_id_is_release = bool(provider_id_is_release)
        self.model_status = ModelStatus(model_status)
        self.model_page_crawl = bool(model_page_crawl)
        self.max_response_bytes = _positive_int(
            max_response_bytes, "max_response_bytes", self.name
        )
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)

        token = _text(auth_token)
        header_name = _text(auth_header_name)
        query_param = _text(auth_query_param)
        if header_name and query_param:
            raise ValueError(
                f"{self.name}: configure authentication as either a header or query parameter"
            )
        if token and not header_name and not query_param:
            header_name = "Authorization"
        if header_name and not _HEADER_NAME.fullmatch(header_name):
            raise ValueError(f"{self.name}: invalid authentication header name")
        if query_param and not _QUERY_PARAMETER_NAME.fullmatch(query_param):
            raise ValueError(f"{self.name}: invalid authentication query parameter name")
        scheme = _text(auth_scheme)
        if any(character in scheme for character in "\r\n"):
            raise ValueError(f"{self.name}: invalid authentication scheme")
        self._auth_token = token
        self._auth_header_name = header_name
        self._auth_query_param = query_param
        self._auth_header_value = f"{scheme} {token}".strip() if token else ""
        self._static_headers = _static_headers(static_headers, self.name)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "json-catalog-v1",
                "url": self.url,
                "provider_namespace": self.provider_namespace,
                "model_card_url_template": self.model_card_url_template,
                "mapping": self.mapping,
                "cursor_param": self.cursor_param,
                "page_size": self.page_size,
                "page_size_param": self.page_size_param,
                "auth_header_name": self._auth_header_name,
                "auth_query_param": self._auth_query_param,
                "auth_scheme": scheme,
                "static_headers": self._static_headers,
                "max_response_bytes": self.max_response_bytes,
                "provider_id_is_release": self.provider_id_is_release,
                "model_status": self.model_status.value,
                "model_page_crawl": self.model_page_crawl,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        next_url = _text(state.get("next_url"))
        cursor = _text(state.get("cursor"))
        resuming_scan = bool(next_url or cursor)
        raw_items_seen = _state_count(state, "raw_items_seen") if resuming_scan else 0
        scan_total = _state_count(state, "scan_total") if resuming_scan else None
        count_is_complete = not resuming_scan or (
            "raw_items_seen" in state and state.get("raw_count_incomplete") is not True
        )
        headers = {"Accept": "application/json", **self._static_headers}
        if self._auth_header_value:
            headers[self._auth_header_name] = self._auth_header_value

        params: dict[str, str | int] = {}
        if next_url:
            request_url = self._safe_next_url(next_url, self.url)
        else:
            request_url = self.url
            if self.page_size_param:
                params[self.page_size_param] = self.page_size
            if cursor:
                self._assert_not_credential(cursor, "pagination cursor")
                params[self.cursor_param] = cursor
        if self._auth_token and self._auth_query_param:
            params[self._auth_query_param] = self._auth_token

        try:
            if next_url:
                response: HttpResponse = self.client.get(
                    request_url,
                    params=params or None,
                    headers=headers,
                )
            else:
                response = self.client.get(
                    request_url,
                    params=params or None,
                    headers=headers,
                )
        except Exception as error:
            if self._auth_token and self._auth_token in str(error):
                raise RuntimeError(f"{self.name}: catalog request failed") from None
            raise

        if not 200 <= response.status < 300:
            raise ValueError(
                f"{self.name}: catalog endpoint returned HTTP {response.status}"
            )
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: catalog response exceeds {self.max_response_bytes} bytes"
            )

        payload = response.json()
        items = _path_value(payload, self.items_path)
        if not _is_sequence(items):
            raise ValueError(f"{self.name}: mapping.items_path did not resolve to a JSON list")
        raw_items_seen = (raw_items_seen or 0) + len(items)

        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for index, item in enumerate(items):
            try:
                records.append(self._record(item, index))
            except (KeyError, TypeError, ValueError) as error:
                raw = _redact_secret(item, self._auth_token)
                raw_mapping = raw if isinstance(raw, Mapping) else {"value": repr(raw)[:1000]}
                candidate_id = (
                    _optional_scalar(_path_value(item, self.id_path))
                    if isinstance(item, Mapping)
                    else None
                )
                candidate_id = _redact_secret(candidate_id, self._auth_token)
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            candidate_id
                            or f"{self.name}:malformed:{content_hash(dict(raw_mapping))[:32]}"
                        ),
                        stage="source_normalize",
                        error=_redact_secret(f"{type(error).__name__}: {error}", self._auth_token),
                        summary={"index": index, "raw": raw_mapping},
                    )
                )
        raw_next_url = _path_value(payload, self.next_url_path) if self.next_url_path else None
        raw_next_cursor = (
            _path_value(payload, self.next_cursor_path) if self.next_cursor_path else None
        )
        parsed_next_url = _optional_scalar(raw_next_url)
        parsed_next_cursor = _optional_scalar(raw_next_cursor)
        has_next_cursor = True
        if self.next_cursor_condition_path:
            has_next_cursor = _required_boolean(
                _path_value(payload, self.next_cursor_condition_path),
                self.name,
                "next-cursor condition",
            )
        response_total = _optional_integer(
            _path_value(payload, self.total_path) if self.total_path else None
        )
        if response_total is not None and count_is_complete:
            scan_total = max(scan_total or 0, response_total)

        if parsed_next_url:
            safe_next_url = self._safe_next_url(parsed_next_url, response.url or request_url)
            if next_url and safe_next_url == request_url:
                raise ValueError(f"{self.name}: pagination URL did not advance")
            next_state: dict[str, Any] = {"next_url": safe_next_url}
        elif has_next_cursor and parsed_next_cursor:
            self._assert_not_credential(parsed_next_cursor, "pagination cursor")
            if cursor and parsed_next_cursor == cursor:
                raise ValueError(f"{self.name}: pagination cursor did not advance")
            next_state = {"cursor": parsed_next_cursor}
        elif has_next_cursor and self.next_cursor_condition_path:
            raise ValueError(f"{self.name}: response signals another page without a cursor")
        else:
            next_state = {}

        if not next_state and scan_total is not None and raw_items_seen < scan_total:
            raise ValueError(
                f"{self.name}: pagination ended after {raw_items_seen} raw item(s), "
                f"before the known total of {scan_total}"
            )
        if next_state:
            next_state["raw_items_seen"] = raw_items_seen
            if scan_total is not None:
                next_state["scan_total"] = scan_total
            if not count_is_complete:
                next_state["raw_count_incomplete"] = True
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=not next_state,
            upstream_count=response_total if response_total is not None else scan_total,
            issues=tuple(issues),
        )

    def _record(self, item: Any, index: int) -> SourceRecord:
        if not isinstance(item, Mapping):
            raise ValueError(f"{self.name}: catalog item {index} is not a JSON object")

        model_id = _required_scalar(
            _path_value(item, self.id_path),
            self.name,
            f"catalog item {index} ID",
        )
        self._assert_not_credential(model_id, f"catalog item {index} ID")
        name = _optional_scalar(_path_value(item, self.name_path)) or model_id
        self._assert_not_credential(name, f"catalog item {index} name")
        card_url = self._model_card_url(model_id, name)
        identifier = Identifier(self.provider_namespace, model_id)
        local_id = f"{self.provider_namespace}:{model_id}#model"
        model = ModelHint(
            local_id=local_id,
            name=name,
            identifiers=(identifier,),
            status=self.model_status,
            locator=_locator(self.name_path),
        )

        relations = tuple(self._base_model_relations(item, model_id, local_id))
        created = self._field_scalar(item, self.created_path, "created timestamp")
        updated = self._field_scalar(item, self.updated_path, "updated timestamp")
        releases = (
            (
                ReleaseHint(
                    local_id=f"{local_id}#release",
                    model_local_id=local_id,
                    version=model_id,
                    identifiers=(identifier,),
                    released_at=created,
                    metadata={"source_kind": "provider_catalog"},
                    locator=_locator(self.id_path),
                ),
            )
            if self.provider_id_is_release
            else ()
        )
        raw = _redact_secret(dict(item), self._auth_token)
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.PROVIDER_PAGE,
            canonical_url=card_url,
            title=name,
            raw=raw,
            published_at=created,
            modified_at=updated,
            identifiers=(identifier,),
            links=(
                Link(
                    card_url,
                    relation="model_page",
                    locator="config:model_card_url_template",
                    crawl=self.model_page_crawl,
                ),
            ),
            models=(model,),
            model_relations=relations,
            releases=releases,
        )

    def _base_model_relations(
        self,
        item: Mapping[str, Any],
        model_id: str,
        subject_local_id: str,
    ) -> Iterable[ModelRelationHint]:
        if not self.base_model_path:
            return ()
        values = _scalar_values(
            _path_value(item, self.base_model_path),
            self.name,
            "base-model field",
        )
        relations = []
        for index, base_id in enumerate(dict.fromkeys(values)):
            self._assert_not_credential(base_id, "base-model field")
            if base_id == model_id:
                continue
            target = ModelHint(
                local_id=f"{self.provider_namespace}:{model_id}#base-model-{index}",
                name=base_id,
                identifiers=(Identifier(self.provider_namespace, base_id),),
                status=ModelStatus.DOCUMENTED,
                locator=_locator(self.base_model_path),
            )
            relations.append(
                ModelRelationHint(
                    subject_local_id=subject_local_id,
                    predicate="base_model",
                    target=target,
                    locator=_locator(self.base_model_path),
                )
            )
        return tuple(relations)

    def _field_scalar(
        self, item: Mapping[str, Any], path: str, label: str
    ) -> str | None:
        if not path:
            return None
        value = _optional_scalar(_path_value(item, path))
        if value:
            self._assert_not_credential(value, label)
        return value or None

    def _model_card_url(self, model_id: str, name: str) -> str:
        values = {
            "id": quote(model_id, safe=""),
            "id_path": _path_safe_identifier(model_id, self.name),
            "name": quote(name, safe=""),
        }
        try:
            rendered = self.model_card_url_template.format_map(values)
        except (KeyError, ValueError) as error:
            raise ValueError(f"{self.name}: invalid model-card URL template") from error
        self._assert_not_credential(rendered, "model-card URL")
        return _web_url(urljoin(self.url, rendered), self.name, "model-card URL")

    def _safe_next_url(self, value: str, base_url: str) -> str:
        self._assert_not_credential(value, "pagination URL")
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

    def _assert_not_credential(self, value: str, label: str) -> None:
        if self._auth_token and self._auth_token in value:
            raise ValueError(f"{self.name}: {label} contained configured credentials")


def _path_value(value: Any, path: str) -> Any:
    path = _text(path)
    if path in {"", "$"}:
        return value
    if path.startswith("$."):
        path = path[2:]
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]
        elif _is_sequence(current) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _scalar_values(value: Any, source: str, label: str) -> tuple[str, ...]:
    if value is _MISSING or value is None or value == "":
        return ()
    if _is_sequence(value):
        result = []
        for item in value:
            scalar = _optional_scalar(item)
            if scalar is None:
                raise ValueError(f"{source}: {label} must contain only scalar values")
            result.append(scalar)
        return tuple(result)
    scalar = _optional_scalar(value)
    if scalar is None:
        raise ValueError(f"{source}: {label} must be a scalar or list of scalars")
    return (scalar,)


def _required_scalar(value: Any, source: str, label: str) -> str:
    result = _optional_scalar(value)
    if not result:
        raise ValueError(f"{source}: {label} is missing")
    return result


def _optional_scalar(value: Any) -> str | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _optional_integer(value: Any) -> int | None:
    if value is _MISSING or value is None or isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source}: {label} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{source}: {label} must be a positive integer")
    return result


def _path_safe_identifier(value: str, source: str) -> str:
    """Encode an identifier for a template path while preserving real separators."""

    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(f"{source}: model ID cannot be represented as a path")
    return quote(value, safe="/")


def _required_boolean(value: Any, source: str, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{source}: {label} must be a boolean")
    return value


def _static_headers(
    value: Mapping[str, str] | None, source: str
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: static headers must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = _text(raw_name)
        header_value = _text(raw_value)
        if not name or not _HEADER_NAME.fullmatch(name):
            raise ValueError(f"{source}: static headers contains an invalid name")
        normalized = name.casefold()
        if normalized in _SENSITIVE_HEADER_NAMES:
            raise ValueError(f"{source}: static headers must not carry credentials")
        if not header_value or "\r" in header_value or "\n" in header_value:
            raise ValueError(f"{source}: static headers contains an invalid value")
        if normalized in {existing.casefold() for existing in result}:
            raise ValueError(f"{source}: static headers contains a duplicate name")
        result[name] = header_value
    return result


def _state_count(state: Mapping[str, Any], key: str) -> int | None:
    if key not in state:
        return None
    value = state.get(key)
    count = _optional_integer(value)
    if count is None:
        raise ValueError(f"invalid JSON catalog checkpoint {key}: {value!r}")
    return count


def _redact_secret(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, Mapping):
        return {
            _redact_secret(key, secret)
            if isinstance(key, str)
            else key: _redact_secret(item, secret)
            for key, item in value.items()
        }
    if _is_sequence(value):
        return [_redact_secret(item, secret) for item in value]
    return value


def _namespace(value: Any, source: str) -> str:
    result = _required_text(value, "provider namespace")
    if any(character.isspace() or ord(character) < 32 for character in result):
        raise ValueError(f"{source}: invalid provider namespace")
    return result.casefold()


def _web_url(value: Any, source: str, label: str) -> str:
    result = canonicalize_url(_required_text(value, label))
    parts = urlsplit(result)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{source}: {label} must be an absolute HTTP(S) URL")
    return result


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    scheme = parts.scheme.casefold()
    port = parts.port if parts.port is not None else (443 if scheme == "https" else 80)
    return scheme, (parts.hostname or "").casefold(), port


def _locator(path: str) -> str:
    return path if path.startswith("$") else f"$.{path}"


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = ["JsonCatalogSourceAdapter"]
