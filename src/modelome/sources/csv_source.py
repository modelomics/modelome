from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from io import StringIO
from typing import Any

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelRelationHint,
    ModelStatus,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash, extract_urls, identifier_from_url

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CsvSourceAdapter:
    """Turn a configured CSV catalog into source records.

    Column names and their meaning are supplied by ``mapping``. The adapter has
    no source-specific vocabulary, which lets catalogs change without a code
    release and makes another CSV corpus a configuration-only addition.
    """

    def __init__(
        self,
        *,
        name: str,
        url: str,
        mapping: Mapping[str, Any],
        artifact_kind: str | ArtifactKind = ArtifactKind.CATALOG_RECORD,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.url = url
        self.mapping = dict(mapping)
        self.artifact_kind = ArtifactKind(artifact_kind)
        self.client = client or HttpClient()
        self.clock = clock
        self.headers = dict(headers or {})
        self.checkpoint_signature = content_hash(
            {
                "adapter": "csv-v1",
                "url": self.url,
                "artifact_kind": self.artifact_kind.value,
                "mapping": self.mapping,
                "header_names": sorted(key.casefold() for key in self.headers),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        request_headers = dict(self.headers)
        if etag := _text(state.get("etag")):
            request_headers["If-None-Match"] = etag
        if last_modified := _text(state.get("http_last_modified")):
            request_headers["If-Modified-Since"] = last_modified

        response: HttpResponse = self.client.get(self.url, headers=request_headers)
        checked_at = _isoformat(self.clock())
        if response.status == 304:
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            # A 304 proves freshness, not an upstream cardinality of zero. Keep
            # the last successful count in storage by leaving this unset.
            return SourcePage(records=(), next_state=next_state, complete=True)

        rows = tuple(self._read_rows(response.text()))
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for line_number, row in enumerate(rows, start=2):
            try:
                records.append(self._record(row, line_number))
            except (KeyError, TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            _record_id(row, self.mapping, self.name)
                        ),
                        stage="source_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"line": line_number, "row": dict(row)},
                    )
                )
        next_state: dict[str, Any] = {
            "checked_at": checked_at,
            "content_hash": content_hash(response.body),
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        if last_modified := _header(response.headers, "last-modified"):
            next_state["http_last_modified"] = last_modified
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(rows),
            authoritative_snapshot=True,
            issues=tuple(issues),
        )

    def _read_rows(self, body: str) -> Iterable[dict[str, str]]:
        delimiter = _text(self.mapping.get("delimiter")) or ","
        if len(delimiter) != 1:
            raise ValueError(f"{self.name}: CSV delimiter must be one character")
        reader = csv.DictReader(StringIO(body.lstrip("\ufeff")), delimiter=delimiter)
        if reader.fieldnames is None:
            return
        required_fields = {
            _required_field(self.mapping, "title_field", self.name),
            *_field_names(self.mapping, "id_fields"),
        }
        if model_field := _text(self.mapping.get("model_field")):
            required_fields.add(model_field)
        missing_fields = sorted(required_fields - set(reader.fieldnames))
        if missing_fields:
            raise ValueError(
                f"{self.name}: CSV schema is missing required field(s): "
                + ", ".join(missing_fields)
            )
        for row in reader:
            # DictReader uses ``None`` for overflow columns and missing values.
            yield {
                str(key): "" if value is None else str(value).strip()
                for key, value in row.items()
                if key is not None
            }

    def _record(self, row: Mapping[str, str], line_number: int) -> SourceRecord:
        record_id = _record_id(row, self.mapping, self.name)

        title_field = _required_field(self.mapping, "title_field", self.name)
        title = row.get(title_field, "").strip() or record_id
        link_fields = _field_names(self.mapping, "link_fields")
        links = _links_from_fields(row, link_fields, line_number)
        canonical_url = links[0].url if links else self.url

        body = []
        for field in _field_names(self.mapping, "body_fields"):
            if value := row.get(field, "").strip():
                body.append(f"{field}: {value}")

        model, relations = self._model_hints(row, record_id, line_number)
        identifiers = _unique_identifiers(
            identifier
            for link in links
            if (identifier := identifier_from_url(link.url)) is not None
        )
        return SourceRecord(
            source_record_id=record_id,
            kind=self.artifact_kind,
            canonical_url=canonical_url,
            title=title,
            raw=dict(row),
            text="\n".join(body),
            published_at=_field_value(row, self.mapping.get("published_field")),
            modified_at=_field_value(row, self.mapping.get("modified_field")),
            identifiers=identifiers,
            links=links,
            models=(model,) if model is not None else (),
            model_relations=relations,
        )

    def _model_hints(
        self,
        row: Mapping[str, str],
        record_id: str,
        line_number: int,
    ) -> tuple[ModelHint | None, tuple[ModelRelationHint, ...]]:
        model_field = _text(self.mapping.get("model_field"))
        if not model_field or not (name := row.get(model_field, "").strip()):
            return None, ()

        status_name = _text(self.mapping.get("model_status")) or ModelStatus.DOCUMENTED.value
        aliases = tuple(
            dict.fromkeys(
                value
                for field in _field_names(self.mapping, "alias_fields")
                for value in _mapped_values(row.get(field, ""), self.mapping.get("alias_separator"))
                if value != name
            )
        )
        model_identifiers = _unique_identifiers(
            identifier
            for field in _field_names(self.mapping, "link_fields")
            for url in extract_urls(row.get(field, ""))
            if (identifier := identifier_from_url(url)) is not None
            and identifier.namespace == "huggingface:model"
        )
        model = ModelHint(
            local_id=f"{record_id}#model",
            name=name,
            identifiers=model_identifiers,
            aliases=aliases,
            status=ModelStatus(status_name),
            locator=f"csv:line={line_number};column={model_field}",
        )

        base_field = _text(self.mapping.get("base_model_field"))
        if not base_field:
            return model, ()
        relations = []
        for index, base_name in enumerate(
            _mapped_values(row.get(base_field, ""), self.mapping.get("base_model_separator"))
        ):
            target = ModelHint(
                local_id=f"{record_id}#base-model-{index}",
                name=base_name,
                status=ModelStatus.DOCUMENTED,
                locator=f"csv:line={line_number};column={base_field}",
            )
            relations.append(
                ModelRelationHint(
                    subject_local_id=model.local_id,
                    predicate="base_model",
                    target=target,
                    locator=f"csv:line={line_number};column={base_field}",
                )
            )
        return model, tuple(relations)


def _required_field(mapping: Mapping[str, Any], key: str, source: str) -> str:
    value = _text(mapping.get(key))
    if not value:
        raise ValueError(f"{source}: mapping.{key} is required")
    return value


def _record_id(row: Mapping[str, str], mapping: Mapping[str, Any], source: str) -> str:
    id_fields = _field_names(mapping, "id_fields")
    identity = {field: row.get(field, "") for field in id_fields}
    if not any(identity.values()):
        identity = dict(row)
    return f"{source}:{content_hash(identity)[:32]}"


def _field_names(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, ())
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence):
        return ()
    return tuple(item for item in (_text(item) for item in value) if item)


def _field_value(row: Mapping[str, str], field: Any) -> str | None:
    name = _text(field)
    if not name:
        return None
    return row.get(name, "").strip() or None


def _mapped_values(value: Any, separator: Any) -> tuple[str, ...]:
    text = _text(value)
    if not text:
        return ()
    delimiter = _text(separator)
    values = text.split(delimiter) if delimiter else [text]
    return tuple(dict.fromkeys(item.strip() for item in values if item.strip()))


def _links_from_fields(
    row: Mapping[str, str], fields: Sequence[str], line_number: int
) -> tuple[Link, ...]:
    links: list[Link] = []
    seen: set[str] = set()
    for field in fields:
        for url in extract_urls(row.get(field, "")):
            if url in seen:
                continue
            seen.add(url)
            links.append(
                Link(
                    url=url,
                    relation="references",
                    locator=f"csv:line={line_number};column={field}",
                )
            )
    return tuple(links)


def _unique_identifiers(values: Iterable[Identifier]) -> tuple[Identifier, ...]:
    return tuple(dict.fromkeys(values))


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.casefold()
    return next((value for key, value in headers.items() if key.casefold() == wanted), None)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["CsvSourceAdapter"]
