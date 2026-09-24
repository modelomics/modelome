"""Pinned, line-oriented checkpoint cards embedded in YAML-like source files.

Some first-party model registries intentionally concatenate small asset cards in
one human-editable file.  This reader does not implement YAML. It accepts only
top-level literal ``name``, optional model-family/architecture, and direct
``checkpoint`` fields, rejecting ambiguous card shapes rather than evaluating
or interpreting a general serialization language.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from modelome.http import HttpResponse
from modelome.models import SourcePage, SourceRecord
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _checkpoint_url,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_FIELD = re.compile(
    r"^(?P<key>name|model_family|model_arch|checkpoint):[ \t]*(?P<value>[^\r\n#]+?)"
    r"[ \t]*(?:#.*)?$"
)
_HANDLE = re.compile(r"^[^\s\x00-\x1f\x7f](?:[^\x00-\x1f\x7f]{0,511})$")


@dataclass(frozen=True, slots=True)
class _CardCheckpoint(_Checkpoint):
    model_family: str | None
    model_arch: str | None


class LineCheckpointCardCatalogSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read literal top-level checkpoint cards from one pinned source file."""

    coverage_limitation = (
        "Covers literal top-level name/family/architecture/checkpoint cards in one "
        "first-party source file at a pinned commit. It does not parse general YAML, "
        "execute code, infer papers, follow artifact URLs, or download checkpoints."
    )

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        source_path: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            repository=repository,
            branch=branch,
            source_path=source_path,
            provider_namespace=provider_namespace,
            max_response_bytes=max_response_bytes,
            max_entries=max_entries,
            **kwargs,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "line-checkpoint-card-catalog-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal top-level named cards with direct checkpoint URLs",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/yaml,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: card catalog returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: card catalog exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_cards(
            response.text(),
            source=self.name,
            path=self.source_path,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        if not records:
            raise ValueError(f"{self.name}: card catalog contains no checkpoint entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        checkpoint: _CardCheckpoint,
        revision: str,
        source: bytes,
    ) -> SourceRecord:
        record = super()._record(checkpoint, revision, source)
        release = record.releases[0]
        metadata = {
            **release.metadata,
            "model_family": checkpoint.model_family,
            "model_arch": checkpoint.model_arch,
        }
        return replace(
            record,
            raw={
                **record.raw,
                "model_family": checkpoint.model_family,
                "model_arch": checkpoint.model_arch,
            },
            text="\n".join(
                item
                for item in (
                    record.text,
                    checkpoint.model_family,
                    checkpoint.model_arch,
                )
                if item
            ),
            releases=(replace(release, metadata=metadata),),
        )


def _parse_cards(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_CardCheckpoint, ...]:
    cards: list[_CardCheckpoint] = []
    current: dict[str, str] | None = None
    seen_handles: set[str] = set()
    for line_number, line in enumerate(document.splitlines(), start=1):
        if line[:1].isspace():
            continue
        match = _FIELD.fullmatch(line)
        if match is None:
            continue
        key = match.group("key")
        value = _scalar(match.group("value"), source, line_number)
        if key == "name":
            if current is not None and "checkpoint" in current:
                cards.append(_card(current, source, path, seen_handles))
                if len(cards) > maximum:
                    raise ValueError(f"{source}: card catalog exceeds {maximum} entries")
            current = {"name": value, "line_number": str(line_number)}
            continue
        if current is None:
            raise ValueError(f"{source}: {key} precedes a top-level name at line {line_number}")
        if key in current:
            raise ValueError(f"{source}: duplicate {key} in card at line {line_number}")
        current[key] = value
    if current is not None and "checkpoint" in current:
        cards.append(_card(current, source, path, seen_handles))
        if len(cards) > maximum:
            raise ValueError(f"{source}: card catalog exceeds {maximum} entries")
    if not cards:
        raise ValueError(f"{source}: card catalog contains no named direct checkpoints")
    return tuple(cards)


def _card(
    card: Mapping[str, str],
    source: str,
    path: str,
    seen_handles: set[str],
) -> _CardCheckpoint:
    handle = card["name"]
    if (
        not _HANDLE.fullmatch(handle)
        or handle != handle.strip()
        or ".." in handle.split("/")
    ):
        raise ValueError(f"{source}: invalid checkpoint handle {handle!r}")
    if handle in seen_handles:
        raise ValueError(f"{source}: duplicate checkpoint handle {handle!r}")
    seen_handles.add(handle)
    return _CardCheckpoint(
        handle=handle,
        url=_checkpoint_url(card["checkpoint"], source, handle),
        locator=f"{path}:line:{card['line_number']}",
        model_family=card.get("model_family"),
        model_arch=card.get("model_arch"),
    )


def _scalar(value: str, source: str, line_number: int) -> str:
    value = value.strip()
    if len(value) >= 2 and value[:1] == value[-1:] and value[:1] in {"'", '"'}:
        value = value[1:-1]
    if not value or any(character in value for character in "\r\n\x00"):
        raise ValueError(f"{source}: invalid literal value at line {line_number}")
    return value
