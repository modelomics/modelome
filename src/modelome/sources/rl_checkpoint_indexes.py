"""Checkpoint entries from OpenAI's RL Clarity Procgen/IMPALA release index."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote

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
from modelome.normalize import canonicalize_url, content_hash

_INDEX_URL = "https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/index.html"
_ARTIFACT_ROOT = "https://openaipublic.blob.core.windows.net/rl-clarity/attribution/models/"
_REPOSITORY_URL = "https://github.com/openai/understanding-rl-vision"
_HANDLE = re.compile(
    r"^(?:coinrun|finite_levels/run_[12]/coinrun_(?:100|300|1000|3000|10000|30000|100000)"
    r"|edit/[a-z0-9_]+|procgen/[a-z0-9_]+|impala/[a-z0-9_]+)$"
)
_PATH_IN_ITEM = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<handle>(?:coinrun|finite_levels/run_[12]/coinrun_"
    r"(?:100|300|1000|3000|10000|30000|100000)|edit/[a-z0-9_]+|procgen/[a-z0-9_]+|"
    r"impala/[a-z0-9_]+))(?![A-Za-z0-9_./-])"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _ListItemParser(HTMLParser):
    """Extract list item text without following links or interpreting markup."""

    def __init__(self, *, maximum: int) -> None:
        super().__init__(convert_charrefs=True)
        self.maximum = maximum
        self.items: list[str] = []
        self._items: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() == "li":
            self._items.append([])

    def handle_data(self, data: str) -> None:
        if self._items:
            self._items[-1].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() != "li" or not self._items:
            return
        item = self._items.pop()
        text = " ".join("".join(item).split())
        if text:
            self.items.append(text)
            if len(self.items) > self.maximum:
                raise ValueError(f"RL Clarity index exceeds {self.maximum} list items")


class RlClarityCheckpointIndexAdapter:
    """Index exact `.jd` RL checkpoints listed by OpenAI's RL Clarity page.

    The page documents each model as `<relpath>/<name>` and declares its primary
    checkpoint at `<relpath>/<name>.jd`, relative to the page's model directory.
    This adapter reads only the page's explicit model list, constructs those
    documented object URLs, and never fetches checkpoint bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only CoinRun, finite-level CoinRun, edited CoinRun, Procgen, and "
        "IMPALA model handles in the OpenAI RL Clarity index. It excludes associated "
        "metadata, derived Lucid files, rollouts, and checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "openai-rl-clarity-checkpoints",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        limits = (("max_response_bytes", max_response_bytes), ("max_entries", max_entries))
        for label, value in limits:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "rl-clarity-checkpoint-index-v1",
                "index_url": _INDEX_URL,
                "artifact_root": _ARTIFACT_ROOT,
                "handle_pattern": _HANDLE.pattern,
                "max_response_bytes": max_response_bytes,
                "max_entries": max_entries,
                "admission": "explicit indexed model handles with documented .jd checkpoint suffix",
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(
            _INDEX_URL, headers={"Accept": "text/html"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: index exceeds {self.max_response_bytes} bytes")
        parser = _ListItemParser(maximum=self.max_entries * 10)
        parser.feed(response.text())
        parser.close()
        handles: list[str] = []
        seen: set[str] = set()
        for item in parser.items:
            for match in _PATH_IN_ITEM.finditer(item):
                handle = match.group("handle")
                if not _HANDLE.fullmatch(handle):
                    raise ValueError(f"{self.name}: invalid checkpoint handle {handle!r}")
                if handle in seen:
                    raise ValueError(f"{self.name}: duplicate checkpoint handle {handle!r}")
                seen.add(handle)
                handles.append(handle)
                if len(handles) > self.max_entries:
                    raise ValueError(
                        f"{self.name}: checkpoint inventory exceeds {self.max_entries}"
                    )
        if not handles:
            raise ValueError(f"{self.name}: index contains no recognized checkpoint handles")
        ordered = sorted(handles)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(handle) for handle in ordered)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "index_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, handle: str) -> SourceRecord:
        encoded = quote(handle, safe="/")
        checkpoint_url = f"{_ARTIFACT_ROOT}{encoded}.jd"
        local_id = f"checkpoint:{content_hash(handle)[:24]}"
        identifier = Identifier("openai:rl-clarity-checkpoint", handle)
        model = ModelHint(
            local_id=local_id,
            name=handle.rsplit("/", 1)[-1],
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=handle,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(handle)[:24]}",
            model_local_id=local_id,
            identifiers=(Identifier("openai:rl-clarity-checkpoint-release", handle),),
            metadata={"checkpoint_path": f"{handle}.jd", "weight_url": checkpoint_url},
            locator=handle,
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{content_hash(handle)[:24]}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(checkpoint_url),
            title=f"OpenAI RL Clarity {handle}",
            raw={"checkpoint_path": f"{handle}.jd", "weight_url": checkpoint_url},
            text=f"RL Clarity model checkpoint: {handle}.jd",
            identifiers=(identifier,),
            links=(
                Link(_INDEX_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(checkpoint_url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["RlClarityCheckpointIndexAdapter"]
