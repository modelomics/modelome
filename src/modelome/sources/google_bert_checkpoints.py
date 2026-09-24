"""Enumerate the original Google Research BERT checkpoint archives.

The archived first-party README links exact Google Cloud Storage archives for
the paper models, multilingual variants, and 24 compact BERT variants. This
adapter accepts only the documented dated ``bert_models`` ZIP name pattern and
never downloads the model archives.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpResponse
from modelome.models import SourcePage
from modelome.normalize import canonicalize_url, content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_MODEL_ARCHIVE = re.compile(
    r"^/(?P<collection>bert_models)/(?P<release>[0-9]{4}_[0-9]{2}_[0-9]{2})/"
    r"(?P<model>(?:wwm_)?(?:uncased|cased|multi_cased|multilingual|chinese)_"
    r"L-[0-9]+_H-[0-9]+_A-[0-9]+)\.zip$",
    re.IGNORECASE,
)
_MARKDOWN_LINK = re.compile(r"\[[^\]\r\n]+\]\((?P<url>https?://[^)\s]+)\)")
_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<tail>.*)$")


class GoogleResearchBertCheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read only direct, dated BERT checkpoint links from Google's README."""

    coverage_limitation = (
        "Covers exact dated BERT model ZIP links under Google Research's "
        "google-research/bert_models GCS prefix, as declared in the pinned "
        "google-research/bert README. It excludes the all-model bundle and does "
        "not crawl or download model bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "google-research-bert-checkpoints")
        kwargs.setdefault("repository", "google-research/bert")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "README.md")
        kwargs.setdefault("provider_namespace", "google-bert:model")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "google-research-bert-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "dated Google BERT model ZIPs declared by markdown links",
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
            headers={"Accept": "text/markdown,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: BERT README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: BERT README exceeds {self.max_response_bytes} bytes"
            )
        checkpoints = _parse_bert_archives(
            response.text(),
            source=self.name,
            path=self.source_path,
            maximum=self.max_entries,
        )
        if not checkpoints:
            raise ValueError(f"{self.name}: README contains no matching checkpoint links")
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(response.body),
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


def _parse_bert_archives(
    document: str,
    *,
    source: str,
    path: str,
    maximum: int,
) -> tuple[_Checkpoint, ...]:
    by_handle: dict[str, _Checkpoint] = {}
    fence: tuple[str, int] | None = None
    for line_number, line in enumerate(document.splitlines(), start=1):
        if match := _FENCE.match(line):
            marker = match.group("fence")
            if fence is None:
                if marker[0] != "`" or "`" not in match.group("tail"):
                    fence = (marker[0], len(marker))
            elif (
                marker[0] == fence[0]
                and len(marker) >= fence[1]
                and not match.group("tail").strip()
            ):
                fence = None
            continue
        if fence is not None:
            continue
        for match in _MARKDOWN_LINK.finditer(line):
            url = canonicalize_url(match.group("url"))
            parsed = urlsplit(url)
            archive = _MODEL_ARCHIVE.fullmatch(parsed.path)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "storage.googleapis.com"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or archive is None
            ):
                continue
            handle = f"{archive.group('release')}/{archive.group('model')}"
            checkpoint = _Checkpoint(
                handle=handle,
                url=url,
                locator=f"{path}:line:{line_number}",
            )
            previous = by_handle.get(handle)
            if previous is not None and previous.url != url:
                raise ValueError(f"{source}: duplicate model archive handle {handle!r}")
            by_handle[handle] = checkpoint
            if len(by_handle) > maximum:
                raise ValueError(f"{source}: checkpoint list exceeds {maximum} entries")
    return tuple(by_handle[key] for key in sorted(by_handle))


__all__ = ["GoogleResearchBertCheckpointSourceAdapter"]
