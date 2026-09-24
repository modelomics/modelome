"""Dopamine's official legacy Atari checkpoint bundle releases."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

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

_DOCS_URL = "https://google.github.io/dopamine/docs/"
_REPOSITORY_URL = "https://github.com/google/dopamine"
_ARTIFACT_HOST = "storage.cloud.google.com"
_BUCKET_PREFIX = "/download-dopamine-rl/"
_BUNDLES = {
    "dqn_checkpoints.tar.gz": "dqn",
    "c51_checkpoints.tar.gz": "c51",
    "rainbow_checkpoints.tar.gz": "rainbow",
    "iqn_checkpoints.tar.gz": "iqn",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


class DopamineCheckpointBundleAdapter:
    """Index only the four checkpoint archives linked by Dopamine's docs.

    The docs describe each archive as a TensorFlow checkpoint collection for one
    agent across 60 Atari games and five runs. Archive links are catalogued as
    bundles; this adapter does not claim to enumerate their internal files.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Indexes only Dopamine's four documented TensorFlow checkpoint archives. "
        "It does not enumerate archive members, JAX checkpoints, or checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "dopamine-checkpoint-bundles",
        max_response_bytes: int = 2 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "dopamine-checkpoint-bundles-v1",
                "docs_url": _DOCS_URL,
                "artifact_host": _ARTIFACT_HOST,
                "bundle_names": sorted(_BUNDLES),
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(_DOCS_URL, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: documentation returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: documentation exceeds {self.max_response_bytes} bytes")
        parser = _Links()
        parser.feed(response.text())
        parser.close()
        links: dict[str, str] = {}
        for href in parser.hrefs:
            parsed = urlsplit(href)
            filename = parsed.path.rsplit("/", 1)[-1]
            if parsed.hostname != _ARTIFACT_HOST or not parsed.path.startswith(_BUCKET_PREFIX):
                continue
            if filename not in _BUNDLES:
                continue
            url = canonicalize_url(href)
            if filename in links and links[filename] != url:
                raise ValueError(f"{self.name}: conflicting URL for {filename}")
            links[filename] = url
        if set(links) != set(_BUNDLES):
            missing = sorted(set(_BUNDLES) - set(links))
            raise ValueError(f"{self.name}: documentation is missing checkpoint bundles: {missing}")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(filename, links[filename]) for filename in sorted(links))
        return SourcePage(
            records=records,
            next_state={"checked_at": checked_at, "docs_sha256": content_hash(response.body)},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, url: str) -> SourceRecord:
        agent = _BUNDLES[filename]
        local_id = f"checkpoint-bundle:{content_hash(agent)[:24]}"
        model = ModelHint(
            local_id=local_id,
            name=f"Dopamine {agent.upper()} Atari checkpoint bundle",
            identifiers=(Identifier("google:dopamine-agent", agent),),
            status=ModelStatus.RELEASED,
            locator=agent,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(agent)[:24]}",
            model_local_id=local_id,
            identifiers=(Identifier("google:dopamine-checkpoint-bundle", filename),),
            metadata={"agent": agent, "archive_filename": filename},
            locator=filename,
        )
        return SourceRecord(
            source_record_id=f"checkpoint-bundle:{content_hash(agent)[:24]}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=url,
            title=f"Dopamine {agent.upper()} Atari checkpoint archive",
            raw={"agent": agent, "archive_filename": filename, "weight_url": url},
            text=(
                f"Official Dopamine TensorFlow checkpoint archive for {agent.upper()}; "
                "the documentation describes five runs across 60 Atari games."
            ),
            identifiers=(Identifier("google:dopamine-checkpoint-bundle", filename),),
            links=(
                Link(_DOCS_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["DopamineCheckpointBundleAdapter"]
