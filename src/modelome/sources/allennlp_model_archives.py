"""First-party AllenNLP pretrained model-card/archive inventory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient
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
from modelome.sources.static_json_checkpoint_registry import (
    _COMMIT,
    _header,
    _isoformat,
    _nonnegative_int,
    _positive_int,
    _required_text,
    _text,
    _utcnow,
)

_HANDLE = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_ARCHIVE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}\.tar\.gz$")


class AllenNLPModelArchiveSourceAdapter:
    """Read AllenNLP's model cards and exact archive filenames, without loading weights."""

    def __init__(
        self,
        *,
        name: str,
        repository: str,
        branch: str,
        modelcards_path: str,
        provider_namespace: str,
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _required_text(repository, "repository")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
            raise ValueError("repository must be owner/name")
        self.branch = _required_text(branch, "branch")
        self.modelcards_path = _required_text(modelcards_path, "modelcards path").strip("/")
        if ".." in self.modelcards_path.split("/"):
            raise ValueError("modelcards path cannot traverse directories")
        self.provider_namespace = _required_text(provider_namespace, "provider namespace")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_entries = _positive_int(max_entries, "max_entries")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock

    @property
    def _repo_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def _commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self._commit_url(), headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        directory_url = (
            f"https://api.github.com/repos/{self.repository}/contents/"
            f"{quote(self.modelcards_path, safe='/')}?ref={revision}"
        )
        directory = self.client.get(
            directory_url, headers={"Accept": "application/vnd.github+json"}
        )
        if directory.status != 200:
            raise ValueError(f"{self.name}: model-card directory returned HTTP {directory.status}")
        if len(directory.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model-card directory exceeds response limit")
        items = directory.json()
        if not isinstance(items, list):
            raise ValueError(f"{self.name}: model-card directory response is not a list")
        cards = [
            item
            for item in items
            if isinstance(item, Mapping)
            and item.get("type") == "file"
            and isinstance(item.get("name"), str)
            and item["name"].endswith(".json")
            and item["name"] != "modelcard-template.json"
        ]
        if len(cards) > self.max_entries:
            raise ValueError(
                f"{self.name}: model-card directory exceeds {self.max_entries} entries"
            )
        records: list[SourceRecord] = []
        for item in cards:
            card_path = f"{self.modelcards_path}/{item['name']}"
            card_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{card_path}"
            response = self.client.get(card_url, headers={"Accept": "application/json"})
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: model card {item['name']} returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: model card {item['name']} exceeds response limit")
            card = response.json()
            if not isinstance(card, Mapping):
                raise ValueError(f"{self.name}: model card {item['name']} is not an object")
            archive = _archive_name(card)
            if archive is None:
                continue
            handle = _text(card.get("id"))
            if not _HANDLE.fullmatch(handle):
                raise ValueError(f"{self.name}: invalid model-card ID in {item['name']}")
            records.append(self._record(handle, archive, card_url, card, response.body, revision))
        if not records:
            raise ValueError(f"{self.name}: no model cards declared direct archives")
        next_state = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "model_count": len(records),
            "source_sha256": content_hash(directory.body),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        handle: str,
        archive: str,
        card_url: str,
        card: Mapping[str, Any],
        body: bytes,
        revision: str,
    ) -> SourceRecord:
        model_id = f"model:{handle}"
        model = ModelHint(
            local_id=model_id,
            name=_text(card.get("display_name")) or handle,
            identifiers=(Identifier(self.provider_namespace, handle),),
            status=ModelStatus.RELEASED,
            locator=card_url,
        )
        archive_url = (
            f"https://storage.googleapis.com/allennlp-public-models/{quote(archive, safe='')}"
        )
        release = ReleaseHint(
            local_id=f"release:{archive}",
            model_local_id=model_id,
            identifiers=(Identifier(f"{self.provider_namespace}:archive", archive),),
            revision=revision,
            metadata={
                "archive_file": archive,
                "model_card": card_url,
                "repository": self.repository,
            },
            locator=card_url,
        )
        return SourceRecord(
            source_record_id=f"archive:{handle}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(archive_url),
            title=_text(card.get("display_name")) or handle,
            raw={
                "model_id": handle,
                "archive_file": archive,
                "model_card": card_url,
                "model_card_sha256": content_hash(body),
                "revision": revision,
            },
            text=_text(card.get("model_details", {}).get("short_description"))
            if isinstance(card.get("model_details"), Mapping)
            else "",
            identifiers=(Identifier(f"{self.provider_namespace}:archive", archive),),
            links=(
                Link(archive_url, relation="checkpoint", locator=card_url, crawl=False),
                Link(card_url, relation="model_card", crawl=False),
                Link(
                    f"{self._repo_url}/tree/{revision}/{self.modelcards_path}",
                    relation="source_repository",
                    crawl=False,
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _archive_name(card: Mapping[str, Any]) -> str | None:
    usage = card.get("model_usage")
    if not isinstance(usage, Mapping):
        return None
    archive = usage.get("archive_file")
    if not isinstance(archive, str) or not _ARCHIVE.fullmatch(archive):
        return None
    return archive
