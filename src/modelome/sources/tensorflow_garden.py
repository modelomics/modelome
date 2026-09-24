"""Enumerate checkpoint rows published in TensorFlow Model Garden documentation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

_REPOSITORY = "tensorflow/models"
_DOCS = (
    "official/vision/MODEL_GARDEN.md",
    "official/nlp/docs/pretrained_models.md",
    "official/nlp/MODEL_GARDEN.md",
)
_SHA = re.compile(r"^[0-9a-f]{40}$")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_INLINE_HUB = re.compile(r"(?P<url>https://tfhub\.dev/[^\s,|`]+)")
_INIT_CHECKPOINT = re.compile(r"task\.init_checkpoint=(?P<url>gs://[^\s,|`]+)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_NLP_DOC = "official/nlp/docs/pretrained_models.md"


class TensorFlowGardenSourceAdapter:
    """Read first-party TensorFlow Model Garden checkpoint tables at a commit.

    At most one documentation file is read per call. Only table rows with a
    declared checkpoint, SavedModel, or pretrained model download are emitted;
    checkpoint payloads are treated as references and never fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers checkpoint-linked rows in the official TensorFlow Models vision "
        "and NLP Model Garden documents. It does not enumerate every research/ "
        "directory, independently hosted model, or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "tensorflow-model-garden",
        max_bytes: int = 8 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self.name = name
        self.max_bytes = max_bytes
        self.client = client or HttpClient(max_response_bytes=max_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "tensorflow-model-garden-v1",
                "repository": _REPOSITORY,
                "docs": _DOCS,
                "max_bytes": max_bytes,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/main"

    def raw_url(self, revision: str, path: str) -> str:
        return f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{quote(path, safe='/')}"

    def blob_url(self, revision: str, path: str) -> str:
        return f"https://github.com/{_REPOSITORY}/blob/{revision}/{quote(path, safe='/')}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        index = state.get("doc_index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= len(_DOCS):
            raise ValueError(f"{self.name}: invalid document cursor")
        revision = state.get("revision")
        if revision is None:
            revision_response = self.client.get(
                self.commit_url, headers={"Accept": "application/vnd.github+json"}
            )
            if revision_response.status != 200:
                raise ValueError(
                    f"{self.name}: commit endpoint returned HTTP {revision_response.status}"
                )
            payload = revision_response.json()
            revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: scan state/commit did not contain a SHA-1 revision")
        checked = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if index == len(_DOCS):
            return SourcePage(
                records=(), next_state={**state, "checked_at": checked}, complete=True
            )
        path = _DOCS[index]
        url = self.raw_url(revision, path)
        response = self.client.get(url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: documentation returned HTTP {response.status}")
        if len(response.body) > self.max_bytes:
            raise ValueError(f"{self.name}: documentation exceeds {self.max_bytes} bytes")
        records = self._records(path, revision, response.text())
        next_index = index + 1
        return SourcePage(
            records=records,
            next_state={"revision": revision, "doc_index": next_index, "checked_at": checked},
            complete=next_index == len(_DOCS),
            upstream_count=len(records),
        )

    def _records(self, path: str, revision: str, markdown: str) -> tuple[SourceRecord, ...]:
        records = []
        row_number = 0
        headers: list[str] = []
        heading = ""
        for line in markdown.splitlines():
            if match := _HEADING.match(line):
                heading = re.sub(r"[`*~]", "", match.group(1)).strip()
                continue
            if not line.lstrip().startswith("|") or re.match(r"\s*\|?\s*:?-{3,}", line):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2:
                continue
            normalized_cells = [
                re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", c).strip().lower() for c in cells
            ]
            if any(c == "model" or c == "name" for c in normalized_cells):
                headers = normalized_cells
                continue
            row_number += 1
            model_cell = cells[0]
            checkpoint_columns = [
                i
                for i, header in enumerate(headers)
                if re.search(
                    r"checkpoint|ckpt|saved.?model|pre.?trained|weights?|download|(?:ex|exr)tra.?params",
                    header,
                )
            ]
            checkpoint_links = []
            for column in checkpoint_columns:
                if column < len(cells):
                    cell_links = _MARKDOWN_LINK.findall(cells[column])
                    if "params" in headers[column]:
                        cell_links = []
                    if "download" in headers[column] and not re.search(
                        r"checkpoint|ckpt|weights?", headers[column]
                    ):
                        cell_links = [
                            (label, target)
                            for label, target in cell_links
                            if re.search(r"checkpoint|ckpt|weights?", label, re.I)
                        ]
                    checkpoint_links.extend(cell_links)
                    # The official NLP experiment table embeds some artifact
                    # references as parameter values instead of Markdown links.
                    # Admit only explicit init_checkpoint and TF Hub fields.
                    text_cell = cells[column]
                    checkpoint_links.extend(
                        ("init_checkpoint", _gcs_https(match.group("url")))
                        for match in _INIT_CHECKPOINT.finditer(text_cell)
                    )
                    checkpoint_links.extend(
                        ("tfhub", match.group("url").rstrip(".,;)>"))
                        for match in _INLINE_HUB.finditer(text_cell)
                    )
            if not checkpoint_links:
                continue
            name = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", model_cell).strip().strip("`")
            if not name or name.lower() in {"model", "name"}:
                continue
            unique_links: dict[str, str] = {}
            for label, target in checkpoint_links:
                unique_links.setdefault(target, label)
            checkpoint_links = [(label, target) for target, label in unique_links.items()]
            model_identity = f"{path}:{heading or 'root'}:{name}"
            identity = f"{model_identity}:{'|'.join(url for _, url in checkpoint_links)}"
            model_id = f"model:{content_hash(identity)[:24]}"
            identifier = Identifier("tensorflow-model-garden:model", model_identity)
            model_identifiers = (identifier,)
            release_identifiers = (Identifier("tensorflow-model-garden:row", identity),)
            if path == _NLP_DOC:
                shared_identity = f"{heading or 'root'} / {name}"
                model_identifiers += (
                    Identifier("tensorflow:model-garden-nlp-model", shared_identity),
                )
                release_identifiers += (
                    Identifier("tensorflow:model-garden-nlp-model-release", shared_identity),
                )
            model = ModelHint(
                local_id=model_id,
                name=name,
                aliases=(),
                identifiers=model_identifiers,
                status=ModelStatus.RELEASED,
                locator=f"{path}:model:{name}",
            )
            source_url = self.blob_url(revision, path)
            release = ReleaseHint(
                local_id=f"release:{content_hash(identity)[:24]}",
                model_local_id=model_id,
                version=_declared_tfhub_version(checkpoint_links),
                revision=revision,
                identifiers=release_identifiers,
                metadata={
                    "document": path,
                    "row": row_number,
                    "checkpoints": [u for _, u in checkpoint_links],
                },
                locator=model.locator,
            )
            refs = tuple(
                Link(
                    target,
                    relation="weights",
                    locator=model.locator,
                    crawl=False,
                    model_local_ids=(model_id,),
                )
                for _, target in checkpoint_links
            )
            records.append(
                SourceRecord(
                    source_record_id=f"tensorflow-garden:{content_hash(identity)[:24]}",
                    kind=ArtifactKind.MODEL_CARD,
                    canonical_url=canonicalize_url(source_url),
                    title=name,
                    raw={
                        "repository": _REPOSITORY,
                        "revision": revision,
                        "document": path,
                        "row": row_number,
                        "checkpoint_links": [
                            {"label": label, "url": target} for label, target in checkpoint_links
                        ],
                    },
                    text=f"TensorFlow Model Garden published model: {name}. Source: {source_url}",
                    identifiers=model_identifiers,
                    links=(
                        Link(source_url, relation="model_card", locator=model.locator, crawl=False),
                        Link(
                            f"https://github.com/{_REPOSITORY}",
                            relation="source_repository",
                            crawl=False,
                        ),
                        *refs,
                    ),
                    models=(model,),
                    releases=(release,),
                )
            )
        return tuple(records)


def _gcs_https(url: str) -> str:
    """Represent an explicitly declared GCS checkpoint path as an HTTPS URL."""
    bucket_and_path = url.removeprefix("gs://")
    return f"https://storage.googleapis.com/{bucket_and_path}"


def _declared_tfhub_version(checkpoint_links: list[tuple[str, str]]) -> str | None:
    """Keep an explicit numeric TF Hub version when the doc declares one.

    Hub handles conventionally end in a version path segment (for example
    ``https://tfhub.dev/tf/bert/1``). Other checkpoint URLs and unversioned
    handles do not provide a reliable release version and remain unset.
    """
    versions = {
        match.group("version")
        for _, url in checkpoint_links
        if urlsplit(url).hostname == "tfhub.dev"
        if (match := re.search(r"/(?P<version>[0-9]+)/?$", urlsplit(url).path))
    }
    return next(iter(versions)) if len(versions) == 1 else None
