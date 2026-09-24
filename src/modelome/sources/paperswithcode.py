from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash, identifier_from_url

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ARXIV_ID = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?$", re.IGNORECASE
)
_METHOD_URL = re.compile(r"^https://paperswithcode\.com/method/[a-z0-9][a-z0-9-]*$")
_PHONE_NUMBER = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){7,}\d")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TOKEN = re.compile(r"[a-z][a-z0-9+-]*", re.IGNORECASE)
_TITLE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "architecture",
        "based",
        "deep",
        "for",
        "in",
        "learning",
        "method",
        "model",
        "models",
        "network",
        "networks",
        "neural",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)
_REQUIRED_COLUMNS = frozenset(
    {
        "paper_url",
        "paper_title",
        "paper_arxiv_id",
        "paper_url_abs",
        "paper_url_pdf",
        "repo_url",
        "is_official",
        "mentioned_in_paper",
        "mentioned_in_github",
        "framework",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PapersWithCodeLinksSourceAdapter:
    """Import the complete, pinned PWC paper-to-code snapshot.

    PWC's ``methods`` archive currently contains visible vandalism, so this adapter
    deliberately consumes only its separate paper/code edge corpus. It retains only
    rows PWC explicitly marks ``is_official`` and records that source-declared flag
    on every edge. The resulting paper and repository artifacts make each edge
    materializable without crawling either endpoint.
    """

    artifact_source: str
    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str = "paperswithcode-links",
        dataset_id: str = "pwc-archive/links-between-paper-and-code",
        metadata_url: str = "https://huggingface.co/api/datasets/pwc-archive/links-between-paper-and-code",
        data_path: str = "data/train-00000-of-00001.parquet",
        license: str = "CC-BY-SA-4.0",
        page_size: int = 10_000,
        max_dataset_bytes: int = 64 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.artifact_source = self.name
        self.dataset_id = _required_text(dataset_id, "dataset ID")
        self.metadata_url = _required_web_url(metadata_url, "metadata URL")
        self.data_path = _safe_data_path(data_path)
        self.license = _required_text(license, "license")
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = int(page_size)
        if max_dataset_bytes < 1:
            raise ValueError("max_dataset_bytes must be positive")
        self.max_dataset_bytes = int(max_dataset_bytes)
        self.client = client or HttpClient(max_response_bytes=self.max_dataset_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paperswithcode-links-v1",
                "dataset_id": self.dataset_id,
                "metadata_url": self.metadata_url,
                "data_path": self.data_path,
                "license": self.license,
                "policy": "is_official=true",
                "page_size": self.page_size,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        prior_revision = _text(state.get("snapshot_revision"))
        prior_paths = _safe_data_paths(state.get("data_paths"))
        resume_snapshot = bool(
            prior_revision and prior_paths and not _text(state.get("completed_snapshot_revision"))
        )
        if resume_snapshot:
            revision, data_paths = prior_revision, prior_paths
            shard_index = _nonnegative_int(state.get("shard_index", 0))
        else:
            metadata_response: HttpResponse = self.client.get(
                self.metadata_url, headers={"Accept": "application/json"}
            )
            metadata = metadata_response.json()
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{self.name}: dataset metadata is not an object")
            revision, data_paths = _snapshot_data_paths(
                metadata, self.dataset_id, self.data_path, self.name
            )
            # Older persisted checkpoints predate data_paths. Their shard cursor
            # remains valid only when the manifest resolves to the same revision.
            shard_index = (
                _nonnegative_int(state.get("shard_index", 0))
                if revision == prior_revision
                else 0
            )
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_snapshot_revision")):
            return SourcePage(
                records=(),
                next_state={
                    "snapshot_revision": revision,
                    "completed_snapshot_revision": revision,
                    "checked_at": checked_at,
                },
                complete=True,
                authoritative_snapshot=False,
            )

        if shard_index >= len(data_paths):
            raise ValueError(f"{self.name}: shard checkpoint is outside the snapshot")
        data_path = data_paths[shard_index]
        data_url = _resolve_url(self.dataset_id, revision, data_path)
        data_response: HttpResponse = self.client.get(
            data_url, headers={"Accept": "application/vnd.apache.parquet"}
        )
        if len(data_response.body) > self.max_dataset_bytes:
            raise ValueError(f"{self.name}: dataset exceeded configured byte limit")
        rows = _read_rows(data_response.body, self.name)
        upstream_count = sum(row.get("is_official") is True for row in rows)
        start = _nonnegative_int(state.get("row_offset", 0)) if revision == prior_revision else 0
        selected, next_offset, complete = _official_page(rows, start, self.page_size)
        shard_complete = complete
        next_shard = shard_index
        if complete and shard_index + 1 < len(data_paths):
            shard_complete = False
            next_shard = shard_index + 1
        records, issues = self._records(selected, revision)
        next_state = {
            "snapshot_revision": revision,
            "data_paths": list(data_paths),
            "shard_index": next_shard,
            "checked_at": checked_at,
            "dataset_url": data_url,
        }
        if shard_complete:
            next_state["completed_snapshot_revision"] = revision
        elif complete:
            next_state["row_offset"] = 0
        else:
            next_state["row_offset"] = next_offset
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=shard_complete,
            upstream_count=upstream_count,
            # Chunked scans cannot reconcile omissions until a separate complete
            # inventory pass observes every source record in one transaction.
            authoritative_snapshot=False,
            issues=tuple(issues),
            # The revision is immutable. Preserve malformed rows as dead letters
            # but do not replay the whole valid batch on every subsequent run.
            advance_on_source_issues=True,
        )

    def _records(
        self, rows: Iterable[tuple[int, Mapping[str, Any]]], revision: str
    ) -> tuple[list[SourceRecord], list[SourceIssue]]:
        repositories: set[str] = set()
        issues: list[SourceIssue] = []
        records: list[SourceRecord] = []
        for index, row in rows:
            try:
                paper_url = _required_web_url(row.get("paper_url"), "paper URL")
                repository_url = _repository_url(row.get("repo_url"))
                title = _required_text(row.get("paper_title"), "paper title")
            except ValueError as error:
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:row:{index}",
                        stage="source_normalize",
                        error=f"ValueError: {error}",
                        summary={"row": index, "is_official": True},
                    )
                )
                continue
            identifiers = [Identifier("paperswithcode:paper", paper_url)]
            if arxiv_id := _arxiv_id(row.get("paper_arxiv_id")):
                identifiers.append(Identifier("arxiv", arxiv_id))
            records.append(
                SourceRecord(
                    source_record_id=(
                        "paper-code:"
                        + content_hash(
                            {"paper_url": paper_url, "repository_url": repository_url}
                        )[:32]
                    ),
                    kind=ArtifactKind.PAPER,
                    canonical_url=paper_url,
                    title=title,
                    raw={
                        "dataset": self.dataset_id,
                        "snapshot_revision": revision,
                        "license": self.license,
                        "admission": "source_declared_is_official",
                        "paper_url": paper_url,
                        "paper_url_abs": _text(row.get("paper_url_abs")) or None,
                        "paper_url_pdf": _text(row.get("paper_url_pdf")) or None,
                        "repository_url": repository_url,
                        "framework": _text(row.get("framework")) or None,
                        "mentioned_in_paper": row.get("mentioned_in_paper") is True,
                        "mentioned_in_github": row.get("mentioned_in_github") is True,
                    },
                    identifiers=tuple(identifiers),
                    links=(
                        Link(
                            repository_url,
                            relation="official_implementation",
                            locator="pwc:is_official",
                            crawl=False,
                        ),
                    ),
                )
            )
            repositories.add(repository_url)

        source_metadata = {
            "dataset": self.dataset_id,
            "snapshot_revision": revision,
            "license": self.license,
            "admission": "source_declared_is_official",
        }
        for repository_url in sorted(repositories):
            identifiers = [Identifier("paperswithcode:repository", repository_url)]
            if identifier := identifier_from_url(repository_url):
                identifiers.append(identifier)
            records.append(
                SourceRecord(
                    source_record_id=f"repository:{content_hash(repository_url)[:32]}",
                    kind=ArtifactKind.CODE_REPOSITORY,
                    canonical_url=repository_url,
                    title=_repository_title(repository_url),
                    raw={
                        **source_metadata,
                        "repository_url": repository_url,
                    },
                    identifiers=tuple(dict.fromkeys(identifiers)),
                )
            )
        return records, issues


class PapersWithCodeValidatedMethodsSourceAdapter:
    """Import PWC method candidates or the stricter arXiv-verified subset.

    The unfiltered archive contains vandalized rows. A retained method must have a
    valid PWC method URL, a source-declared method-paper and source-paper relation,
    and no phone-number payload. The strict tier additionally requires title overlap
    and a substantially matching title returned by arXiv. The broadest tier keeps
    paper-linked rows that are not arXiv-linked as lower-confidence candidates. These
    are source-backed identities, never release or weight assertions.
    """

    artifact_source: str
    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str = "paperswithcode-methods-validated",
        dataset_id: str = "pwc-archive/methods",
        metadata_url: str = "https://huggingface.co/api/datasets/pwc-archive/methods",
        data_path: str = "data/train-00000-of-00001.parquet",
        license: str = "CC-BY-SA-4.0",
        arxiv_api_url: str = "https://export.arxiv.org/api/query",
        arxiv_batch_size: int = 100,
        arxiv_delay_seconds: float = 3.0,
        admission: str = "arxiv_verified",
        max_dataset_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        pause: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.artifact_source = self.name
        self.dataset_id = _required_text(dataset_id, "dataset ID")
        self.metadata_url = _required_web_url(metadata_url, "metadata URL")
        self.data_path = _safe_data_path(data_path)
        self.license = _required_text(license, "license")
        self.arxiv_api_url = _required_web_url(arxiv_api_url, "arXiv API URL")
        if arxiv_batch_size < 1 or arxiv_batch_size > 200:
            raise ValueError("arXiv batch size must be between 1 and 200")
        if arxiv_delay_seconds < 3:
            raise ValueError("arXiv delay must be at least 3 seconds")
        if max_dataset_bytes < 1:
            raise ValueError("max_dataset_bytes must be positive")
        self.arxiv_batch_size = int(arxiv_batch_size)
        self.arxiv_delay_seconds = float(arxiv_delay_seconds)
        self.admission = _required_text(admission, "method admission").casefold()
        if self.admission not in {
            "arxiv_verified",
            "candidate",
            "paper_linked_candidate",
            "paper_candidate",
        }:
            raise ValueError(
                "method admission must be arxiv_verified, candidate, "
                "paper_linked_candidate, or paper_candidate"
            )
        self.max_dataset_bytes = int(max_dataset_bytes)
        self.client = client or HttpClient(max_response_bytes=self.max_dataset_bytes)
        self.pause = pause
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paperswithcode-methods-validated-v1",
                "dataset_id": self.dataset_id,
                "metadata_url": self.metadata_url,
                "data_path": self.data_path,
                "license": self.license,
                "arxiv_api_url": self.arxiv_api_url,
                "arxiv_batch_size": self.arxiv_batch_size,
                "policy": self.admission,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        prior_revision = _text(state.get("snapshot_revision"))
        prior_paths = _safe_data_paths(state.get("data_paths"))
        resume_snapshot = bool(
            prior_revision and prior_paths and not _text(state.get("completed_snapshot_revision"))
        )
        if resume_snapshot:
            revision, data_paths = prior_revision, prior_paths
            shard_index = _nonnegative_int(state.get("shard_index", 0))
        else:
            metadata_response: HttpResponse = self.client.get(
                self.metadata_url, headers={"Accept": "application/json"}
            )
            metadata = metadata_response.json()
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{self.name}: dataset metadata is not an object")
            revision, data_paths = _snapshot_data_paths(
                metadata, self.dataset_id, self.data_path, self.name
            )
            # Older persisted checkpoints predate data_paths. Their shard cursor
            # remains valid only when the manifest resolves to the same revision.
            shard_index = (
                _nonnegative_int(state.get("shard_index", 0))
                if revision == prior_revision
                else 0
            )
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_snapshot_revision")):
            return SourcePage(
                records=(),
                next_state={
                    "snapshot_revision": revision,
                    "completed_snapshot_revision": revision,
                    "checked_at": checked_at,
                },
                complete=True,
                authoritative_snapshot=False,
            )

        if shard_index >= len(data_paths):
            raise ValueError(f"{self.name}: shard checkpoint is outside the snapshot")
        data_path = data_paths[shard_index]
        data_url = _resolve_url(self.dataset_id, revision, data_path)
        data_response: HttpResponse = self.client.get(
            data_url, headers={"Accept": "application/vnd.apache.parquet"}
        )
        if len(data_response.body) > self.max_dataset_bytes:
            raise ValueError(f"{self.name}: dataset exceeded configured byte limit")
        rows = _read_method_rows(data_response.body, self.name)
        candidates, rejected = _method_candidates(
            rows,
            require_title_overlap=self.admission == "arxiv_verified",
            require_arxiv_source=self.admission
            not in {"paper_linked_candidate", "paper_candidate"},
            allow_missing_source=self.admission == "paper_candidate",
        )
        records = []
        if self.admission == "arxiv_verified":
            arxiv_titles = self._arxiv_titles(
                sorted(
                    arxiv_id
                    for item in candidates
                    if (arxiv_id := _text(item.get("arxiv_id")))
                )
            )
            for candidate in candidates:
                verified_title = arxiv_titles.get(candidate["arxiv_id"])
                if verified_title is None or not _titles_substantially_match(
                    candidate["source_title"], verified_title
                ):
                    rejected["arxiv_title_mismatch"] = (
                        rejected.get("arxiv_title_mismatch", 0) + 1
                    )
                    continue
                records.append(self._record(candidate, revision, verified_title))
        else:
            records = [self._record(candidate, revision, None) for candidate in candidates]
        complete = shard_index + 1 == len(data_paths)
        next_state: dict[str, Any] = {
            "snapshot_revision": revision,
            "data_paths": list(data_paths),
            "shard_index": shard_index + 1,
            "checked_at": checked_at,
            "dataset_url": data_url,
            "candidate_count": _nonnegative_int(state.get("candidate_count", 0))
            + len(candidates),
            "validated_count": _nonnegative_int(state.get("validated_count", 0))
            + len(records),
            "rejected": _merge_counts(state.get("rejected"), rejected),
        }
        if complete:
            next_state["completed_snapshot_revision"] = revision
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=next_state["candidate_count"],
            authoritative_snapshot=complete and len(data_paths) == 1,
        )

    def _arxiv_titles(self, identifiers: Sequence[str]) -> dict[str, str]:
        titles: dict[str, str] = {}
        for start in range(0, len(identifiers), self.arxiv_batch_size):
            if start:
                self.pause(self.arxiv_delay_seconds)
            batch = identifiers[start : start + self.arxiv_batch_size]
            response: HttpResponse = self.client.get(
                self.arxiv_api_url,
                params={"id_list": ",".join(batch)},
                headers={"Accept": "application/atom+xml"},
            )
            titles.update(_parse_arxiv_titles(response.body, self.name))
        return titles

    def _record(
        self, candidate: Mapping[str, Any], revision: str, verified_title: str | None
    ) -> SourceRecord:
        method_url = candidate["method_url"]
        display_name = candidate["full_name"]
        short_name = candidate["name"]
        arxiv_id = _text(candidate.get("arxiv_id")) or None
        aliases = (short_name,) if short_name != display_name else ()
        local_id = f"pwc-method:{content_hash(method_url)[:32]}"
        links = []
        if candidate["source_url"]:
            links.append(
                Link(
                    candidate["source_url"],
                    relation="source_paper",
                    locator="$.source_url",
                    crawl=False,
                )
            )
        links.append(
            Link(
                candidate["paper_url"],
                relation="associated_paper",
                locator="$.paper.url",
                crawl=False,
            )
        )
        if candidate["code_snippet_url"]:
            links.append(
                Link(
                    candidate["code_snippet_url"],
                    relation="code_reference",
                    locator="$.code_snippet_url",
                    crawl=False,
                )
            )
        return SourceRecord(
            # The method URL identifies the method, while this source record
            # represents its particular paper relationship. Include both so a
            # repeated method URL with another linked paper cannot overwrite or
            # collapse that provenance across pages or shards.
            source_record_id=(
                "method-paper:"
                + content_hash(
                    {"method_url": method_url, "paper_url": candidate["paper_url"]}
                )[:32]
            ),
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=method_url,
            title=display_name,
            raw={
                "dataset": self.dataset_id,
                "snapshot_revision": revision,
                "license": self.license,
                "admission": self.admission,
                "method_url": method_url,
                "name": short_name,
                "full_name": display_name,
                "paper_url": candidate["paper_url"],
                "paper_title": candidate["paper_title"],
                "source_url": candidate["source_url"],
                "source_title": candidate["source_title"],
                "arxiv_id": arxiv_id,
                "code_snippet_url": candidate["code_snippet_url"],
                "verified_arxiv_title": verified_title,
                "num_papers": candidate["num_papers"],
                "collections": candidate["collections"],
            },
            identifiers=tuple(
                identifier
                for identifier in (
                    Identifier("paperswithcode:method", method_url),
                    Identifier("arxiv", arxiv_id) if arxiv_id else None,
                )
                if identifier is not None
            ),
            links=tuple(links),
            models=(
                ModelHint(
                    local_id=local_id,
                    name=display_name,
                    identifiers=(Identifier("paperswithcode:method", method_url),),
                    aliases=aliases,
                    status=(
                        ModelStatus.DOCUMENTED
                        if self.admission == "arxiv_verified"
                        else ModelStatus.CANDIDATE
                    ),
                    confidence=(
                        0.8
                        if self.admission == "arxiv_verified"
                        else 0.4
                        if self.admission == "candidate"
                        else 0.3
                        if self.admission == "paper_linked_candidate"
                        else 0.2
                    ),
                    locator="$.full_name",
                ),
            ),
        )


class PapersWithCodeEvaluationMethodsSourceAdapter:
    """Retain source-backed model labels from PWC evaluation tables as candidates.

    Evaluation tables provide a broad historical plane that is distinct from PWC's
    method archive: each retained label is attached to a task, paper, and (when
    supplied) implementation link.  Table rows can still contain aliases, ablations,
    or malformed content, so this adapter deliberately emits *candidate* identities
    only.  It never asserts an official implementation or a release.
    """

    artifact_source: str
    disable_derived_extraction = True

    def __init__(
        self,
        *,
        name: str = "paperswithcode-evaluation-methods",
        dataset_id: str = "pwc-archive/evaluation-tables",
        metadata_url: str = "https://huggingface.co/api/datasets/pwc-archive/evaluation-tables",
        license: str = "CC-BY-SA-4.0",
        max_dataset_bytes: int = 80 * 1024 * 1024,
        max_model_rows_per_shard: int = 200_000,
        client: HttpClient | Any | None = None,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.artifact_source = self.name
        self.dataset_id = _required_text(dataset_id, "dataset ID")
        self.metadata_url = _required_web_url(metadata_url, "metadata URL")
        self.license = _required_text(license, "license")
        if max_dataset_bytes < 1:
            raise ValueError("max_dataset_bytes must be positive")
        if max_model_rows_per_shard < 1:
            raise ValueError("max_model_rows_per_shard must be positive")
        self.max_dataset_bytes = int(max_dataset_bytes)
        self.max_model_rows_per_shard = int(max_model_rows_per_shard)
        self.client = client or HttpClient(max_response_bytes=self.max_dataset_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "paperswithcode-evaluation-methods-v1",
                "dataset_id": self.dataset_id,
                "metadata_url": self.metadata_url,
                "license": self.license,
                "policy": "paper-linked-phone-free-evaluation-label-candidates",
                "max_model_rows_per_shard": self.max_model_rows_per_shard,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        prior_revision = _text(state.get("snapshot_revision"))
        prior_paths = _safe_data_paths(state.get("data_paths"))
        shard_index = _nonnegative_int(state.get("shard_index", 0))
        checked_at = _isoformat(self.clock())

        # An unfinished snapshot is immutable, so resume it without repeatedly
        # requesting its manifest. Once it completes, check the manifest again on
        # the next run to discover a newer snapshot.
        if prior_revision and prior_paths and not _text(state.get("completed_snapshot_revision")):
            revision = prior_revision
            data_paths = prior_paths
        else:
            metadata_response: HttpResponse = self.client.get(
                self.metadata_url, headers={"Accept": "application/json"}
            )
            metadata = metadata_response.json()
            if not isinstance(metadata, Mapping):
                raise ValueError(f"{self.name}: dataset metadata is not an object")
            revision, data_paths = _evaluation_snapshot(metadata, self.dataset_id, self.name)
            if revision == _text(state.get("completed_snapshot_revision")):
                return SourcePage(
                    records=(),
                    next_state={
                        "snapshot_revision": revision,
                        "completed_snapshot_revision": revision,
                        "data_paths": list(data_paths),
                        "checked_at": checked_at,
                    },
                    complete=True,
                    authoritative_snapshot=False,
                )
            shard_index = 0

        if shard_index >= len(data_paths):
            raise ValueError(f"{self.name}: shard checkpoint is outside the snapshot")
        data_path = data_paths[shard_index]
        data_url = _resolve_url(self.dataset_id, revision, data_path)
        data_response: HttpResponse = self.client.get(
            data_url, headers={"Accept": "application/vnd.apache.parquet"}
        )
        if len(data_response.body) > self.max_dataset_bytes:
            raise ValueError(f"{self.name}: dataset shard exceeded configured byte limit")
        rows = _read_evaluation_rows(data_response.body, self.name)
        records, rejected, raw_model_rows = _evaluation_records(
            rows,
            revision=revision,
            data_path=data_path,
            dataset_id=self.dataset_id,
            license=self.license,
            max_model_rows=self.max_model_rows_per_shard,
        )
        next_shard = shard_index + 1
        candidate_count = _nonnegative_int(state.get("candidate_count", 0)) + len(records)
        raw_model_count = _nonnegative_int(state.get("raw_model_rows", 0)) + raw_model_rows
        next_state: dict[str, Any] = {
            "snapshot_revision": revision,
            "data_paths": list(data_paths),
            "shard_index": next_shard,
            "checked_at": checked_at,
            "dataset_url": data_url,
            "candidate_count": candidate_count,
            "raw_model_rows": raw_model_count,
            "rejected": rejected,
        }
        complete = next_shard == len(data_paths)
        if complete:
            next_state["completed_snapshot_revision"] = revision
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=candidate_count,
            # Each shard is immutable, but a multi-shard scan cannot infer a
            # disappearance until a separate complete reconciliation pass exists.
            authoritative_snapshot=False,
        )


def _read_rows(body: bytes, source: str) -> list[Mapping[str, Any]]:
    try:
        table = pq.read_table(pa.BufferReader(body))
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ValueError(f"{source}: response is not a readable Parquet table") from error
    missing = sorted(_REQUIRED_COLUMNS - set(table.column_names))
    if missing:
        raise ValueError(
            f"{source}: Parquet schema is missing required column(s): {', '.join(missing)}"
        )
    return table.select(sorted(_REQUIRED_COLUMNS)).to_pylist()


_METHOD_REQUIRED_COLUMNS = frozenset(
    {
        "url",
        "name",
        "full_name",
        "paper",
        "source_url",
        "source_title",
        "num_papers",
        "collections",
    }
)


def _read_method_rows(body: bytes, source: str) -> list[Mapping[str, Any]]:
    try:
        table = pq.read_table(pa.BufferReader(body))
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ValueError(f"{source}: response is not a readable Parquet table") from error
    missing = sorted(_METHOD_REQUIRED_COLUMNS - set(table.column_names))
    if missing:
        raise ValueError(
            f"{source}: Parquet schema is missing required column(s): {', '.join(missing)}"
        )
    columns = sorted(_METHOD_REQUIRED_COLUMNS | {"description", "code_snippet_url"})
    available = [column for column in columns if column in table.column_names]
    return table.select(available).to_pylist()


def _read_evaluation_rows(body: bytes, source: str) -> list[Mapping[str, Any]]:
    try:
        table = pq.read_table(pa.BufferReader(body), columns=["task", "subtasks", "datasets"])
    except (pa.ArrowException, OSError, ValueError) as error:
        raise ValueError(f"{source}: response is not a readable Parquet table") from error
    required = {"task", "subtasks", "datasets"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(
            f"{source}: Parquet schema is missing required column(s): {', '.join(missing)}"
        )
    return table.to_pylist()


def _evaluation_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    revision: str,
    data_path: str,
    dataset_id: str,
    license: str,
    max_model_rows: int,
) -> tuple[list[SourceRecord], dict[str, int], int]:
    """Collapse repeated table entries into paper-backed candidate observations."""

    # A model may be evaluated in several tables under one paper. Preserve the
    # task and dataset scope so those exact evaluation identities do not collapse.
    candidates: dict[
        tuple[str, str, tuple[str, ...], tuple[str, ...]], dict[str, Any]
    ] = {}
    rejected: dict[str, int] = {}
    raw_model_rows = 0

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for root in rows:
        task = _evaluation_task_name(root)
        initial_task_path = (task,) if task else ()
        pending: list[tuple[Any, tuple[str, ...], tuple[str, ...]]] = [
            (subtask, (), initial_task_path + ((label,) if label else ()))
            for subtask in _dataset_items(root.get("subtasks"))
            for label in [_evaluation_task_name(subtask)]
        ] + [
            *(
                (dataset, (label,) if label else (), initial_task_path)
                for dataset in _dataset_items(root.get("datasets"))
                for label in [_evaluation_dataset_name(dataset)]
            ),
        ]
        while pending:
            value, dataset_scope, task_path = pending.pop()
            if isinstance(value, Mapping):
                name = _text(value.get("model_name"))
                if name:
                    raw_model_rows += 1
                    if raw_model_rows > max_model_rows:
                        raise ValueError("evaluation table exceeded max_model_rows_per_shard")
                    if (
                        len(name) > 240
                        or _CONTROL_CHARACTER.search(name)
                        or _PHONE_NUMBER.search(name)
                        or not _TOKEN.search(name)
                    ):
                        reject("invalid_model_name")
                    else:
                        paper_url = _optional_web_url(value.get("paper_url"))
                        paper_title = _text(value.get("paper_title"))
                        if not paper_url or not paper_title:
                            reject("missing_paper_provenance")
                        else:
                            key = (
                                name.casefold(),
                                paper_url,
                                tuple(part.casefold() for part in task_path),
                                dataset_scope,
                            )
                            entry = candidates.setdefault(
                                key,
                                {
                                    "name": name,
                                    "paper_url": paper_url,
                                    "paper_title": paper_title,
                                    "tasks": task_path,
                                    "datasets": dataset_scope,
                                    "code_links": set(),
                                    "model_links": set(),
                                    "model_link_titles": {},
                                },
                            )
                            for code in _evaluation_code_urls(value.get("code_links")):
                                entry["code_links"].add(code)
                            for model_url in _evaluation_code_urls(value.get("model_links")):
                                entry["model_links"].add(model_url)
                            for model_url, label in _evaluation_model_link_titles(
                                value.get("model_links")
                            ):
                                entry["model_link_titles"].setdefault(model_url, set()).add(label)
                # The metrics object can contain tens of thousands of metric keys;
                # model observations live only on this declared structural path.
                pending.extend(
                    (value.get(key), dataset_scope, task_path)
                    for key in ("sota", "rows")
                )
                pending.extend(
                    (
                        subtask,
                        dataset_scope,
                        task_path + ((label,) if label else ()),
                    )
                    for subtask in _dataset_items(value.get("subtasks"))
                    for label in [_evaluation_task_name(subtask)]
                )
                pending.extend(
                    (
                        dataset,
                        dataset_scope + ((label,) if label else ()),
                        task_path,
                    )
                    for dataset in _dataset_items(value.get("datasets"))
                    for label in [_evaluation_dataset_name(dataset)]
                )
                pending.extend(
                    (
                        dataset,
                        dataset_scope + ((label,) if label else ()),
                        task_path,
                    )
                    for dataset in _dataset_items(value.get("subdatasets"))
                    for label in [_evaluation_dataset_name(dataset)]
                )
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                pending.extend((item, dataset_scope, task_path) for item in value)

    records: list[SourceRecord] = []
    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: (
            item["name"].casefold(),
            item["paper_url"],
            tuple(part.casefold() for part in item["tasks"]),
            item["datasets"],
        ),
    )
    for entry in ordered_candidates:
        identity = content_hash(
            {
                "name": entry["name"],
                "paper_url": entry["paper_url"],
                "tasks": entry["tasks"],
                "datasets": entry["datasets"],
            }
        )[:32]
        identifier = Identifier("paperswithcode:evaluation-result", identity)
        links = [
            Link(
                entry["paper_url"],
                relation="evaluated_in",
                locator="$.datasets[].sota.rows[].paper_url",
                crawl=False,
            )
        ]
        links.extend(
            Link(
                code_url,
                relation="reported_implementation",
                locator="$.datasets[].sota.rows[].code_links[].url",
                crawl=False,
            )
            for code_url in sorted(entry["code_links"])
        )
        links.extend(
            Link(
                model_url,
                relation="model_artifact",
                locator="$.datasets[].sota.rows[].model_links[].url",
                crawl=False,
                model_local_ids=(
                    ("pwc-linked-model:" + content_hash(model_identifier.value)[:32],)
                    if (model_identifier := identifier_from_url(model_url))
                    and model_identifier.namespace == "huggingface:model"
                    else ()
                ),
            )
            for model_url in sorted(entry["model_links"])
        )
        local_id = f"pwc-evaluation:{identity}"
        linked_model_hints: dict[str, ModelHint] = {}
        for model_url in sorted(entry["model_links"]):
            model_identifier = identifier_from_url(model_url)
            if not model_identifier or model_identifier.namespace != "huggingface:model":
                continue
            linked_local_id = "pwc-linked-model:" + content_hash(model_identifier.value)[:32]
            linked_model_hints.setdefault(
                linked_local_id,
                ModelHint(
                    local_id=linked_local_id,
                    name=(
                        sorted(entry["model_link_titles"].get(model_url, set()))[0]
                        if entry["model_link_titles"].get(model_url)
                        else model_identifier.value
                    ),
                    identifiers=(model_identifier,),
                    status=ModelStatus.CANDIDATE,
                    confidence=0.65,
                    locator="$.datasets[].sota.rows[].model_links[].url",
                ),
            )
        records.append(
            SourceRecord(
                source_record_id=f"evaluation-result:{identity}",
                kind=ArtifactKind.CATALOG_RECORD,
                canonical_url=entry["paper_url"],
                title=entry["name"],
                raw={
                    "dataset": dataset_id,
                    "snapshot_revision": revision,
                    "data_path": data_path,
                    "license": license,
                    "admission": "paper_linked_evaluation_candidate",
                    "model_name": entry["name"],
                    "paper_url": entry["paper_url"],
                    "paper_title": entry["paper_title"],
                    "tasks": list(entry["tasks"]),
                    "datasets": list(entry["datasets"]),
                    "code_urls": sorted(entry["code_links"]),
                    "model_artifact_urls": sorted(entry["model_links"]),
                    "model_artifacts": [
                        {
                            "url": model_url,
                            "titles": sorted(entry["model_link_titles"].get(model_url, set())),
                        }
                        for model_url in sorted(entry["model_links"])
                    ],
                },
                identifiers=(identifier,),
                links=tuple(links),
                models=(
                    ModelHint(
                        local_id=local_id,
                        name=entry["name"],
                        identifiers=(identifier,),
                        status=ModelStatus.CANDIDATE,
                        confidence=0.35,
                        locator="$.datasets[].sota.rows[].model_name",
                    ),
                    *linked_model_hints.values(),
                ),
            )
        )
    return records, dict(sorted(rejected.items())), raw_model_rows


def _dataset_items(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _evaluation_dataset_name(value: Any) -> str:
    if not isinstance(value, Mapping):
        label = _text(value)
        return label if len(label) <= 240 and not _CONTROL_CHARACTER.search(label) else ""
    for key in ("dataset_name", "full_name", "name"):
        if label := _text(value.get(key)):
            return label if len(label) <= 240 and not _CONTROL_CHARACTER.search(label) else ""
    nested = value.get("dataset")
    if isinstance(nested, Mapping):
        for key in ("dataset_name", "full_name", "name"):
            if label := _text(nested.get(key)):
                return label if len(label) <= 240 and not _CONTROL_CHARACTER.search(label) else ""
    label = _text(nested)
    return label if len(label) <= 240 and not _CONTROL_CHARACTER.search(label) else ""


def _evaluation_task_name(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    label = _text(value.get("task"))
    return label if len(label) <= 240 and not _CONTROL_CHARACTER.search(label) else ""


def _method_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_title_overlap: bool,
    require_arxiv_source: bool,
    allow_missing_source: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for row in rows:
        method_url = _text(row.get("url"))
        name = _text(row.get("name"))
        full_name = _text(row.get("full_name")) or name
        description = _text(row.get("description"))
        paper = row.get("paper")
        source_url = _text(row.get("source_url"))
        source_title = _text(row.get("source_title"))
        if (
            not method_url
            or _METHOD_URL.fullmatch(method_url) is None
            or not name
            or not full_name
            or len(name) > 160
            or len(full_name) > 240
            or not isinstance(paper, Mapping)
            or (not allow_missing_source and (not source_url or not source_title))
        ):
            reject("invalid_method_shape")
            continue
        if _PHONE_NUMBER.search(" ".join((name, full_name, description))):
            reject("phone_payload")
            continue
        paper_url = _text(paper.get("url"))
        paper_title = _text(paper.get("title"))
        arxiv_id = _arxiv_id_from_url(source_url) if source_url else None
        if not paper_url or not paper_title:
            reject("missing_linked_paper")
            continue
        if require_arxiv_source and arxiv_id is None:
            reject("missing_linked_arxiv_paper")
            continue
        shared = _meaningful_tokens(full_name) & _meaningful_tokens(
            f"{paper_title} {source_title}"
        )
        if require_title_overlap and len(shared) < 2:
            reject("insufficient_method_title_overlap")
            continue
        candidates.append(
            {
                "method_url": canonicalize_url(method_url),
                "name": name,
                "full_name": full_name,
                "paper_url": _required_web_url(paper_url, "method paper URL"),
                "paper_title": paper_title,
                "source_url": _required_web_url(source_url, "method source URL")
                if source_url
                else None,
                "source_title": source_title or None,
                "arxiv_id": arxiv_id,
                "code_snippet_url": _optional_web_url(row.get("code_snippet_url")),
                "num_papers": str(row.get("num_papers") or ""),
                "collections": str(row.get("collections") or ""),
            }
        )
    # Duplicate archive rows can occur within a shard and are also stable across
    # shards. Suppress exact repeats here; shard-level repeats retain the same
    # deterministic method-paper source identity during downstream upsert.
    unique_candidates: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        unique_candidates.setdefault(content_hash(candidate), candidate)
    return list(unique_candidates.values()), rejected


def _arxiv_id_from_url(value: str) -> str | None:
    parts = urlsplit(value)
    if parts.hostname not in {"arxiv.org", "www.arxiv.org"}:
        return None
    segments = [segment for segment in parts.path.split("/") if segment]
    if len(segments) < 2 or segments[0] not in {"abs", "pdf"}:
        return None
    return _arxiv_id(segments[1].removesuffix(".pdf"))


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN.findall(value)
        if len(token) > 1 and token.casefold() not in _TITLE_STOPWORDS
    }


def _titles_substantially_match(left: str, right: str) -> bool:
    left_tokens = _meaningful_tokens(left)
    right_tokens = _meaningful_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens)
    return (2 * overlap) >= len(left_tokens) + len(right_tokens)


def _parse_arxiv_titles(body: bytes, source: str) -> dict[str, str]:
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as error:
        raise ValueError(f"{source}: arXiv response is not valid Atom XML") from error
    namespace = "{http://www.w3.org/2005/Atom}"
    titles: dict[str, str] = {}
    for entry in root.findall(f"{namespace}entry"):
        raw_identifier = _xml_text(entry.find(f"{namespace}id"))
        identifier = _arxiv_id_from_url(raw_identifier) or _arxiv_id(raw_identifier)
        title = " ".join(_xml_text(entry.find(f"{namespace}title")).split())
        if identifier is not None and title:
            titles[identifier] = title
    return titles


def _xml_text(element: ElementTree.Element[str] | None) -> str:
    return element.text if element is not None and element.text is not None else ""


def _official_page(
    rows: Sequence[Mapping[str, Any]], start: int, page_size: int
) -> tuple[list[tuple[int, Mapping[str, Any]]], int, bool]:
    if start > len(rows):
        raise ValueError("PWC row checkpoint is outside the current snapshot")
    selected: list[tuple[int, Mapping[str, Any]]] = []
    for index in range(start, len(rows)):
        row = rows[index]
        if row.get("is_official") is not True:
            continue
        selected.append((index, row))
        if len(selected) >= page_size:
            return selected, index + 1, False
    return selected, len(rows), True


def _snapshot_revision(
    metadata: Mapping[str, Any], dataset_id: str, data_path: str, source: str
) -> str:
    returned_id = _text(metadata.get("id"))
    if returned_id and returned_id != dataset_id:
        raise ValueError(f"{source}: metadata dataset ID does not match configuration")
    revision = _text(metadata.get("sha"))
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError(f"{source}: metadata is missing a valid immutable revision")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes, bytearray)):
        raise ValueError(f"{source}: metadata siblings are invalid")
    if data_path not in {
        _text(item.get("rfilename"))
        for item in siblings
        if isinstance(item, Mapping)
    }:
        raise ValueError(f"{source}: metadata does not declare configured Parquet path")
    return revision


def _snapshot_data_paths(
    metadata: Mapping[str, Any], dataset_id: str, configured_path: str, source: str
) -> tuple[str, tuple[str, ...]]:
    """Discover every shard in the configured train-file family at one revision."""
    returned_id = _text(metadata.get("id"))
    if returned_id and returned_id != dataset_id:
        raise ValueError(f"{source}: metadata dataset ID does not match configuration")
    revision = _text(metadata.get("sha"))
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError(f"{source}: metadata is missing a valid immutable revision")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes, bytearray)):
        raise ValueError(f"{source}: metadata siblings are invalid")
    paths = {
        _safe_data_path(_text(item.get("rfilename")))
        for item in siblings
        if isinstance(item, Mapping)
        and _text(item.get("rfilename")).startswith("data/")
        and _text(item.get("rfilename")).endswith(".parquet")
    }
    if configured_path in paths:
        return revision, (configured_path,)

    match = re.fullmatch(r"(.*)-\d{5}-of-\d{5}(\.parquet)", configured_path)
    if match is None:
        raise ValueError(f"{source}: metadata does not declare configured Parquet path")
    prefix, suffix = match.groups()
    pattern = re.compile(re.escape(prefix) + r"-(\d{5})-of-(\d{5})" + re.escape(suffix))
    shards: list[tuple[int, int, str]] = []
    for path in paths:
        shard_match = pattern.fullmatch(path)
        if shard_match:
            shards.append((int(shard_match.group(1)), int(shard_match.group(2)), path))
    if not shards:
        raise ValueError(f"{source}: metadata does not declare configured Parquet path")
    totals = {total for _, total, _ in shards}
    indices = {index for index, _, _ in shards}
    declared_total = next(iter(totals))
    if (
        len(totals) != 1
        or len(shards) != declared_total
        or indices != set(range(declared_total))
    ):
        raise ValueError(f"{source}: metadata declares an incomplete Parquet shard set")
    return revision, tuple(path for _, _, path in sorted(shards))


def _merge_counts(prior: Any, current: Mapping[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    if isinstance(prior, Mapping):
        for key, value in prior.items():
            if isinstance(key, str) and isinstance(value, int) and value >= 0:
                merged[key] = value
    for key, value in current.items():
        merged[key] = merged.get(key, 0) + value
    return dict(sorted(merged.items()))


def _evaluation_snapshot(
    metadata: Mapping[str, Any], dataset_id: str, source: str
) -> tuple[str, tuple[str, ...]]:
    returned_id = _text(metadata.get("id"))
    if returned_id and returned_id != dataset_id:
        raise ValueError(f"{source}: metadata dataset ID does not match configuration")
    revision = _text(metadata.get("sha"))
    if _COMMIT.fullmatch(revision) is None:
        raise ValueError(f"{source}: metadata is missing a valid immutable revision")
    siblings = metadata.get("siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes, bytearray)):
        raise ValueError(f"{source}: metadata siblings are invalid")
    paths = tuple(
        sorted(
            _safe_data_path(_text(item.get("rfilename")))
            for item in siblings
            if isinstance(item, Mapping)
            and _text(item.get("rfilename")).startswith("data/")
            and _text(item.get("rfilename")).endswith(".parquet")
        )
    )
    if not paths:
        raise ValueError(f"{source}: metadata declares no evaluation-table Parquet shards")
    return revision, paths


def _resolve_url(dataset_id: str, revision: str, data_path: str) -> str:
    encoded_id = quote(dataset_id, safe="/")
    return canonicalize_url(
        "https://huggingface.co/datasets/"
        f"{encoded_id}/resolve/{revision}/{quote(data_path, safe='/')}"
    )


def _repository_url(value: Any) -> str:
    url = _required_web_url(value, "repository URL")
    parts = urlsplit(url)
    segments = [segment for segment in parts.path.split("/") if segment]
    if parts.hostname and parts.hostname.casefold() in {"github.com", "www.github.com"}:
        if len(segments) < 2:
            raise ValueError("repository URL does not contain owner and repository")
        return canonicalize_url(
            f"https://github.com/{segments[0]}/{segments[1].removesuffix('.git')}"
        )
    return url


def _repository_title(value: str) -> str:
    parts = urlsplit(value)
    path = "/".join(segment for segment in parts.path.split("/") if segment)
    return path or value


def _arxiv_id(value: Any) -> str | None:
    result = _text(value)
    if not result or _ARXIV_ID.fullmatch(result) is None:
        return None
    return re.sub(r"v\d+$", "", result, flags=re.IGNORECASE)


def _required_web_url(value: Any, field: str) -> str:
    text = _required_text(value, field)
    canonical = canonicalize_url(text)
    parts = urlsplit(canonical)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError(f"{field} must be an absolute HTTP(S) URL")
    return canonical


def _safe_data_path(value: str) -> str:
    path = _required_text(value, "dataset data path")
    if path.startswith("/") or ".." in path.split("/"):
        raise ValueError("dataset data path must be a relative file path")
    return path


def _safe_data_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    try:
        paths = tuple(_safe_data_path(_text(item)) for item in value)
    except ValueError:
        return ()
    return paths if paths and len(paths) == len(set(paths)) else ()


def _optional_web_url(value: Any) -> str | None:
    try:
        return _required_web_url(value, "evaluation paper URL")
    except ValueError:
        return None


def _evaluation_code_urls(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    urls = {
        url
        for item in value
        if isinstance(item, Mapping)
        if (url := _optional_web_url(item.get("url")))
    }
    return tuple(sorted(urls))


def _evaluation_model_link_titles(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    pairs = {
        (url, title)
        for item in value
        if isinstance(item, Mapping)
        if (url := _optional_web_url(item.get("url")))
        if (title := _text(item.get("title")))
        if len(title) <= 240 and not _CONTROL_CHARACTER.search(title)
    }
    return tuple(sorted(pairs))


def _required_text(value: Any, field: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError("PWC row checkpoint must be an integer") from None
    if result < 0:
        raise ValueError("PWC row checkpoint must be non-negative")
    return result


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
