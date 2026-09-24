"""Pinned ingestion for ESPnet's first-party Model Zoo CSV."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

Clock = Callable[[], datetime]

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RECIPE_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_HUGGING_FACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_EXPECTED_HEADERS = (
    "corpus",
    "task",
    "name",
    "url",
    "fs",
    "lang",
    "gender",
    "pytorch",
    "espnet",
    "commit",
    "valid",
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _ZooRow:
    corpus: str
    task: str
    name: str
    url: str
    declared_url: str
    sample_rate: str | None
    language: str | None
    gender: str | None
    pytorch_version: str | None
    espnet_version: str | None
    recipe_commit: str | None
    validated: bool
    locator: str


class EspnetModelZooSourceAdapter:
    """Read every structurally valid ESPnet Model Zoo row at one Git commit.

    ESPnet itself uses this CSV to power its public model query and download
    commands. The adapter retains both currently validated and invalidated rows:
    a false ``valid`` flag is status evidence, not a reason to erase a model from
    the historical registry. The table's leading control row has no corpus/task
    and is rejected rather than becoming a fictitious model.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers structurally valid entries in ESPnet Model Zoo's public table.csv "
        "at one observed commit. It does not enumerate models absent from that "
        "table, verify that a linked Zenodo or Hugging Face artifact still resolves, "
        "infer papers, or transfer model archives."
    )

    def __init__(
        self,
        *,
        name: str = "espnet-model-zoo",
        repository: str = "espnet/espnet_model_zoo",
        branch: str = "master",
        table_path: str = "espnet_model_zoo/table.csv",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_rows: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.table_path = _safe_path(table_path)
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_rows = _positive_int(max_rows, "max_rows")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "espnet-model-zoo-v1",
                "repository": self.repository,
                "branch": self.branch,
                "table_path": self.table_path,
                "max_response_bytes": self.max_response_bytes,
                "max_rows": self.max_rows,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.table_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.table_path, safe='/')}"
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
            headers={"Accept": "text/csv,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: table returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: table exceeds {self.max_response_bytes} bytes")
        rows, control_rows = _parse_rows(response.text(), self.name, self.max_rows)
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: table contains no model rows")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "table_url": self.raw_url(revision),
            "table_sha256": content_hash(response.body),
            "model_count": len(records),
            "validated_model_count": sum(int(row.validated) for row in rows),
            "invalidated_model_count": sum(int(not row.validated) for row in rows),
            "skipped_control_rows": control_rows,
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

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(self, row: _ZooRow, revision: str, table: bytes) -> SourceRecord:
        model_identifier = Identifier("espnet:model", row.name)
        model = ModelHint(
            local_id=f"model:{row.name}",
            name=row.name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=row.locator,
        )
        table_url = self.blob_url(revision)
        links = [
            Link(table_url, relation="model_card", locator=row.locator, crawl=False),
            Link(self.repository_url, relation="source_repository", crawl=False),
            Link("https://github.com/espnet/espnet", relation="source_implementation", crawl=False),
        ]
        if row.recipe_commit and _RECIPE_COMMIT.fullmatch(row.recipe_commit):
            links.append(
                Link(
                    f"https://github.com/espnet/espnet/commit/{row.recipe_commit}",
                    relation="source_revision",
                    locator=row.locator,
                    crawl=False,
                )
            )
        if hub_id := _hugging_face_id(row.url, row.name):
            links.append(
                Link(
                    f"https://huggingface.co/{hub_id}",
                    relation="linked_model_artifact",
                    locator=row.locator,
                    crawl=False,
                )
            )
        else:
            links.append(Link(row.url, relation="model_artifact", locator=row.locator, crawl=False))
        release = ReleaseHint(
            local_id=f"release:{row.name}",
            model_local_id=model.local_id,
            version=row.espnet_version,
            revision=revision,
            identifiers=(Identifier("espnet:model-release", row.name),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "corpus": row.corpus,
                "task": row.task,
                "source_url": row.url,
                "declared_url": row.declared_url,
                "sample_rate_hz": row.sample_rate,
                "language": row.language,
                "gender": row.gender,
                "pytorch_version": row.pytorch_version,
                "espnet_version": row.espnet_version,
                "recipe_commit": row.recipe_commit,
                "validated_by_espnet": row.validated,
            },
            locator=row.locator,
        )
        return SourceRecord(
            source_record_id=f"model:{row.name}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(table_url),
            title=row.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "table_path": self.table_path,
                "table_sha256": content_hash(table),
                "corpus": row.corpus,
                "task": row.task,
                "url": row.url,
                "declared_url": row.declared_url,
                "sample_rate_hz": row.sample_rate,
                "language": row.language,
                "gender": row.gender,
                "pytorch_version": row.pytorch_version,
                "espnet_version": row.espnet_version,
                "recipe_commit": row.recipe_commit,
                "validated_by_espnet": row.validated,
            },
            text="\n".join(
                part
                for part in (
                    row.name,
                    f"task: {row.task}",
                    f"corpus: {row.corpus}",
                    f"language: {row.language}" if row.language else "",
                    f"sample rate: {row.sample_rate}" if row.sample_rate else "",
                    "ESPnet validation: " + ("valid" if row.validated else "invalidated"),
                )
                if part
            ),
            identifiers=(model_identifier,),
            links=tuple(dict.fromkeys(links)),
            models=(model,),
            releases=(release,),
        )


def _parse_rows(document: str, source: str, maximum: int) -> tuple[tuple[_ZooRow, ...], int]:
    try:
        reader = csv.DictReader(io.StringIO(document))
    except csv.Error as error:
        raise ValueError(f"{source}: malformed CSV: {error}") from error
    headers = tuple(reader.fieldnames or ())
    if headers != _EXPECTED_HEADERS:
        raise ValueError(f"{source}: table headers changed or are incomplete")
    rows = []
    controls = 0
    seen_names: set[str] = set()
    for line_number, raw in enumerate(reader, start=2):
        if raw is None:
            continue
        if None in raw:
            raise ValueError(f"{source}: row {line_number} has more fields than its header")
        corpus = _text(raw.get("corpus"))
        task = _text(raw.get("task"))
        name = _text(raw.get("name"))
        declared_url = _text(raw.get("url"))
        url = _model_url(declared_url, source, f"row {line_number} URL")
        if not corpus and not task:
            controls += 1
            continue
        if not corpus or not task or not name:
            raise ValueError(f"{source}: row {line_number} lacks corpus, task, or model name")
        if name in seen_names:
            raise ValueError(f"{source}: duplicate Model Zoo model name {name!r}")
        seen_names.add(name)
        rows.append(
            _ZooRow(
                corpus=corpus,
                task=task,
                name=name,
                url=url,
                declared_url=declared_url,
                sample_rate=_text(raw.get("fs")) or None,
                language=_text(raw.get("lang")) or None,
                gender=_text(raw.get("gender")) or None,
                pytorch_version=_text(raw.get("pytorch")) or None,
                espnet_version=_text(raw.get("espnet")) or None,
                recipe_commit=_text(raw.get("commit")) or None,
                validated=_boolean(raw.get("valid"), source, line_number),
                locator=f"{source}:line:{line_number}",
            )
        )
    if len(rows) > maximum:
        raise ValueError(f"{source}: table exceeds {maximum} model rows")
    return tuple(rows), controls


def _hugging_face_id(url: str, name: str) -> str | None:
    parts = urlsplit(url)
    if parts.hostname not in {"huggingface.co", "www.huggingface.co"}:
        return None
    path = parts.path.strip("/")
    candidate = path if path else name
    return candidate if _HUGGING_FACE_ID.fullmatch(candidate) else None


def _model_url(url: str, source: str, field: str) -> str:
    # ESPnet's table has historically used this exact bare-host spelling as the
    # documented Hugging Face base URL. Normalize only that explicit shorthand;
    # do not invent schemes for arbitrary malformed source values.
    if url.rstrip("/").casefold() == "huggingface.co":
        return "https://huggingface.co/"
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"{source}: {field} is not an HTTP(S) URL")
    return url


def _boolean(value: Any, source: str, line_number: int) -> bool:
    text = _text(value).casefold()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{source}: row {line_number} valid field must be true or false")


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _safe_path(value: str) -> str:
    path = _required_text(value, "table path")
    invalid_part = any(part in {"", ".", ".."} for part in path.split("/"))
    if path.startswith("/") or "\\" in path or invalid_part:
        raise ValueError("table path is not a safe relative path")
    return path


def _required_text(value: Any, field: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], key: str) -> str:
    value = headers.get(key) or headers.get(key.casefold()) or headers.get(key.title())
    return _text(value)


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
