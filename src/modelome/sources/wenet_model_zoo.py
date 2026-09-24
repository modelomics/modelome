"""Pinned ingestion of WeNet's first-party pretrained-model table."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

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
_HEADER = ("Datasets", "Languages", "Checkpoint Model", "Runtime Model", "Contributor")
_MARKDOWN_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\((?P<url>https?://[^)\s]+)\)")
_ANY_MARKDOWN_LINK = re.compile(r"\[(?P<label>[^\]]+)\]\([^)]+\)")
_TAG = re.compile(r"<[^>]*>")
_SAFE_NAME = re.compile(r"^[^/\\\x00-\x1f]{1,200}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Artifact:
    kind: str
    label: str
    url: str
    filename: str


@dataclass(frozen=True, slots=True)
class _ModelRow:
    dataset: str
    language: str
    contributor: str
    artifacts: tuple[_Artifact, ...]
    locator: str


class WenetPretrainedModelSourceAdapter:
    """Enumerate explicit checkpoint/runtime archives from WeNet's model list.

    The reader pins `docs/pretrained_models.md` to one Git commit and admits only
    rows from the exact five-column model table. Artifact URLs must use WeNet's
    documented `/downloads?models=wenet&version=<archive>` schema. URLs are
    recorded as references; archive contents are never fetched.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only rows in WeNet's first-party pretrained-model table at one "
        "pinned commit. It does not include models absent from that table, infer "
        "checkpoint metadata, resolve download endpoints, or fetch archive bytes."
    )

    def __init__(
        self,
        *,
        name: str = "wenet-pretrained-models",
        repository: str = "wenet-e2e/wenet",
        branch: str = "main",
        document_path: str = "docs/pretrained_models.md",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_rows: int = 2_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        if self.repository != "wenet-e2e/wenet":
            raise ValueError("repository must be wenet-e2e/wenet")
        self.branch = _required_text(branch, "branch")
        self.document_path = _safe_path(document_path)
        if self.document_path != "docs/pretrained_models.md":
            raise ValueError("document_path must be docs/pretrained_models.md")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_rows = _positive_int(max_rows, "max_rows")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "wenet-pretrained-models-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "max_response_bytes": self.max_response_bytes,
                "max_rows": self.max_rows,
                "url_schema": "https://wenet.org.cn/downloads?models=wenet&version=<archive>",
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
            f"{quote(revision, safe='')}/{quote(self.document_path, safe='/')}"
        )

    def blob_url(self, revision: str) -> str:
        return (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.document_path, safe='/')}"
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
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: model list returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: model list exceeds {self.max_response_bytes} bytes")
        rows = _parse_model_table(
            response.text(), source=self.name, maximum=self.max_rows
        )
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: model list contains no pretrained model rows")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(response.body),
            "model_count": len(records),
            "artifact_count": sum(len(row.artifacts) for row in rows),
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
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(self, row: _ModelRow, revision: str, document: bytes) -> SourceRecord:
        model_name = row.artifacts[0].label
        if model_name.casefold() in {"model", "checkpoint", "download"}:
            model_name = row.dataset
        handle = f"{row.dataset}/{model_name}"
        model_local_id = f"model:{handle}"
        namespace = "wenet:pretrained-model"
        model_identifier = Identifier(namespace, handle)
        model = ModelHint(
            local_id=model_local_id,
            name=model_name,
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=row.locator,
        )
        releases = tuple(
            ReleaseHint(
                local_id=f"release:{handle}:{artifact.kind}",
                model_local_id=model_local_id,
                version=artifact.filename.removesuffix(".tar.gz"),
                revision=revision,
                identifiers=(
                    Identifier(f"{namespace}:{artifact.kind}", artifact.filename),
                ),
                metadata={
                    "repository": self.repository,
                    "revision": revision,
                    "dataset": row.dataset,
                    "language": row.language,
                    "artifact_type": artifact.kind,
                    "archive_filename": artifact.filename,
                    "declared_download_url": artifact.url,
                    "contributor": row.contributor,
                },
                locator=row.locator,
            )
            for artifact in row.artifacts
        )
        links = [
            Link(
                self.blob_url(revision),
                relation="model_card",
                locator=row.locator,
                crawl=False,
                model_local_ids=(model_local_id,),
            ),
            Link(
                self.repository_url,
                relation="source_implementation",
                crawl=False,
                model_local_ids=(model_local_id,),
            ),
        ]
        links.extend(
            Link(
                artifact.url,
                relation="weights" if artifact.kind == "checkpoint" else "inference_artifact",
                locator=row.locator,
                crawl=False,
                model_local_ids=(model_local_id,),
            )
            for artifact in row.artifacts
        )
        return SourceRecord(
            source_record_id=f"model:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(self.blob_url(revision)),
            title=f"{row.dataset} {model_name}",
            raw={
                "repository": self.repository,
                "revision": revision,
                "document_path": self.document_path,
                "document_sha256": content_hash(document),
                "dataset": row.dataset,
                "language": row.language,
                "contributor": row.contributor,
                "artifacts": [
                    {
                        "type": artifact.kind,
                        "label": artifact.label,
                        "filename": artifact.filename,
                        "url": artifact.url,
                    }
                    for artifact in row.artifacts
                ],
            },
            text=f"WeNet pretrained model for {row.dataset} ({row.language})",
            identifiers=(model_identifier,),
            links=tuple(links),
            models=(model,),
            releases=releases,
        )


def _parse_model_table(document: str, *, source: str, maximum: int) -> tuple[_ModelRow, ...]:
    lines = document.splitlines()
    active = False
    found_table = False
    rows: list[_ModelRow] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            active = stripped.lstrip("# ").casefold() == "model list"
            continue
        if not active or not stripped:
            continue
        cells = _cells(stripped)
        if cells == _HEADER:
            found_table = True
            continue
        if found_table and _is_separator(cells):
            continue
        if not found_table:
            continue
        if not cells:
            continue
        if len(cells) != len(_HEADER):
            if rows:
                break
            continue
        dataset = _cell_label(cells[0])
        language = _plain(cells[1])
        contributor = _plain(cells[4])
        if not dataset or not _SAFE_NAME.fullmatch(dataset):
            raise ValueError(f"{source}: invalid dataset/model identity on line {line_number}")
        artifacts = tuple(
            artifact
            for kind, cell in (("checkpoint", cells[2]), ("runtime", cells[3]))
            if (artifact := _artifact(kind, cell, source=source, line_number=line_number))
        )
        if not artifacts:
            continue
        rows.append(
            _ModelRow(
                dataset=dataset,
                language=language,
                contributor=contributor,
                artifacts=artifacts,
                locator=f"docs/pretrained_models.md:L{line_number}",
            )
        )
        if len(rows) > maximum:
            raise ValueError(f"{source}: model list exceeds {maximum} rows")
    if not found_table:
        raise ValueError(f"{source}: exact pretrained model table header was not found")
    handles = [
        (
            row.dataset,
            row.dataset
            if row.artifacts[0].label.casefold() in {"model", "checkpoint", "download"}
            else row.artifacts[0].label,
        )
        for row in rows
    ]
    if len(handles) != len(set(handles)):
        raise ValueError(f"{source}: duplicate dataset/model identity")
    return tuple(rows)


def _artifact(kind: str, cell: str, *, source: str, line_number: int) -> _Artifact | None:
    value = _plain(cell)
    if value.casefold() in {"na", "n/a", "-", ""}:
        return None
    links = tuple(_MARKDOWN_LINK.finditer(cell))
    if len(links) != 1:
        raise ValueError(f"{source}: {kind} cell on line {line_number} must have one link")
    match = links[0]
    label = _plain(match.group("label"))
    parsed = urlsplit(match.group("url"))
    query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "wenet.org.cn"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/downloads"
        or set(query) != {"models", "version"}
        or query["models"] != ["wenet"]
        or len(query["version"]) != 1
    ):
        raise ValueError(f"{source}: {kind} cell on line {line_number} has an invalid WeNet URL")
    filename = query["version"][0]
    if not filename.endswith(".tar.gz") or "/" in filename or not _SAFE_NAME.fullmatch(label):
        raise ValueError(f"{source}: invalid {kind} archive on line {line_number}")
    return _Artifact(kind, label, canonicalize_url(match.group("url")), filename)


def _cells(line: str) -> tuple[str, ...]:
    value = line.strip()
    if not value.startswith("|") and "|" not in value:
        return ()
    value = value.strip("|")
    return tuple(cell.strip() for cell in value.split("|"))


def _is_separator(cells: tuple[str, ...]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _cell_label(cell: str) -> str:
    match = _ANY_MARKDOWN_LINK.search(cell)
    return _plain(match.group("label")) if match else _plain(cell)


def _plain(value: str) -> str:
    value = _ANY_MARKDOWN_LINK.sub(lambda match: match.group("label"), value)
    value = _TAG.sub(" ", value)
    value = unescape(value)
    return re.sub(r"[`*_]", "", value).strip()


def _repository(value: str) -> str:
    repository = _required_text(value, "repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be an owner/name pair")
    return repository


def _safe_path(value: str) -> str:
    path = _required_text(value, "document_path")
    if path.startswith("/") or "\\" in path or ".." in path.split("/"):
        raise ValueError("document_path must be a safe repository-relative path")
    return path


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _header(headers: Mapping[str, Any], name: str) -> str:
    return next(
        (str(value) for key, value in headers.items() if key.casefold() == name.casefold()),
        "",
    )
