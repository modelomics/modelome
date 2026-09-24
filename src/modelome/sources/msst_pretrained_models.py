"""Enumerate the first-party MSST documentation checkpoint manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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

_REPOSITORY = "ZFTurbo/Music-Source-Separation-Training"
_BRANCH = "main"
_DOCUMENT_PATH = "docs/pretrained_models.md"
_COMMIT_URL = f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"
_RAW_PREFIX = f"https://raw.githubusercontent.com/{_REPOSITORY}/"
_REPOSITORY_URL = f"https://github.com/{_REPOSITORY}"
_CHECKPOINT_SUFFIXES = (".ckpt", ".th", ".chpt", ".bin")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"\[(?P<label>[^]]+)\]\((?P<url>https?://[^)]+)\)")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class _Checkpoint:
    filename: str
    url: str
    model_type: str
    instruments: str
    metrics: str
    line_number: int


class MsstPretrainedModelsSourceAdapter:
    """Read checkpoint URLs and model context from MSST's maintained table."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only checkpoint files explicitly linked in docs/pretrained_models.md "
        "at the fetched repository revision. The table includes community-contributed "
        "entries and does not enumerate every checkpoint hosted elsewhere or linked "
        "from the separate MelRoformer experiments table."
    )

    def __init__(
        self,
        *,
        name: str = "msst-pretrained-checkpoints",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_models: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_models", max_models),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_models = max_models
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "msst-pretrained-models-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "document_path": _DOCUMENT_PATH,
                "max_response_bytes": max_response_bytes,
                "max_models": max_models,
                "suffixes": _CHECKPOINT_SUFFIXES,
            }
        )

    @property
    def commit_url(self) -> str:
        return _COMMIT_URL

    def raw_url(self, revision: str) -> str:
        return f"{_RAW_PREFIX}{revision}/{_DOCUMENT_PATH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _SHA1.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        document_url = self.raw_url(revision)
        document_response: HttpResponse = self.client.get(
            document_url, headers={"Accept": "text/plain"}
        )
        if document_response.status != 200:
            raise ValueError(
                f"{self.name}: checkpoint manifest returned HTTP {document_response.status}"
            )
        if len(document_response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: checkpoint manifest exceeds response limit")
        checkpoints = _parse_checkpoints(
            document_response.text(), maximum=self.max_models, source=self.name
        )
        records = tuple(self._record(checkpoint, revision) for checkpoint in checkpoints)
        if not records:
            raise ValueError(f"{self.name}: manifest contains no checkpoint links")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "document_url": document_url,
                "document_sha256": content_hash(document_response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, checkpoint: _Checkpoint, revision: str) -> SourceRecord:
        model_name = checkpoint.filename
        local_id = f"model:{model_name}"
        identifier = Identifier("msst:checkpoint", model_name)
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=f"{_DOCUMENT_PATH}:line-{checkpoint.line_number}",
        )
        release = ReleaseHint(
            local_id=f"release:{model_name}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("msst:checkpoint-file", model_name),),
            metadata={
                "repository": _REPOSITORY,
                "manifest_revision": revision,
                "manifest_path": _DOCUMENT_PATH,
                "manifest_line": checkpoint.line_number,
                "filename": model_name,
                "weight_url": checkpoint.url,
                "model_type": checkpoint.model_type,
                "instruments": checkpoint.instruments,
                "published_metrics": checkpoint.metrics,
            },
            locator=f"{_DOCUMENT_PATH}:line-{checkpoint.line_number}",
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(checkpoint.url),
            title=model_name,
            raw={
                "repository": _REPOSITORY,
                "manifest_revision": revision,
                "manifest_path": _DOCUMENT_PATH,
                "manifest_line": checkpoint.line_number,
                "filename": model_name,
                "weight_url": checkpoint.url,
                "model_type": checkpoint.model_type,
                "instruments": checkpoint.instruments,
                "published_metrics": checkpoint.metrics,
            },
            text=(
                f"Music-Source-Separation-Training checkpoint {model_name}\n"
                f"model_type: {checkpoint.model_type}\n"
                f"instruments: {checkpoint.instruments}\n"
                f"metrics: {checkpoint.metrics}"
            ),
            identifiers=(identifier,),
            links=(
                Link(
                    f"{_REPOSITORY_URL}/blob/{revision}/{_DOCUMENT_PATH}",
                    "model_card",
                    locator=f"{_DOCUMENT_PATH}:line-{checkpoint.line_number}",
                    crawl=False,
                ),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False),
                Link(checkpoint.url, "weights", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_checkpoints(document: str, *, maximum: int, source: str) -> tuple[_Checkpoint, ...]:
    checkpoints: list[_Checkpoint] = []
    seen: set[str] = set()
    columns: tuple[int, int | None, int | None, int] | None = None
    for line_number, line in enumerate(document.splitlines(), start=1):
        cells = _table_cells(line)
        if cells is None:
            if line.lstrip().startswith("#"):
                columns = None
            continue
        header = tuple(cell.strip().casefold() for cell in cells)
        if "checkpoint" in header:
            if "model type" not in header:
                columns = None
                continue
            columns = (
                header.index("model type"),
                header.index("instruments") if "instruments" in header else None,
                header.index("metrics") if "metrics" in header else None,
                header.index("checkpoint"),
            )
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        if columns is None:
            continue
        model_index, instruments_index, metrics_index, checkpoint_index = columns
        if len(cells) <= max(
            model_index,
            checkpoint_index,
            instruments_index if instruments_index is not None else 0,
            metrics_index if metrics_index is not None else 0,
        ):
            continue
        row = cells
        model_type = _plain_text(row[model_index])
        instruments = _plain_text(row[instruments_index]) if instruments_index is not None else ""
        metrics = _plain_text(row[metrics_index]) if metrics_index is not None else ""
        for match in _LINK.finditer(row[checkpoint_index]):
            url = match.group("url").strip()
            filename = urlsplit(url).path.rsplit("/", 1)[-1]
            if not filename.casefold().endswith(_CHECKPOINT_SUFFIXES):
                continue
            _validate_checkpoint_url(url, filename, source)
            if filename in seen:
                raise ValueError(f"{source}: duplicate checkpoint filename {filename!r}")
            seen.add(filename)
            checkpoints.append(
                _Checkpoint(
                    filename=filename,
                    url=canonicalize_url(url),
                    model_type=model_type,
                    instruments=instruments,
                    metrics=metrics,
                    line_number=line_number,
                )
            )
            if len(checkpoints) > maximum:
                raise ValueError(f"{source}: checkpoint manifest exceeds {maximum} entries")
    return tuple(checkpoints)


def _table_cells(line: str) -> tuple[str, ...] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return tuple(cell.strip() for cell in stripped[1:-1].split("|"))


def _plain_text(markdown: str) -> str:
    return re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", markdown).replace("<br>", " ").strip()


def _validate_checkpoint_url(url: str, filename: str, source: str) -> None:
    parsed = urlsplit(url)
    parts = parsed.path.split("/")
    allowed_github_releases = {
        ("ZFTurbo", "Music-Source-Separation-Training"),
        ("TRvlvr", "model_repo"),
        ("jarredou", "models"),
    }
    valid = False
    if parsed.scheme == "https" and parsed.hostname == "github.com":
        valid = (
            len(parts) == 7
            and (parts[1], parts[2]) in allowed_github_releases
            and parts[3:5] == ["releases", "download"]
            and parts[-1] == filename
        )
    elif parsed.scheme == "https" and parsed.hostname == "dl.fbaipublicfiles.com":
        valid = parts[:3] == ["", "demucs", "hybrid_transformer"] and parts[-1] == filename
    elif parsed.scheme == "https" and parsed.hostname == "huggingface.co":
        valid = (
            len(parts) == 6
            and bool(parts[1])
            and bool(parts[2])
            and parts[3] in {"resolve", "blob"}
            and parts[4] == "main"
            and parts[5] == filename
        )
    if (
        not valid
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: unsupported checkpoint URL for {filename!r}")


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["MsstPretrainedModelsSourceAdapter"]
