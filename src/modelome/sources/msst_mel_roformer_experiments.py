"""Read the linked MSST MelRoformer experiment checkpoint table."""

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
_DOCUMENT_PATH = "docs/mel_roformer_experiments.md"
_COMMIT_URL = f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"
_RAW_PREFIX = f"https://raw.githubusercontent.com/{_REPOSITORY}/"
_REPOSITORY_URL = f"https://github.com/{_REPOSITORY}"
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"\[(?P<label>[^]]+)\]\((?P<url>https?://[^)]+)\)")
_CHECKPOINT_SUFFIXES = (".ckpt", ".zip.001", ".zip.002")


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class _Experiment:
    filename: str
    urls: tuple[str, ...]
    score: str
    parameters: Mapping[str, str]
    comment: str
    line_number: int


class MsstMelRoformerExperimentsSourceAdapter:
    """Enumerate model and multipart checkpoint files from the experiment index."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only weight files explicitly linked in MSST's mel_roformer_experiments.md "
        "at the fetched revision. The split large checkpoint is represented as one model "
        "with two file links; config-only rows are not included."
    )

    def __init__(
        self,
        *,
        name: str = "msst-mel-roformer-experiments",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_models: int = 50,
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
                "adapter": "msst-mel-roformer-experiments-v1",
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
        experiments = _parse_experiments(
            document_response.text(), maximum=self.max_models, source=self.name
        )
        records = tuple(self._record(experiment, revision) for experiment in experiments)
        if not records:
            raise ValueError(f"{self.name}: manifest contains no checkpoint links")
        file_count = sum(len(experiment.urls) for experiment in experiments)
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "document_url": document_url,
                "document_sha256": content_hash(document_response.body),
                "model_count": len(records),
                "checkpoint_file_count": file_count,
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, experiment: _Experiment, revision: str) -> SourceRecord:
        model_name = experiment.filename
        local_id = f"model:{model_name}"
        identifier = Identifier("msst:mel-roformer-checkpoint", model_name)
        locator = f"{_DOCUMENT_PATH}:line-{experiment.line_number}"
        model = ModelHint(
            local_id=local_id,
            name=model_name,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=locator,
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
                "manifest_line": experiment.line_number,
                "filename": model_name,
                "weight_urls": experiment.urls,
                "sdr_score": experiment.score,
                "experiment_parameters": dict(experiment.parameters),
                "comment": experiment.comment,
                "multipart": len(experiment.urls) > 1,
            },
            locator=locator,
        )
        links = [
            Link(
                f"{_REPOSITORY_URL}/blob/{revision}/{_DOCUMENT_PATH}",
                "model_card",
                locator=locator,
                crawl=False,
            ),
            Link(_REPOSITORY_URL, "source_implementation", crawl=False),
        ]
        links.extend(
            Link(url, "weights", locator=f"{locator}:part-{index}", crawl=False)
            for index, url in enumerate(experiment.urls, start=1)
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(experiment.urls[0]),
            title=model_name,
            raw={
                "repository": _REPOSITORY,
                "manifest_revision": revision,
                "manifest_path": _DOCUMENT_PATH,
                "manifest_line": experiment.line_number,
                "filename": model_name,
                "weight_urls": experiment.urls,
                "sdr_score": experiment.score,
                "experiment_parameters": dict(experiment.parameters),
                "comment": experiment.comment,
            },
            text=(
                f"MSST MelRoformer experiment checkpoint {model_name}\n"
                f"SDR score: {experiment.score}\n"
                + "\n".join(f"{key}: {value}" for key, value in experiment.parameters.items())
                + (f"\ncomment: {experiment.comment}" if experiment.comment else "")
            ),
            identifiers=(identifier,),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


def _parse_experiments(document: str, *, maximum: int, source: str) -> tuple[_Experiment, ...]:
    rows: list[_Experiment] = []
    seen_names: set[str] = set()
    columns: dict[str, int] | None = None
    for line_number, line in enumerate(document.splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            columns = None
            continue
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        lowered = tuple(re.sub(r"\s+", " ", cell).casefold() for cell in cells)
        if "dl checkpoint" in lowered:
            columns = {name: lowered.index(name) for name in lowered}
            if "average sdr score" not in columns:
                columns = None
            continue
        if columns is None:
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        required = max(columns.values())
        if len(cells) <= required:
            continue
        checkpoint_cell = cells[columns["dl checkpoint"]]
        file_urls: list[str] = []
        for match in _LINK.finditer(checkpoint_cell):
            url = match.group("url").strip()
            filename = urlsplit(url).path.rsplit("/", 1)[-1]
            if not filename.casefold().endswith(_CHECKPOINT_SUFFIXES):
                continue
            _validate_url(url, filename, source)
            file_urls.append(canonicalize_url(url))
        if not file_urls:
            continue
        experiment = _experiment_from_row(
            cells,
            columns,
            tuple(file_urls),
            line_number,
            source,
        )
        if experiment.filename in seen_names:
            raise ValueError(f"{source}: duplicate experiment checkpoint {experiment.filename!r}")
        seen_names.add(experiment.filename)
        rows.append(experiment)
        if len(rows) > maximum:
            raise ValueError(f"{source}: experiment table exceeds {maximum} checkpoints")
    return tuple(rows)


def _experiment_from_row(
    cells: tuple[str, ...],
    columns: Mapping[str, int],
    urls: tuple[str, ...],
    line_number: int,
    source: str,
) -> _Experiment:
    filenames = tuple(urlsplit(url).path.rsplit("/", 1)[-1] for url in urls)
    if len(filenames) == 1:
        filename = filenames[0]
    else:
        if len(filenames) != 2 or not filenames[0].endswith(".zip.001"):
            raise ValueError(f"{source}: unsupported multipart checkpoint at line {line_number}")
        base = filenames[0].removesuffix(".001")
        if filenames[1] != f"{base}.002":
            raise ValueError(
                f"{source}: multipart checkpoint parts do not match at line {line_number}"
            )
        filename = base
    omitted = {"average sdr score", "dl checkpoint", "comment"}
    parameters = {
        label: _plain_text(cells[index])
        for label, index in columns.items()
        if label not in omitted and index < len(cells)
    }
    score = _plain_text(cells[columns["average sdr score"]])
    comment = _plain_text(cells[columns["comment"]]) if "comment" in columns else ""
    return _Experiment(filename, urls, score, parameters, comment, line_number)


def _plain_text(markdown: str) -> str:
    return re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", markdown).replace("<br>", " ").strip()


def _validate_url(url: str, filename: str, source: str) -> None:
    parsed = urlsplit(url)
    parts = parsed.path.split("/")
    valid = (
        parsed.scheme == "https"
        and parsed.hostname == "github.com"
        and len(parts) == 7
        and parts[1:5]
        == ["ZFTurbo", "Music-Source-Separation-Training", "releases", "download"]
        and parts[-1] == filename
        and bool(parts[5])
    )
    if (
        not valid
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: unsupported experiment checkpoint URL for {filename!r}")


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["MsstMelRoformerExperimentsSourceAdapter"]
