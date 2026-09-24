"""Exact PyTorch checkpoints in Megvii's first-party YOLOX model zoo."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_REPOSITORY = "Megvii-BaseDetection/YOLOX"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
_CONFIG_LINK = re.compile(r"\[([^\]]+)\]\((\./exps/default/[A-Za-z0-9_.-]+\.py)\)")
_MODEL_CONFIGS = {
    "yolox-s": ("yolox_s.py", {"yolox_s.pth"}),
    "yolox-m": ("yolox_m.py", {"yolox_m.pth"}),
    "yolox-l": ("yolox_l.py", {"yolox_l.pth"}),
    "yolox-x": ("yolox_x.py", {"yolox_x.pth"}),
    "yolox-darknet53": ("yolov3.py", {"yolox_darknet.pth", "yolox_darknet53.pth"}),
    "yolox-nano": ("yolox_nano.py", {"yolox_nano.pth"}),
    "yolox-tiny": ("yolox_tiny.py", {"yolox_tiny.pth", "yolox_tiny_32dot8.pth"}),
}


class YOLOXModelZooSourceAdapter:
    """Index source-declared model variants and historical releases from README tables."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers direct `.pth` GitHub release assets in the official YOLOX README's "
        "standard and light model tables, including their explicitly labeled legacy "
        "tables. It excludes ONNX exports, OneDrive mirrors, user-trained weights, "
        "and models not listed in those tables."
    )

    def __init__(
        self,
        *,
        name: str = "yolox-model-zoo",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_readme_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 1000,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or min(max_readme_bytes, max_entries) <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_readme_bytes = max_readme_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_readme_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "yolox-model-zoo-v1",
                "repository": repository,
                "branch": branch,
                "max_readme_bytes": max_readme_bytes,
                "max_entries": max_entries,
                "admission": "YOLOX README model rows with direct GitHub .pth release links",
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/README.md"

    def readme_page_url(self, revision: str) -> str:
        return f"https://github.com/{self.repository}/blob/{revision}/README.md"

    def config_url(self, revision: str, path: str) -> str:
        return f"https://github.com/{self.repository}/blob/{revision}/{path.removeprefix('./')}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        if revision == state.get("completed_revision"):
            count = state.get("checkpoint_count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"{self.name}: invalid completed checkpoint count")
            return SourcePage((), dict(state), True, upstream_count=count)

        source_url = self.raw_url(revision)
        response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_readme_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        checkpoints = _parse_model_zoo(readme, maximum=self.max_entries)
        if not checkpoints:
            raise ValueError(f"{self.name}: README contains no recognized checkpoint rows")
        page_url = self.readme_page_url(revision)
        records = tuple(
            self._record(entry, revision, page_url, response.body) for entry in checkpoints
        )
        next_state = {
            "completed_revision": revision,
            "checkpoint_count": len(records),
            "source_url": source_url,
            "source_sha256": content_hash(response.body),
        }
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, entry: _CheckpointRow, revision: str, page_url: str, source: bytes
    ) -> SourceRecord:
        model_value = _model_key(entry.name)
        model_local_id = f"model:{model_value}"
        model_identifier = Identifier("yolox:model", model_value)
        model = ModelHint(
            local_id=model_local_id,
            name=entry.name,
            aliases=(entry.config_path.rsplit("/", 1)[-1].removesuffix(".py"),),
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=f"README.md {entry.table_section} table: {entry.name}",
        )
        release_filename = urlsplit(entry.weight_url).path.rsplit("/", 1)[-1]
        tag = _release_coordinates(entry.weight_url)[0]
        release_value = f"{tag}/{release_filename}"
        checkpoint_identifier = Identifier("yolox:checkpoint", release_value)
        config_url = self.config_url(revision, entry.config_path)
        release = ReleaseHint(
            local_id=f"release:{release_value}",
            model_local_id=model_local_id,
            version=tag,
            identifiers=(checkpoint_identifier,),
            metadata={
                "checkpoint_filename": release_filename,
                "weight_url": entry.weight_url,
                "config_path": entry.config_path,
                "config_url": config_url,
                "table_section": entry.table_section,
                "table_metrics": entry.metrics,
            },
            locator=model.locator,
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{release_value}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(entry.weight_url),
            title=f"{entry.name} checkpoint ({tag})",
            raw={
                "repository": self.repository,
                "revision": revision,
                "source_path": "README.md",
                "source_sha256": content_hash(source),
                "model_name": entry.name,
                "checkpoint_filename": release_filename,
                "release_tag": tag,
                "weight_url": entry.weight_url,
                "config_path": entry.config_path,
                "config_url": config_url,
                "table_section": entry.table_section,
                "table_metrics": entry.metrics,
            },
            text=f"Official {entry.name} model-zoo checkpoint: {entry.weight_url}",
            identifiers=(checkpoint_identifier, model_identifier),
            links=(
                Link(page_url, relation="model_card", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
                Link(entry.weight_url, relation="weights", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
                Link(config_url, relation="model_config", crawl=False,
                     locator=model.locator, model_local_ids=(model_local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


class _CheckpointRow:
    __slots__ = ("name", "config_path", "weight_url", "table_section", "metrics")

    def __init__(
        self,
        name: str,
        config_path: str,
        weight_url: str,
        table_section: str,
        metrics: Mapping[str, str],
    ) -> None:
        self.name = name
        self.config_path = config_path
        self.weight_url = weight_url
        self.table_section = table_section
        self.metrics = dict(metrics)


def _parse_model_zoo(markdown: str, *, maximum: int) -> tuple[_CheckpointRow, ...]:
    rows: dict[str, _CheckpointRow] = {}
    model_type = ""
    legacy = False
    in_table = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#### "):
            model_type = stripped[5:].strip().casefold()
            legacy = False
            in_table = False
        if "<summary>Legacy models</summary>" in stripped:
            legacy = True
            in_table = False
        elif stripped == "</details>":
            legacy = False
            in_table = False
        if not stripped.startswith("|"):
            if in_table:
                in_table = False
            continue
        fields = _table_cells(stripped)
        if not fields:
            continue
        if "model" in fields[0].casefold() and "weights" in " ".join(fields).casefold():
            in_table = True
            continue
        if all(re.fullmatch(r":?-{2,}:?", field.replace(" ", "")) for field in fields):
            continue
        if not in_table:
            continue
        model_match = _CONFIG_LINK.search(fields[0])
        if model_match is None:
            # Ignore non-checkpoint rows, but fail closed if a model-looking row
            # contains a direct release asset that could otherwise be omitted.
            if any(_release_coordinates(url) for _, url in _LINK.findall(stripped)):
                raise ValueError("YOLOX README has a checkpoint row without its config link")
            continue
        name = model_match.group(1).strip()
        config_path = model_match.group(2)
        release_links = [
            url for _, url in _LINK.findall("|".join(fields)) if _release_coordinates(url)
        ]
        if not release_links:
            continue
        if len(release_links) != 1:
            raise ValueError(f"YOLOX README row must have one GitHub checkpoint URL: {name}")
        weight_url = release_links[0]
        model_mapping = _MODEL_CONFIGS.get(_model_key(name))
        coordinates = _release_coordinates(weight_url)
        if (
            model_mapping is None
            or config_path.rsplit("/", 1)[-1].casefold() != model_mapping[0]
            or coordinates is None
            or coordinates[1].casefold() not in model_mapping[1]
        ):
            raise ValueError(f"YOLOX README has an unexpected model/checkpoint mapping: {name}")
        section = ("legacy-" if legacy else "") + (model_type or "unspecified")
        metrics = {
            f"column_{index}": field.strip()
            for index, field in enumerate(fields[1:-1], start=1)
            if field.strip()
        }
        key = weight_url
        if key in rows:
            raise ValueError(f"YOLOX README contains duplicate checkpoint URL: {weight_url}")
        rows[key] = _CheckpointRow(name, config_path, weight_url, section, metrics)
        if len(rows) > maximum:
            raise ValueError("YOLOX README model-zoo table exceeds entry limit")
    return tuple(rows.values())


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _release_coordinates(url: str) -> tuple[str, str] | None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.query
        or parts.fragment
        or parts.netloc != "github.com"
    ):
        return None
    match = re.fullmatch(
        r"/Megvii-BaseDetection/(?:YOLOX|storage)/releases/download/([^/]+)/(yolox_[a-z0-9_]+\.pth)",
        parts.path,
        re.IGNORECASE,
    )
    return (match.group(1), match.group(2)) if match else None


def _model_key(name: str) -> str:
    return name.casefold()
