"""First-party RT-1 model inventory from the public Google Research repository."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_PATH = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_MODEL_DIR = re.compile(r"^trained_checkpoints/([A-Za-z0-9_.-]+)(?:/|$)")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RoboticsTransformerCheckpointSourceAdapter:
    """Enumerate RT-1 SavedModels and their exact file URLs at one Git commit.

    The first-party repository README declares three RT-1 variants in
    ``trained_checkpoints/``. The adapter enumerates only model directories that
    contain both ``policy_specs.pbtxt`` and ``variables/variables.index``, then
    links every source-declared file in that SavedModel directory at the
    observed commit. It reads tree metadata only and never fetches weights.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers TensorFlow SavedModel directories beneath "
        "google-research/robotics_transformer/trained_checkpoints that contain "
        "the repository's policy specification and variables index at an "
        "observed Git tree. It does not include RT-1-X, later derivatives, or "
        "models outside that directory, verify raw-file availability, or fetch "
        "checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "google-robotics-transformer-checkpoints",
        repository: str = "google-research/robotics_transformer",
        branch: str = "master",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_files: int = 100_000,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.branch = _required_text(branch, "branch")
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_files = _positive_int(max_files, "max_files")
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "robotics-transformer-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "max_response_bytes": max_response_bytes,
                "max_files": max_files,
                "admission": "SavedModel has policy_specs.pbtxt and variables/variables.index",
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def tree_url(self) -> str:
        branch = quote(self.branch, safe="")
        return (
            f"https://api.github.com/repos/{self.repository}/git/trees/"
            f"{branch}?recursive=1"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.tree_url,
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: Git tree returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Git tree exceeds {self.max_response_bytes} bytes")
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: Git tree response is not an object")
        revision = _text(payload.get("sha"))
        if not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: Git tree response has no full commit SHA")
        tree = payload.get("tree")
        if not isinstance(tree, list):
            raise ValueError(f"{self.name}: Git tree response has no tree list")
        if payload.get("truncated") is True:
            raise ValueError(f"{self.name}: Git tree response is truncated")
        if len(tree) > self.max_files:
            raise ValueError(f"{self.name}: Git tree exceeds {self.max_files} entries")

        files: dict[str, dict[str, Any]] = {}
        for item in tree:
            if not isinstance(item, Mapping) or item.get("type") != "blob":
                continue
            path = _text(item.get("path"))
            if not _SAFE_PATH.fullmatch(path):
                continue
            if _MODEL_DIR.match(path):
                files[path] = {"size": item.get("size"), "sha": _text(item.get("sha"))}
        models = _models_from_tree(files)
        if not models:
            raise ValueError(
                f"{self.name}: no complete RT-1 SavedModels found in trained_checkpoints"
            )

        checked_at = _isoformat(self.clock())
        records = tuple(
            self._record(model_name, model_files, revision, response.body)
            for model_name, model_files in sorted(models.items())
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "model_count": len(records),
                "file_count": sum(len(record.links) - 2 for record in records),
                "tree_sha256": content_hash(response.body),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        model_name: str,
        model_files: Mapping[str, Mapping[str, Any]],
        revision: str,
        tree_body: bytes,
    ) -> SourceRecord:
        local_id = f"model:{model_name}"
        model_id = Identifier("rt1:model", model_name)
        model_path = f"trained_checkpoints/{model_name}"
        model_page = f"{self.repository_url}/tree/{revision}/{model_path}"
        model = ModelHint(
            local_id=local_id,
            name=f"RT-1 {model_name}",
            identifiers=(model_id,),
            status=ModelStatus.RELEASED,
            locator=model_path,
        )
        file_links = []
        manifest = []
        for path, data in sorted(model_files.items()):
            relative = path.removeprefix(model_path + "/")
            raw_url = (
                f"https://raw.githubusercontent.com/{self.repository}/"
                f"{revision}/{quote(path, safe='/')}"
            )
            file_links.append(
                Link(
                    raw_url,
                    relation="model_artifact",
                    locator=path,
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            manifest.append(
                {
                    "path": relative,
                    "url": raw_url,
                    "git_blob_sha": data.get("sha"),
                    "size_bytes": data.get("size"),
                }
            )
        metadata = {
            "repository": self.repository,
            "revision": revision,
            "model_directory": model_path,
            "saved_model_files": manifest,
        }
        release = ReleaseHint(
            local_id=f"release:{model_name}:{revision}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("rt1:checkpoint", f"{model_name}@{revision}"),),
            metadata=metadata,
            locator=model_path,
        )
        links = (
            Link(model_page, relation="model_card", crawl=False),
            Link(self.repository_url, relation="source_implementation", crawl=False),
            *file_links,
        )
        return SourceRecord(
            source_record_id=f"rt1:{model_name}@{revision}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(model_page),
            title=f"RT-1 {model_name}",
            raw=metadata | {"git_tree_sha256": content_hash(tree_body)},
            text="\n".join(
                (f"RT-1 checkpoint: {model_name}", f"Git revision: {revision}", model_path)
            ),
            identifiers=(model_id,),
            links=links,
            models=(model,),
            releases=(release,),
        )


def _models_from_tree(
    files: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for path, metadata in files.items():
        match = _MODEL_DIR.match(path)
        if match:
            grouped.setdefault(match.group(1), {})[path] = metadata
    required = {"policy_specs.pbtxt", "variables/variables.index"}
    return {
        name: members
        for name, members in grouped.items()
        if required.issubset(
            {path.removeprefix(f"trained_checkpoints/{name}/") for path in members}
        )
    }


def _repository(value: str) -> str:
    value = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError("repository must be owner/name")
    return value


def _required_text(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
