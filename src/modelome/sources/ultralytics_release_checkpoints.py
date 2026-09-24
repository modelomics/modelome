"""Join Ultralytics' official task tables to exact GitHub release assets."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
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

_REPOSITORY = "ultralytics/ultralytics"
_ASSETS_REPOSITORY = "ultralytics/assets"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ROW = re.compile(
    r"^\|\s*(?P<label>YOLO(?:v8|11)(?:-(?:seg|cls|pose|obb))?)\s*\|(?P<rest>.*)$"
)
_FILENAME = re.compile(r"`(?P<filename>yolo(?:v8|11)[nsmlx](?:-(?:seg|cls|pose|obb))?\.pt)`")


class UltralyticsReleaseCheckpointSourceAdapter:
    """Discover official YOLOv8/YOLO11 checkpoints by exact asset filename.

    Model names and tasks come from Ultralytics' first-party docs. Download URLs
    come only from public, published GitHub release asset metadata, and assets
    are admitted only when their full filename occurs in the docs table.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only YOLOv8 and YOLO11 `.pt` filenames explicitly declared in "
        "the official model docs and exactly matched to public assets in the "
        "official ultralytics/assets release API. It excludes other model "
        "families, non-`.pt` exports, drafts, and assets absent from the docs."
    )

    def __init__(
        self,
        *,
        name: str = "ultralytics-yolo-release-checkpoints",
        repository: str = _REPOSITORY,
        assets_repository: str = _ASSETS_REPOSITORY,
        branch: str = "main",
        model_docs: Sequence[str] = ("docs/en/models/yolov8.md", "docs/en/models/yolo11.md"),
        max_response_bytes: int = 4 * 1024 * 1024,
        max_releases: int = 500,
        max_assets_per_release: int = 1000,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or assets_repository != _ASSETS_REPOSITORY:
            raise ValueError("repositories must be the official Ultralytics repositories")
        if not name.strip() or not branch.strip():
            raise ValueError("name and branch are required")
        docs = tuple(model_docs)
        if set(docs) != {"docs/en/models/yolov8.md", "docs/en/models/yolo11.md"}:
            raise ValueError("model_docs must contain the YOLOv8 and YOLO11 first-party docs")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_response_bytes, max_releases, max_assets_per_release)
        ):
            raise ValueError("limits must be positive integers")
        self.name = name
        self.repository = repository
        self.assets_repository = assets_repository
        self.branch = branch
        self.model_docs = docs
        self.max_response_bytes = max_response_bytes
        self.max_releases = max_releases
        self.max_assets_per_release = max_assets_per_release
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "ultralytics-release-checkpoints-v1",
                "repository": repository,
                "assets_repository": assets_repository,
                "branch": branch,
                "model_docs": docs,
                "max_response_bytes": max_response_bytes,
                "max_releases": max_releases,
                "max_assets_per_release": max_assets_per_release,
                "matching": "exact-doc-filename-to-release-asset-basename",
            }
        )

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    @property
    def releases_url(self) -> str:
        return f"https://api.github.com/repos/{self.assets_repository}/releases"

    def docs_url(self, revision: str, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        revision = _response_object(revision_response, self.name, "commit")
        sha = revision.get("sha")
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError(f"{self.name}: invalid Ultralytics documentation revision")

        names_to_tasks: dict[str, str] = {}
        docs_digests: dict[str, str] = {}
        for path in self.model_docs:
            response = self.client.get(self.docs_url(sha, path), headers={"Accept": "text/plain"})
            if response.status != 200:
                raise ValueError(f"{self.name}: model docs returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: model docs exceed response limit")
            try:
                document = response.body.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{self.name}: model docs are not UTF-8") from exc
            parsed = _parse_docs(document)
            if not parsed:
                raise ValueError(f"{self.name}: no eligible checkpoint filenames in {path}")
            names_to_tasks.update(parsed)
            docs_digests[path] = content_hash(response.body)

        releases = self._release_inventory()
        inventory_digest = content_hash(list(releases))
        if (
            sha == state.get("completed_revision")
            and inventory_digest == state.get("release_inventory_digest")
            and docs_digests == state.get("docs_digests")
        ):
            next_state = dict(state)
            return SourcePage(
                (), next_state, True, upstream_count=_nonnegative(state.get("record_count"))
            )

        records = tuple(
            self._record(asset, release, names_to_tasks, sha, docs_digests)
            for release in releases
            for asset in release["assets"]
            if asset["name"] in names_to_tasks
        )
        if not records:
            raise ValueError(f"{self.name}: no release asset exactly matched the official docs")
        next_state = {
            "completed_revision": sha,
            "docs_digests": docs_digests,
            "release_inventory_digest": inventory_digest,
            "record_count": len(records),
        }
        return SourcePage(
            records, next_state, True, upstream_count=len(records), authoritative_snapshot=True
        )

    def _release_inventory(self) -> tuple[Mapping[str, Any], ...]:
        releases: list[Mapping[str, Any]] = []
        per_page = 50
        for page_number in range(1, (self.max_releases + per_page - 1) // per_page + 1):
            response = self.client.get(
                f"{self.releases_url}?per_page={per_page}&page={page_number}",
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: GitHub releases returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: GitHub releases exceed response limit")
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"{self.name}: GitHub releases response is not a list")
            if len(releases) + len(payload) > self.max_releases:
                raise ValueError(f"{self.name}: release inventory exceeds max_releases")
            for release in payload:
                if not isinstance(release, Mapping):
                    continue
                # The standard endpoint excludes drafts. Fail closed on malformed rows.
                tag = _nonempty(release.get("tag_name"))
                html_url = _github_release_url(release.get("html_url"), self.assets_repository)
                release_id = _positive_int(release.get("id"))
                assets_url = _github_assets_api_url(
                    release.get("assets_url"), self.assets_repository, release_id
                )
                if not tag or not html_url or not release_id or not assets_url:
                    raise ValueError(f"{self.name}: malformed public release row")
                assets = self._release_assets(assets_url, release_id, tag)
                releases.append({"id": release_id, "tag": tag, "url": html_url, "assets": assets})
            if len(payload) < per_page:
                return tuple(releases)
        raise ValueError(f"{self.name}: release inventory reached pagination limit")

    def _release_assets(
        self, assets_url: str, release_id: int, release_tag: str
    ) -> tuple[Mapping[str, Any], ...]:
        assets: list[Mapping[str, Any]] = []
        per_page = 100
        max_pages = (self.max_assets_per_release + per_page - 1) // per_page + 1
        for page_number in range(1, max_pages + 1):
            response = self.client.get(
                f"{assets_url}?per_page={per_page}&page={page_number}",
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: assets for release {release_id} returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(
                    f"{self.name}: assets for release {release_id} exceed response limit"
                )
            payload = response.json()
            if not isinstance(payload, list):
                raise ValueError(f"{self.name}: release assets response is not a list")
            if len(assets) + len(payload) > self.max_assets_per_release:
                raise ValueError(
                    f"{self.name}: release {release_id} exceeds max_assets_per_release"
                )
            for asset in payload:
                if not isinstance(asset, Mapping):
                    raise ValueError(f"{self.name}: malformed asset in release {release_id}")
                asset_id = _positive_int(asset.get("id"))
                filename = _nonempty(asset.get("name"))
                url = _github_release_asset_url(
                    asset.get("browser_download_url"), self.assets_repository, release_tag
                )
                if not asset_id or not filename or not url:
                    raise ValueError(f"{self.name}: invalid asset in release {release_id}")
                assets.append({"name": filename, "url": url, "id": asset_id})
            if len(payload) < per_page:
                return tuple(assets)
        raise ValueError(f"{self.name}: assets for release {release_id} reached pagination limit")

    def _record(
        self,
        asset: Mapping[str, Any],
        release: Mapping[str, Any],
        names_to_tasks: Mapping[str, str],
        revision: str,
        docs_digests: Mapping[str, str],
    ) -> SourceRecord:
        filename = str(asset["name"])
        family = "YOLOv8" if filename.startswith("yolov8") else "YOLO11"
        display_name = (
            f"{family}{filename[len('yolov8') if family == 'YOLOv8' else len('yolo11') : -3]}"
        )
        model_id = filename.removesuffix(".pt")
        model_local_id = f"model:{model_id}"
        release_tag = str(release["tag"])
        url = str(asset["url"])
        task = names_to_tasks[filename]
        doc_path = self.model_docs[0] if family == "YOLOv8" else self.model_docs[1]
        source_locator = f"{doc_path}: {filename} ({task})"
        model = ModelHint(
            local_id=model_local_id,
            name=display_name,
            aliases=(filename,),
            identifiers=(Identifier("ultralytics:model-file", filename),),
            status=ModelStatus.RELEASED,
            locator=source_locator,
        )
        release_value = f"{release_tag}/{filename}"
        return SourceRecord(
            source_record_id=f"ultralytics-release-asset:{asset.get('id') or release_value}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{display_name} checkpoint ({release_tag})",
            identifiers=(
                Identifier("ultralytics:checkpoint", release_value),
                Identifier("ultralytics:model-file", filename),
            ),
            links=(
                Link(str(release["url"]), relation="source_release", crawl=False),
                Link(url, relation="weights", crawl=False),
                Link(
                    self.docs_url(
                        revision, self.model_docs[0] if family == "YOLOv8" else self.model_docs[1]
                    ),
                    relation="model_card",
                    crawl=False,
                    locator=source_locator,
                    model_local_ids=(model_local_id,),
                ),
            ),
            raw={
                "record_type": "ultralytics_release_checkpoint",
                "release_tag": release_tag,
                "release_url": release["url"],
                "filename": filename,
                "download_url": url,
                "task": task,
                "docs_revision": revision,
                "docs_digests": dict(docs_digests),
                "asset_metadata_only": True,
            },
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_value}",
                    model_local_id=model_local_id,
                    version=release_tag,
                    identifiers=(Identifier("ultralytics:checkpoint", release_value),),
                    metadata={"checkpoint_filename": filename, "weight_url": url, "task": task},
                    locator=source_locator,
                ),
            ),
        )


def _parse_docs(document: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in document.splitlines():
        row = _MODEL_ROW.match(line)
        if row is None:
            continue
        label = row.group("label")
        task = {
            "YOLOv8": "object-detection",
            "YOLOv8-seg": "instance-segmentation",
            "YOLOv8-cls": "image-classification",
            "YOLOv8-pose": "pose-estimation",
            "YOLOv8-obb": "oriented-object-detection",
            "YOLO11": "object-detection",
            "YOLO11-seg": "instance-segmentation",
            "YOLO11-cls": "image-classification",
            "YOLO11-pose": "pose-estimation",
            "YOLO11-obb": "oriented-object-detection",
        }.get(label)
        if task is None:
            continue
        for match in _FILENAME.finditer(row.group("rest")):
            result[match.group("filename")] = task
    return result


def _response_object(response: Any, name: str, label: str) -> Mapping[str, Any]:
    if response.status != 200:
        raise ValueError(f"{name}: {label} endpoint returned HTTP {response.status}")
    if len(response.body) > 4 * 1024 * 1024:
        raise ValueError(f"{name}: {label} response exceeds limit")
    payload = response.json()
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name}: invalid {label} response")
    return payload


def _github_release_url(value: Any, repository: str) -> str:
    return _github_url(value, "github.com", f"/{repository}/releases/tag/")


def _github_release_asset_url(value: Any, repository: str, release_tag: str) -> str:
    prefix = f"/{repository}/releases/download/{quote(release_tag, safe='')}/"
    return _github_url(value, "github.com", prefix)


def _github_assets_api_url(value: Any, repository: str, release_id: int) -> str:
    expected = f"https://api.github.com/repos/{repository}/releases/{release_id}/assets"
    return value if isinstance(value, str) and value == expected else ""


def _github_url(value: Any, host: str, path_prefix: str) -> str:
    if not isinstance(value, str):
        return ""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or not parsed.path.startswith(path_prefix)
    ):
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    return value


def _nonempty(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _positive_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


__all__ = ["UltralyticsReleaseCheckpointSourceAdapter"]
