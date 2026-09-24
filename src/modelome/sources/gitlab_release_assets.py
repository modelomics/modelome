"""Bounded, checkpointable discovery from GitLab's public project/release APIs.

The adapter advances through GitLab.com public projects and then the paginated
release list for each project. It performs at most one API request per call to
``fetch_page`` and only projects explicitly declared release links whose URLs
look like checkpoint files. It never downloads asset bytes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_API = "https://gitlab.com/api/v4"
_PAGE_SIZE = 100
_CHECKPOINT_SUFFIXES = (
    ".safetensors",
    ".gguf",
    ".ggml",
    ".onnx",
    ".engine",
    ".rtxplan",
    ".plan",
    ".pt",
    ".pte",
    ".pth",
    ".ckpt",
    ".bin",
    ".model",
    ".h5",
    ".keras",
    ".tflite",
    ".ptl",
    ".pb",
    ".pdparams",
    ".msgpack",
    ".weights",
)
_MODEL_CONTEXT = re.compile(
    r"\b(model|checkpoint|weights?|pretrained|neural|llm|transformer)\b", re.I
)
_GENERIC = frozenset({"asset", "binary", "checkpoint", "download", "file", "model", "weights"})
_LINK_RE = re.compile(r"<([^>]+)>\s*((?:;\s*[^,]+)*)")
_REL_RE = re.compile(r'\brel\s*=\s*(?:"([^"]+)"|([^;\s,]+))', re.I)


@dataclass(frozen=True, slots=True)
class GitLabProjectIdRange:
    """One bounded slice of public GitLab project IDs: ``(after, through]``."""

    name: str
    initial_project_id: int
    max_project_id: int
    max_projects: int

    def adapter_kwargs(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "initial_project_id": self.initial_project_id,
            "max_project_id": self.max_project_id,
            "max_projects": self.max_projects,
        }


def plan_gitlab_project_id_ranges(
    *,
    initial_project_id: int,
    max_project_id: int,
    shard_count: int,
    source_name_prefix: str = "gitlab-public-release-assets",
) -> tuple[GitLabProjectIdRange, ...]:
    """Partition a numeric project-ID interval into disjoint bounded slices.

    Sparse project IDs do not consume the project cap. Every slice is at most
    10,000 IDs wide; use additional shards for larger ranges.
    """

    lower = _positive_or_zero_integer(initial_project_id, "initial_project_id")
    upper = _positive_or_zero_integer(max_project_id, "max_project_id")
    if upper <= lower:
        raise ValueError("max_project_id must be greater than initial_project_id")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise ValueError("shard_count must be a positive integer")
    span = upper - lower
    if shard_count > span:
        raise ValueError("shard_count cannot exceed the number of IDs in the range")
    if not isinstance(source_name_prefix, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", source_name_prefix.strip()
    ):
        raise ValueError("source_name_prefix must be a simple source-name token")
    base, remainder = divmod(span, shard_count)
    cursor = lower
    result: list[GitLabProjectIdRange] = []
    for index in range(shard_count):
        width = base + (index < remainder)
        end = cursor + width
        if width > 10_000:
            raise ValueError("each range must cover at most 10000 IDs; increase shard_count")
        result.append(
            GitLabProjectIdRange(
                name=f"{source_name_prefix.strip()}-{cursor + 1}-{end}",
                initial_project_id=cursor,
                max_project_id=end,
                max_projects=width,
            )
        )
        cursor = end
    return tuple(result)


class GitLabPublicReleaseAssetsSourceAdapter:
    """Scan public projects and their explicit release asset links.

    A source call makes no more than one GET. Checkpoint state carries the
    public-project cursor, queued projects, and the current project's release
    cursor. This makes the large scan resumable and rate-limit pacing external.
    """

    name = "gitlab-public-release-assets"
    disable_derived_extraction = True
    coverage_limitation = (
        "Covers public GitLab.com projects and release links returned by the "
        "public REST API. Projects without releases and model files not declared "
        "as release links are not represented. Release lists use offset pagination "
        "ordered by creation time, not a snapshot cursor; a release deleted during "
        "a scan can shift later pages and may require a later rescan."
    )

    def __init__(
        self,
        *,
        name: str = "gitlab-public-release-assets",
        initial_project_id: int | None = None,
        max_project_id: int | None = None,
        max_projects: int | None = None,
        client: HttpClient | Any | None = None,
        page_size: int = _PAGE_SIZE,
        max_projects_per_page: int = _PAGE_SIZE,
        max_releases_per_page: int = _PAGE_SIZE,
        max_assets_per_release: int = 1_000,
    ) -> None:
        for label, value in (
            ("page_size", page_size),
            ("max_projects_per_page", max_projects_per_page),
            ("max_releases_per_page", max_releases_per_page),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
                raise ValueError(f"{label} must be an integer from 1 to 100")
        if (
            isinstance(max_assets_per_release, bool)
            or not isinstance(max_assets_per_release, int)
            or not 1 <= max_assets_per_release <= 10_000
        ):
            raise ValueError("max_assets_per_release must be an integer from 1 to 10000")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be nonempty text")
        if initial_project_id is not None:
            initial_project_id = _positive_or_zero_integer(initial_project_id, "initial_project_id")
        if max_project_id is not None:
            max_project_id = _positive_or_zero_integer(max_project_id, "max_project_id")
        if (initial_project_id is None) != (max_project_id is None):
            raise ValueError("initial_project_id and max_project_id must be set together")
        if max_project_id is not None and max_project_id <= initial_project_id:
            raise ValueError("max_project_id must be greater than initial_project_id")
        if max_projects is not None:
            max_projects = _positive_or_zero_integer(max_projects, "max_projects")
            if max_projects < 1:
                raise ValueError("max_projects must be positive")
        if initial_project_id is not None:
            width = max_project_id - initial_project_id
            if width > 10_000:
                raise ValueError("bounded project ranges must cover at most 10000 IDs")
            if max_projects is None:
                max_projects = width
            elif max_projects > width:
                raise ValueError("max_projects cannot exceed the numeric project-ID range")
        self.name = name.strip()
        self.initial_project_id = initial_project_id
        self.max_project_id = max_project_id
        self.max_projects = max_projects
        self.client = client if client is not None else HttpClient()
        self.page_size = min(page_size, max_projects_per_page)
        self.max_projects_per_page = max_projects_per_page
        self.max_releases_per_page = max_releases_per_page
        self.max_assets_per_release = max_assets_per_release
        self.checkpoint_signature = content_hash(
            {
                "adapter": "gitlab-public-release-assets-v2",
                "api": _API,
                "name": self.name,
                "initial_project_id": initial_project_id,
                "max_project_id": max_project_id,
                "max_projects": max_projects,
                "page_size": self.page_size,
                "max_projects_per_page": max_projects_per_page,
                "max_releases_per_page": max_releases_per_page,
                "max_assets_per_release": max_assets_per_release,
                "checkpoint_suffixes": _CHECKPOINT_SUFFIXES,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if not isinstance(state, Mapping):
            raise TypeError("gitlab-public-release-assets: state must be a mapping")
        current = _project_state(state.get("current_project"))
        queue = _project_queue(state.get("project_queue", []))
        projects_url = _optional_text(state.get("projects_next_url"))
        last_project_id = _optional_positive_int(state.get("last_project_id"))
        projects_seen = _positive_or_zero_integer(state.get("projects_seen", 0), "projects_seen")
        projects_started = state.get("projects_started", False)
        if not isinstance(projects_started, bool):
            raise ValueError("projects_started must be boolean")
        release_url = _optional_text(state.get("release_next_url"))
        release_page = _optional_positive_int(state.get("release_page")) or 1

        if current is None and queue:
            current, queue = queue[0], queue[1:]
            release_url = None
            release_page = 1

        if current is not None:
            request_url = release_url or (
                f"{_API}/projects/{current['id']}/releases?order_by=created_at&sort=asc"
                f"&per_page={self.max_releases_per_page}&page={release_page}"
            )
            response = self._get(request_url, "release", allow_not_found=True)
            if response is None:
                next_state = {
                    "projects_started": projects_started,
                    "project_queue": queue,
                    "current_project": None,
                    "projects_next_url": projects_url,
                    "last_project_id": last_project_id,
                    "projects_seen": projects_seen,
                }
                complete = not queue and not projects_url and projects_started
                return SourcePage(
                    records=(),
                    next_state=next_state,
                    complete=complete,
                    upstream_count=0 if complete else None,
                )
            payload = response.json()
            if not _is_sequence(payload) or len(payload) > self.max_releases_per_page:
                raise ValueError("GitLab release response is not a bounded JSON array")
            records = tuple(
                record
                for index, release in enumerate(payload)
                for record in self._release_records(current, release, index)
            )
            next_release = _next_link(response.headers, response.url or request_url)
            if not next_release and len(payload) == self.max_releases_per_page:
                next_release = _set_query_parameter(request_url, "page", str(release_page + 1))
            if next_release and _same_url(next_release, request_url):
                raise ValueError("GitLab release pagination cursor did not advance")
            if next_release and not payload:
                raise ValueError("GitLab release pagination supplied a cursor for an empty page")
            next_state: dict[str, Any] = {
                "projects_started": projects_started,
                "project_queue": queue,
                "current_project": current if next_release else None,
                "projects_next_url": projects_url,
                "last_project_id": last_project_id,
                "projects_seen": projects_seen,
            }
            if next_release:
                next_state["release_next_url"] = next_release
                next_state["release_page"] = release_page + 1
            complete = not next_release and not queue and not projects_url and projects_started
            return SourcePage(
                records=records,
                next_state=next_state,
                complete=complete,
                upstream_count=None,
            )

        if projects_started and not queue and not projects_url:
            return SourcePage(records=(), next_state=dict(state), complete=True, upstream_count=0)

        request_url = projects_url or (
            f"{_API}/projects?visibility=public&order_by=id&sort=asc"
            f"&pagination=keyset&per_page={self.page_size}"
            + (
                f"&id_after={self.initial_project_id}"
                if self.initial_project_id is not None
                else ""
            )
        )
        response = self._get(request_url, "project")
        payload = response.json()
        if not _is_sequence(payload) or len(payload) > self.page_size:
            raise ValueError("GitLab project response is not a bounded JSON array")
        projects = [_project(item) for item in payload]
        if any(item is None for item in projects):
            raise ValueError("GitLab project response contains an invalid project")
        projects = [item for item in projects if item is not None]
        project_ids = [item["id"] for item in projects]
        if (
            last_project_id is not None and project_ids and project_ids[0] <= last_project_id
        ) or any(
            project_ids[index] <= project_ids[index - 1] for index in range(1, len(project_ids))
        ):
            raise ValueError("GitLab project IDs did not advance past the previous keyset page")
        if (
            last_project_id is None
            and self.initial_project_id is not None
            and project_ids
            and project_ids[0] <= self.initial_project_id
        ):
            raise ValueError("GitLab project IDs did not advance past the initial cursor")
        next_projects = _next_link(response.headers, response.url or request_url)
        if next_projects and _same_url(next_projects, request_url):
            raise ValueError("GitLab project pagination cursor did not advance")
        if next_projects and not project_ids:
            raise ValueError("GitLab project pagination supplied a cursor for an empty page")
        range_done = False
        if self.max_project_id is not None:
            bounded_projects = [item for item in projects if item["id"] <= self.max_project_id]
            range_done = len(bounded_projects) < len(projects)
            projects = bounded_projects
            project_ids = [item["id"] for item in projects]
        if self.max_projects is not None:
            remaining = self.max_projects - projects_seen
            projects = projects[: max(remaining, 0)]
            project_ids = [item["id"] for item in projects]
            range_done = range_done or projects_seen + len(projects) >= self.max_projects
        if not next_projects and len(payload) == self.page_size and project_ids:
            next_projects = _set_query_parameter(request_url, "id_after", str(project_ids[-1]))
        if range_done:
            next_projects = None
        if len(projects) > self.max_projects_per_page:
            raise ValueError("GitLab project page exceeded max_projects_per_page")
        next_state = {
            "projects_started": True,
            "project_queue": projects,
            "current_project": None,
            "projects_next_url": next_projects,
            "last_project_id": project_ids[-1] if project_ids else last_project_id,
            "projects_seen": projects_seen + len(projects),
        }
        complete = range_done or (not projects and not next_projects)
        return SourcePage(
            records=(),
            next_state=next_state,
            complete=complete,
            upstream_count=0 if complete else None,
        )

    def _get(self, url: str, kind: str, *, allow_not_found: bool = False) -> HttpResponse | None:
        _safe_api_url(url)
        response: HttpResponse = self.client.get(url, headers={"Accept": "application/json"})
        if allow_not_found and response.status == 404:
            return None
        if response.status != 200:
            raise ValueError(f"GitLab {kind} endpoint returned HTTP {response.status}")
        return response

    def _release_records(
        self, project: Mapping[str, Any], release: Any, release_index: int
    ) -> tuple[SourceRecord, ...]:
        if not isinstance(release, Mapping):
            return ()
        tag = _text(release.get("tag_name"))
        if not tag:
            return ()
        release_url = (
            _text(release.get("_links", {}).get("self"))
            if isinstance(release.get("_links"), Mapping)
            else ""
        )
        if not _safe_release_web_url(release_url):
            release_url = f"https://gitlab.com/{project['path']}/-/releases/{quote(tag, safe='')}"
        else:
            release_path = urlsplit(release_url).path.partition("/-/releases/")[0]
            current_path = unquote(release_path.lstrip("/"))
            if current_path:
                project = {
                    **project,
                    "path": current_path,
                    "web_url": f"https://gitlab.com/{current_path}",
                }
        assets = release.get("assets")
        links = assets.get("links") if isinstance(assets, Mapping) else None
        if not _is_sequence(links):
            return ()
        if len(links) > self.max_assets_per_release:
            raise ValueError("GitLab release exceeds max_assets_per_release")
        context = " ".join(_text(release.get(key)) for key in ("name", "description", "tag_name"))
        result: list[SourceRecord] = []
        for asset_index, asset in enumerate(links):
            if not isinstance(asset, Mapping):
                continue
            label = _text(asset.get("name"))
            url = _text(asset.get("url"))
            path_name = urlsplit(url).path.rsplit("/", 1)[-1]
            filename = path_name or label
            suffix = next(
                (ext for ext in _CHECKPOINT_SUFFIXES if filename.casefold().endswith(ext)),
                "",
            )
            if not suffix or not _safe_asset_url(url):
                continue
            label_stem = label[: -len(suffix)] if label.casefold().endswith(suffix) else label
            model_name = _descriptive_name(filename[: -len(suffix)]) or _descriptive_name(
                label_stem
            )
            if not model_name or not _MODEL_CONTEXT.search(context):
                continue
            if not _identity_matches(model_name, context):
                continue
            asset_key = _text(asset.get("id")) or f"{asset_index}:{url}"
            stable_key = content_hash({"project": project["id"], "tag": tag, "asset": asset_key})[
                :32
            ]
            result.append(
                SourceRecord(
                    source_record_id=f"gitlab-release-asset:{project['id']}:{stable_key}",
                    kind=ArtifactKind.WEIGHTS,
                    canonical_url=url,
                    title=label or filename,
                    published_at=_text(release.get("released_at")) or None,
                    identifiers=(
                        Identifier("gitlab:project-id", str(project["id"])),
                        Identifier("gitlab:project", project["path"]),
                        Identifier("gitlab:release-tag", tag),
                    ),
                    links=(Link(release_url, relation="source_release", crawl=False),),
                    raw={
                        "record_type": "gitlab_release_asset_candidate",
                        "asset": dict(asset),
                        "release": {
                            "tag_name": tag,
                            "released_at": _text(release.get("released_at")),
                        },
                        "project": project,
                        "discovery_basis": "gitlab_public_release_api_asset_link",
                        "is_verified_model_checkpoint": False,
                    },
                    models=(
                        ModelHint(
                            local_id="release-asset-model",
                            name=model_name,
                            status=ModelStatus.CANDIDATE,
                            confidence=0.2,
                            locator=f"releases[{release_index}].assets.links[{asset_index}]",
                        ),
                    ),
                )
            )
        return tuple(result)


def _project(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    project_id = value.get("id")
    path = _text(value.get("path_with_namespace")) or _text(value.get("path"))
    if isinstance(project_id, bool) or not isinstance(project_id, int) or project_id < 1:
        return None
    if not path or any(part in {".", ".."} for part in path.split("/")):
        return None
    if value.get("visibility") not in (None, "public"):
        return None
    return {"id": project_id, "path": path, "web_url": _text(value.get("web_url"))}


def _project_queue(value: Any) -> list[dict[str, Any]]:
    if not _is_sequence(value):
        raise ValueError("project_queue must be an array")
    projects = [_project(item) for item in value]
    if any(item is None for item in projects):
        raise ValueError("project_queue contains an invalid project")
    return [item for item in projects if item is not None]


def _project_state(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    project = _project(value)
    if project is None:
        raise ValueError("current_project is invalid")
    return project


def _safe_api_url(value: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "gitlab.com"
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/api/v4/")
        or parsed.fragment
    ):
        raise ValueError("GitLab pagination URL is outside the public API")


def _safe_asset_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
    )


def _safe_release_web_url(value: str) -> bool:
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "gitlab.com"
        and parsed.username is None
        and parsed.password is None
        and "/-/releases/" in parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def _next_link(headers: Mapping[str, Any], base_url: str) -> str:
    raw = next((str(v) for k, v in headers.items() if k.casefold() == "link"), "")
    for match in _LINK_RE.finditer(raw):
        relation = _REL_RE.search(match.group(2))
        if relation and (relation.group(1) or relation.group(2)) == "next":
            candidate = urljoin(base_url, match.group(1))
            _safe_api_url(candidate)
            return candidate
    return ""


def _set_query_parameter(url: str, key: str, value: str) -> str:
    parsed = urlsplit(url)
    query = [(name, item) for name, item in parse_qsl(parsed.query) if name != key]
    query.append((key, value))
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(query), ""))


def _same_url(left: str, right: str) -> bool:
    left_parts, right_parts = urlsplit(left), urlsplit(right)
    return (
        left_parts.scheme.casefold() == right_parts.scheme.casefold()
        and (left_parts.hostname or "").casefold() == (right_parts.hostname or "").casefold()
        and left_parts.path == right_parts.path
        and sorted(parse_qsl(left_parts.query)) == sorted(parse_qsl(right_parts.query))
    )


def _descriptive_name(value: str) -> str:
    parts = re.split(r"[\s_-]+", value.strip())
    kept: list[str] = []
    for part in parts:
        clean = re.sub(r"[^a-zA-Z0-9.+]", "", part)
        folded = clean.casefold()
        if (
            clean
            and folded not in _GENERIC
            and any(ch.isalpha() for ch in clean)
            and (
                sum(ch.isalpha() for ch in clean) >= 2
                or re.fullmatch(r"\d+(?:\.\d+)?[bmk]", folded)
            )
        ):
            kept.append(clean)
    return " ".join(kept)


def _identity_matches(name: str, context: str) -> bool:
    names = {token for token in re.findall(r"[a-z0-9]+", name.casefold()) if len(token) > 2}
    context_tokens = set(re.findall(r"[a-z0-9]+", context.casefold()))
    return bool(names & context_tokens)


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("pagination state values must be positive integers")
    return value


def _positive_or_zero_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _optional_text(value: Any) -> str:
    text = _text(value)
    return text


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


__all__ = ["GitLabPublicReleaseAssetsSourceAdapter"]
