"""SevenNet's first-party pretrained checkpoint release map."""

from __future__ import annotations

import ast
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

_REPOSITORY = "MDIL-SNU/SevenNet"
_BRANCH = "main"
_CONST_PATH = "sevenn/_const.py"
_UTIL_PATH = "sevenn/util.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_HOST = "github.com"
_RELEASE_PREFIX = f"https://github.com/{_REPOSITORY}/releases/download/"


class SevenNetPretrainedRegistryAdapter:
    """Index exact release assets in SevenNet's download map and model-name map."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the current CHECKPOINT_DOWNLOAD_LINKS entries in SevenNet's first-party "
        "source when their GitHub release metadata confirms the exact asset. Older models "
        "without a release-download mapping are outside coverage. No checkpoint bytes "
        "are requested."
    )

    def __init__(
        self,
        *,
        name: str = "sevennet-pretrained-registry",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "sevennet-pretrained-registry-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "paths": [_CONST_PATH, _UTIL_PATH],
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"

    def _raw_url(self, revision: str, path: str) -> str:
        return f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{path}"

    def _release_api_url(self, tag: str) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/releases/tags/{quote(tag, safe='')}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha", "") if isinstance(commit, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        const_url = self._raw_url(revision, _CONST_PATH)
        util_url = self._raw_url(revision, _UTIL_PATH)
        const_response = self.client.get(const_url, headers={"Accept": "text/plain"})
        util_response = self.client.get(util_url, headers={"Accept": "text/plain"})
        for label, response in (
            ("constant source", const_response),
            ("utility source", util_response),
        ):
            if response.status != 200:
                raise ValueError(f"{self.name}: {label} returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: {label} exceeds {self.max_response_bytes} bytes")
        entries = _parse_checkpoint_map(const_response.text(), util_response.text(), self.name)
        asset_metadata = self._verify_release_assets(entries)
        records = tuple(
            _record(
                model_name,
                checkpoint_constant,
                url,
                asset_metadata[url],
                revision,
                const_url,
                util_url,
            )
            for model_name, checkpoint_constant, url in entries
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "const_source_url": const_url,
                "util_source_url": util_url,
                "const_source_sha256": content_hash(const_response.body),
                "util_source_sha256": content_hash(util_response.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _verify_release_assets(
        self, entries: tuple[tuple[str, str, str], ...]
    ) -> dict[str, Mapping[str, Any]]:
        grouped: dict[str, list[tuple[str, str]]] = {}
        for model_name, _, url in entries:
            parts = urlsplit(url).path.split("/")
            tag = parts[-2]
            filename = parts[-1]
            grouped.setdefault(tag, []).append((model_name, filename))
        verified: dict[str, Mapping[str, Any]] = {}
        for tag, expected_assets in grouped.items():
            response = self.client.get(
                self._release_api_url(tag), headers={"Accept": "application/vnd.github+json"}
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: release metadata for {tag} returned "
                    f"HTTP {response.status}"
                )
            payload = response.json()
            assets = payload.get("assets", []) if isinstance(payload, Mapping) else []
            by_name = {
                asset.get("name"): asset
                for asset in assets
                if isinstance(asset, Mapping) and isinstance(asset.get("name"), str)
            }
            for _, filename in expected_assets:
                asset = by_name.get(filename)
                expected_url = f"{_RELEASE_PREFIX}{tag}/{filename}"
                if asset is None or asset.get("browser_download_url") != expected_url:
                    raise ValueError(f"{self.name}: release {tag} does not declare {filename}")
                if not isinstance(asset.get("id"), int) or not isinstance(asset.get("size"), int):
                    raise ValueError(f"{self.name}: release asset {filename} lacks metadata")
                verified[f"{_RELEASE_PREFIX}{tag}/{filename}"] = asset
        return verified


def _literal_assignments(text: str, source: str) -> dict[str, ast.AST]:
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        raise ValueError(f"{source}: source is not valid Python") from error
    values: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    return values


def _fstring(node: ast.AST, variables: Mapping[str, str], source: str) -> str:
    if not isinstance(node, ast.JoinedStr):
        raise ValueError(f"{source}: expected a literal f-string")
    pieces: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            pieces.append(value.value)
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            replacement = variables.get(value.value.id)
            if replacement is None:
                raise ValueError(f"{source}: unresolved f-string variable {value.value.id!r}")
            pieces.append(replacement)
        else:
            raise ValueError(f"{source}: unsupported checkpoint URL expression")
    return "".join(pieces)


def _parse_checkpoint_map(
    const_text: str, util_text: str, source: str
) -> tuple[tuple[str, str, str], ...]:
    constants = _literal_assignments(const_text, source)
    utilities = _literal_assignments(util_text, source)
    links_node = constants.get("CHECKPOINT_DOWNLOAD_LINKS")
    model_names_node = utilities.get("checkpoint_to_name")
    if not isinstance(links_node, ast.Dict) or not isinstance(model_names_node, ast.Dict):
        raise ValueError(f"{source}: expected literal checkpoint and model-name mappings")
    prefix_node = constants.get("_git_prefix")
    if not isinstance(prefix_node, ast.Constant) or not isinstance(prefix_node.value, str):
        raise ValueError(f"{source}: release URL prefix is not literal")
    prefix = prefix_node.value
    if prefix != f"https://{_RELEASE_HOST}/{_REPOSITORY}/releases/download":
        raise ValueError(f"{source}: unexpected GitHub release URL prefix")
    names: dict[str, str] = {}
    for key, value in zip(model_names_node.keys, model_names_node.values, strict=True):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, str)
            or not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
        ):
            raise ValueError(f"{source}: model-name mapping must contain literal strings")
        names[key.value] = value.value

    rows: list[tuple[str, str, str]] = []
    for key, value in zip(links_node.keys, links_node.values, strict=True):
        if not isinstance(key, ast.Name) or not isinstance(value, ast.JoinedStr):
            raise ValueError(f"{source}: checkpoint links must map constants to literal f-strings")
        checkpoint_constant = key.id
        url = _fstring(value, {"_git_prefix": prefix}, source)
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != _RELEASE_HOST
            or parsed.path.count("/") < 6
            or not parsed.path.endswith(".pth")
            or not url.startswith(f"{_RELEASE_PREFIX}")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{source}: unexpected checkpoint URL for {checkpoint_constant}")
        model_name = names.get(checkpoint_constant)
        if model_name is None:
            raise ValueError(f"{source}: no canonical model name for {checkpoint_constant}")
        rows.append((model_name, checkpoint_constant, url))
    if not rows:
        raise ValueError(f"{source}: no release checkpoint links found")
    if len({name for name, _, _ in rows}) != len(rows):
        raise ValueError(f"{source}: duplicate canonical model name")
    if len({url for _, _, url in rows}) != len(rows):
        raise ValueError(f"{source}: duplicate checkpoint URL")
    return tuple(rows)


def _record(
    model_name: str,
    checkpoint_constant: str,
    url: str,
    asset: Mapping[str, Any],
    revision: str,
    const_url: str,
    util_url: str,
) -> SourceRecord:
    filename = urlsplit(url).path.rsplit("/", 1)[-1]
    tag = urlsplit(url).path.split("/")[-2]
    model_id = f"model:{model_name}"
    identity = Identifier("sevennet:checkpoint", model_name)
    model = ModelHint(
        model_id,
        f"SevenNet {model_name}",
        aliases=(filename,),
        identifiers=(identity,),
        status=ModelStatus.RELEASED,
    )
    return SourceRecord(
        source_record_id=f"checkpoint:{model_name}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=canonicalize_url(url),
        title=model.name,
        raw={
            "model_name": model_name,
            "checkpoint_constant": checkpoint_constant,
            "checkpoint_filename": filename,
            "checkpoint_url": url,
            "release_tag": tag,
            "github_asset_id": asset["id"],
            "size_bytes": asset["size"],
            "repository": _REPOSITORY,
            "source_revision": revision,
            "binary_reachability_checked": False,
        },
        text=f"SevenNet checkpoint map declares {model_name} at {url}",
        identifiers=(identity,),
        links=(
            Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
            Link(const_url, "source_implementation", crawl=False),
            Link(util_url, "source_model_names", crawl=False),
            Link(f"https://github.com/{_REPOSITORY}", "source_repository", crawl=False),
        ),
        models=(model,),
        releases=(
            ReleaseHint(
                f"release:{model_name}",
                model_id,
                revision=tag,
                identifiers=(Identifier("sevennet:release", tag),),
                metadata={
                    "repository": _REPOSITORY,
                    "release_tag": tag,
                    "model_name": model_name,
                    "checkpoint_filename": filename,
                    "github_asset_id": asset["id"],
                    "size_bytes": asset["size"],
                    "source_revision": revision,
                    "binary_reachability_checked": False,
                },
            ),
        ),
    )
