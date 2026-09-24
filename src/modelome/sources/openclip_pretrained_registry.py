"""OpenCLIP's source-declared direct pretrained checkpoint registry."""

from __future__ import annotations

import ast
import hashlib
import json
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

_REPOSITORY = "mlfoundations/open_clip"
_SOURCE_PATH = "src/open_clip/pretrained.py"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HELPERS = {"_pcfg", "_slpcfg", "_apcfg", "_mccfg", "_clapcfg", "_mc2cfg", "_pecfg"}


class OpenCLIPPretrainedRegistrySourceAdapter:
    """Enumerate direct, non-OpenAI URLs in OpenCLIP's official source registry.

    HF-only entries are excluded to avoid re-enumerating Hub repositories.
    OpenAI-tagged entries and OpenAI's hosted checkpoint URLs are also excluded.
    """

    coverage_limitation = (
        "Covers only OpenCLIP registry entries with a direct HTTP(S) checkpoint URL, "
        "excluding OpenAI-owned checkpoints and HF-only entries. It does not cover "
        "local/user checkpoints or sources that omit a direct URL."
    )

    def __init__(
        self,
        *,
        name: str = "openclip-direct-pretrained-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        client: HttpClient | Any | None = None,
        max_source_bytes: int = 2 * 1024 * 1024,
        max_models: int = 2_000,
    ) -> None:
        if repository != _REPOSITORY or not name.strip() or not branch.strip():
            raise ValueError("repository is fixed; name and branch are required")
        if max_source_bytes < 1 or max_models < 1:
            raise ValueError("source and model limits must be positive")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_source_bytes = max_source_bytes
        self.max_models = max_models
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "openclip-direct-pretrained-v1",
                "repository": repository,
                "branch": branch,
                "source_path": _SOURCE_PATH,
                "admission": "direct URL; exclude OpenAI-owned and HF-only entries",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/"
            f"{quote(self.branch, safe='')}"
        )
        commit_response = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit_payload = commit_response.json()
        commit = commit_payload.get("sha") if isinstance(commit_payload, Mapping) else None
        if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
            raise ValueError(f"{self.name}: commit endpoint returned an invalid commit SHA")
        if commit == state.get("completed_revision"):
            return SourcePage((), dict(state), True, upstream_count=state.get("model_count"))

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{commit}/{_SOURCE_PATH}"
        )
        source_response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if source_response.status != 200:
            raise ValueError(
                f"{self.name}: pretrained source returned HTTP {source_response.status}"
            )
        if len(source_response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: pretrained source exceeds byte limit")
        source = source_response.body.decode("utf-8")
        checkpoints = _parse_registry(source)
        if len(checkpoints) > self.max_models:
            raise ValueError(f"{self.name}: registry exceeds model limit")
        source_sha256 = hashlib.sha256(source_response.body).hexdigest()
        records = tuple(
            _record(self.name, self.repository, commit, source_sha256, model, tag, config)
            for model, tag, config in checkpoints
        )
        return SourcePage(
            records,
            {"completed_revision": commit, "model_count": len(records)},
            True,
            upstream_count=len(records),
        )


def _parse_registry(source: str) -> tuple[tuple[str, str, Mapping[str, Any]], ...]:
    """Interpret only literal dictionaries and known config-helper calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("OpenCLIP pretrained registry is not valid Python") from error
    names: dict[str, Any] = {}
    registry: Any = None
    for statement in tree.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not all(isinstance(target, ast.Name) for target in statement.targets):
            continue
        value = _literal(statement.value, names)
        for target in statement.targets:
            assert isinstance(target, ast.Name)
            names[target.id] = value
            if target.id == "_PRETRAINED":
                registry = value
    if not isinstance(registry, Mapping):
        raise ValueError("OpenCLIP source does not contain a literal _PRETRAINED mapping")
    entries: list[tuple[str, str, Mapping[str, Any]]] = []
    for model_name, tags in registry.items():
        if not isinstance(model_name, str) or not isinstance(tags, Mapping):
            continue
        for tag, config in tags.items():
            if not isinstance(tag, str) or not isinstance(config, Mapping):
                continue
            url = config.get("url")
            if not isinstance(url, str) or not _admitted_url(url, tag):
                continue
            entries.append((model_name, tag, config))
    return tuple(entries)


def _literal(node: ast.AST, names: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.Dict):
        return {
            _literal(key, names): _literal(value, names)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "dict":
            values = {
                keyword.arg: _literal(keyword.value, names)
                for keyword in node.keywords
                if keyword.arg is not None
            }
            if node.args and isinstance(node.args[0], ast.Dict):
                values = {**_literal(node.args[0], names), **values}
            return values
        if isinstance(node.func, ast.Name) and node.func.id in _HELPERS:
            values: dict[str, Any] = {}
            if node.args:
                values["url"] = _literal(node.args[0], names)
            values.update(
                {
                    keyword.arg: _literal(keyword.value, names)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
            return values
    return None


def _admitted_url(value: str, tag: str) -> bool:
    url = canonicalize_url(value)
    parts = urlsplit(url)
    host = (parts.hostname or "").casefold()
    return (
        parts.scheme == "https"
        and bool(host)
        and tag.casefold() != "openai"
        and host != "openaipublic.azureedge.net"
        and host != "huggingface.co"
        and not host.endswith(".huggingface.co")
    )


def _record(
    source: str,
    repository: str,
    commit: str,
    source_sha256: str,
    model_name: str,
    tag: str,
    config: Mapping[str, Any],
) -> SourceRecord:
    identity = f"{model_name}:{tag}"
    record_id = f"{repository}/{_SOURCE_PATH}:{identity}"
    model_local_id = f"{identity}#model"
    release_local_id = f"{identity}#release"
    url = canonicalize_url(str(config["url"]))
    repo_url = f"https://github.com/{repository}"
    source_url = f"{repo_url}/blob/{commit}/{_SOURCE_PATH}"
    model = ModelHint(
        local_id=model_local_id,
        name=f"{model_name} ({tag})",
        identifiers=(Identifier("openclip:pretrained-model", model_name),),
        aliases=(tag,),
        status=ModelStatus.RELEASED,
        locator=f"_PRETRAINED[{model_name!r}][{tag!r}]",
    )
    release = ReleaseHint(
        local_id=release_local_id,
        model_local_id=model_local_id,
        version=tag,
        identifiers=(Identifier("openclip:pretrained-checkpoint", identity),),
        metadata={"architecture": model_name, "direct_url": url},
        locator=f"_PRETRAINED[{model_name!r}][{tag!r}].url",
    )
    raw = {
        "registry": {
            "repository": repository,
            "commit": commit,
            "path": _SOURCE_PATH,
            "sha256": source_sha256,
        },
        "entry": {"model": model_name, "tag": tag, "config": dict(config)},
    }
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=url,
        title=f"{model_name} ({tag})",
        raw=raw,
        text=json.dumps(raw, ensure_ascii=False, sort_keys=True),
        identifiers=(Identifier("openclip:checkpoint", identity),),
        links=(
            Link(url, relation="checkpoint", locator=release.locator, crawl=False),
            Link(source_url, relation="source_registry", locator="_PRETRAINED", crawl=False),
        ),
        models=(model,),
        releases=(release,),
    )


__all__ = ["OpenCLIPPretrainedRegistrySourceAdapter", "_parse_registry"]
