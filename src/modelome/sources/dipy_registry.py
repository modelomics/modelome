"""DIPY's first-party neuroimaging pretrained-weight fetcher inventory.

The adapter reads the literal ``fetch_*_weights = _make_fetcher(...)``
declarations from DIPY's fetcher module. It never imports that module or fetches
any checkpoint bytes. Only public Figshare file IDs declared beside model-weight
fetchers are admitted; example data fetchers and other hosts are ignored.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

Clock = Callable[[], datetime]
_SHA = re.compile(r"^[0-9a-f]{40}$")
_FILE_ID = re.compile(r"^[0-9]{4,}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_NAME = re.compile(r"^fetch_[a-z0-9_]+_weights$")
_FIGSHARE_BASE = "https://ndownloader.figshare.com/files/"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DipyPretrainedRegistrySourceAdapter:
    """Resolve and parse DIPY's literal pretrained-weight fetcher declarations."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only DIPY fetcher declarations named fetch_*_weights that use the "
        "literal public Figshare file endpoint and provide model-weight descriptions. "
        "It does not cover arbitrary DIPY datasets, non-Figshare model assets, or "
        "download any model bytes."
    )

    def __init__(
        self,
        *,
        name: str = "dipy-pretrained-models",
        repository: str = "dipy/dipy",
        branch: str = "master",
        source_path: str = "dipy/data/fetcher.py",
        max_source_bytes: int = 8 * 1024 * 1024,
        max_models: int = 100,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name, self.repository, self.branch, self.source_path = (
            name,
            repository,
            branch,
            source_path,
        )
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
            raise ValueError("repository must be owner/name")
        if not source_path or source_path.startswith("/") or ".." in source_path.split("/"):
            raise ValueError("source_path must be a safe relative path")
        if max_source_bytes <= 0 or max_models <= 0:
            raise ValueError("source limits must be positive")
        self.max_source_bytes, self.max_models = max_source_bytes, max_models
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "dipy-pretrained-fetchers-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "max_source_bytes": max_source_bytes,
                "max_models": max_models,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        path = quote(self.source_path, safe="/")
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(self.commit_url, headers={"Accept": "application/vnd.github+json"})
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        sha = commit.json().get("sha") if isinstance(commit.json(), Mapping) else None
        if not isinstance(sha, str) or not _SHA.fullmatch(sha):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if sha == state.get("completed_revision"):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked},
                complete=True,
                upstream_count=_count(state.get("model_count")),
            )
        response = self.client.get(
            self.raw_url(sha), headers={"Accept": "text/x-python,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: fetcher source returned HTTP {response.status}")
        if len(response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: fetcher source exceeds {self.max_source_bytes} bytes")
        entries = _parse_fetchers(response.text(), self.name, self.source_path, self.max_models)
        if not entries:
            raise ValueError(f"{self.name}: no eligible pretrained weight fetchers")
        revision_url = f"{self.repository_url}/blob/{sha}/{quote(self.source_path, safe='/')}"
        records = tuple(
            self._record(name, doc, assets, md5s, loc, sha, revision_url)
            for name, doc, assets, md5s, loc in entries
        )
        next_state = {
            "completed_revision": sha,
            "checked_at": checked,
            "source_url": self.raw_url(sha),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self,
        fetcher: str,
        doc: str,
        assets: tuple[tuple[str, str], ...],
        md5s: tuple[str, ...],
        locator: str,
        revision: str,
        source_url: str,
    ) -> SourceRecord:
        model_id = fetcher.removeprefix("fetch_").removesuffix("_weights")
        local_id = f"dipy:{fetcher}#model"
        model = ModelHint(
            local_id=local_id,
            name=model_id.replace("_", " "),
            aliases=(fetcher,),
            identifiers=(Identifier("dipy:model", fetcher),),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        links = []
        artifact_rows = []
        for index, (file_id, filename) in enumerate(assets):
            url = f"{_FIGSHARE_BASE}{file_id}"
            md5 = md5s[index] if len(md5s) == len(assets) else None
            links.append(
                Link(
                    url,
                    relation="weights",
                    locator=f"{locator}.args[{index}]",
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            artifact_rows.append({"file_id": file_id, "filename": filename, "url": url, "md5": md5})
        release = ReleaseHint(
            local_id=f"{local_id}:release:{revision}",
            model_local_id=local_id,
            revision=revision,
            identifiers=(Identifier("dipy:fetcher", fetcher),),
            metadata={
                "fetcher": fetcher,
                "description": doc,
                "artifacts": artifact_rows,
                "source_revision": revision,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"{self.name}:{fetcher}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(source_url),
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "fetcher": fetcher,
                "description": doc,
                "artifacts": artifact_rows,
            },
            text=doc,
            identifiers=(Identifier("dipy:fetcher", fetcher),),
            links=tuple(links),
            models=(model,),
            releases=(release,),
        )


def _parse_fetchers(
    source: str, name: str, path: str, maximum: int
) -> tuple[tuple[str, str, tuple[tuple[str, str], ...], tuple[str, ...], str], ...]:
    try:
        module = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise ValueError(f"{name}: source is not valid Python") from exc
    result = []
    for node in module.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue
        fetcher = node.targets[0].id
        if not _NAME.fullmatch(fetcher):
            continue
        call = node.value
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "_make_fetcher"
            or len(call.args) < 5
        ):
            continue
        try:
            declared_name = ast.literal_eval(call.args[0])
            base = ast.literal_eval(call.args[2])
            ids = ast.literal_eval(call.args[3])
            filenames = (
                [_filename_literal(arg) for arg in call.args[4].elts]
                if isinstance(call.args[4], (ast.List, ast.Tuple))
                else []
            )
            keywords = {
                kw.arg: ast.literal_eval(kw.value)
                for kw in call.keywords
                if kw.arg in {"doc", "md5_list"}
            }
        except (ValueError, TypeError):
            continue
        if (
            declared_name != fetcher
            or base != _FIGSHARE_BASE
            or not isinstance(ids, list)
            or not isinstance(filenames, list)
            or not ids
            or len(ids) != len(filenames)
        ):
            continue
        doc = keywords.get("doc")
        if not isinstance(doc, str) or " model weights " not in f" {doc.casefold()} ":
            continue
        if len(result) >= maximum:
            raise ValueError(f"{name}: source exceeds {maximum} eligible model fetchers")
        assets = []
        for file_id, filename in zip(ids, filenames, strict=True):
            if not isinstance(file_id, str) or not _FILE_ID.fullmatch(file_id):
                assets = []
                break
            if not isinstance(filename, str) or not filename or "/" in filename or "\\" in filename:
                assets = []
                break
            assets.append((file_id, filename))
        if not assets:
            continue
        md5_raw = keywords.get("md5_list", [])
        md5s = (
            tuple(md5_raw)
            if isinstance(md5_raw, list)
            and all(isinstance(x, str) and _MD5.fullmatch(x) for x in md5_raw)
            else ()
        )
        result.append((fetcher, doc, tuple(assets), md5s, f"{path}:L{node.lineno}"))
    if len({entry[0] for entry in result}) != len(result):
        raise ValueError(f"{name}: duplicate fetcher declarations")
    return tuple(result)


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _filename_literal(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        node = node.args[0]
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None
