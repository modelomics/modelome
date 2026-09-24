"""Enumerate PoolFormer checkpoint URLs from timm's archived v0.6.13 config.

The current timm registry keeps the PoolFormer presets on Hugging Face Hub, but
the v0.6.13 source also declared the original sail-sg release URLs. This narrow
adapter records those historical references without fetching checkpoint bytes.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODELS = ("poolformer_s12", "poolformer_s24", "poolformer_s36", "poolformer_m36", "poolformer_m48")
_URL_PREFIX = "https://github.com/sail-sg/poolformer/releases/download/v1.0/"
_PATH = "timm/models/poolformer.py"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TimmLegacyPoolFormerSourceAdapter:
    """Read the five literal PoolFormer URLs from timm's v0.6.13 source."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the five original PoolFormer checkpoint URLs declared in "
        "timm v0.6.13. It does not enumerate current Hub revisions or fetch weights."
    )

    def __init__(
        self,
        *,
        name: str = "timm-legacy-poolformer-v0613",
        repository: str = "huggingface/pytorch-image-models",
        tag: str = "v0.6.13",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.repository = _repository(repository)
        self.tag = _required_text(tag, "tag")
        self.client = client or HttpClient(max_response_bytes=2 * 1024 * 1024)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {"adapter": "timm-legacy-poolformer-v1", "repository": self.repository, "tag": self.tag}
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/commits/{quote(self.tag, safe='')}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: tag did not resolve to a SHA-1 revision")
        checked_at = _isoformat(self.clock())
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=state.get("model_count"),
            )

        source_url = self._raw_url(revision)
        source_response: HttpResponse = self.client.get(source_url)
        if source_response.status != 200:
            raise ValueError(f"{self.name}: archived config returned HTTP {source_response.status}")
        if len(source_response.body) > 2 * 1024 * 1024:
            raise ValueError(f"{self.name}: archived config exceeds 2097152 bytes")
        source = source_response.text()
        urls = _poolformer_urls(source, self.name)
        records = tuple(
            self._record(name, urls[name], revision, content_hash(source_response.body))
            for name in _MODELS
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(source_response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{_PATH}"
        )

    def _record(
        self, model_name: str, weights_url: str, revision: str, source_hash: str
    ) -> SourceRecord:
        model_id = f"model:{model_name}"
        release_id = f"release:{model_name}:v0.6.13"
        code_url = f"{self.repository_url}/blob/{quote(revision, safe='')}/{_PATH}"
        model = ModelHint(
            local_id=model_id,
            name=model_name,
            identifiers=(Identifier("timm:model", model_name),),
            status=ModelStatus.RELEASED,
            locator=f"{_PATH}:default_cfgs[{model_name!r}]",
        )
        release = ReleaseHint(
            local_id=release_id,
            model_local_id=model_id,
            version="v0.6.13",
            revision=revision,
            identifiers=(Identifier("timm:checkpoint-url", weights_url),),
            metadata={
                "repository": self.repository,
                "tag": self.tag,
                "module_path": _PATH,
                "config_key": model_name,
                "weights_url": weights_url,
                "source_sha256": source_hash,
            },
            locator=model.locator,
        )
        return SourceRecord(
            source_record_id=f"legacy-poolformer:{model_name}:v0.6.13",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(code_url),
            title=f"{model_name} pretrained weights (timm v0.6.13)",
            raw={
                "repository": self.repository,
                "tag": self.tag,
                "revision": revision,
                "module_path": _PATH,
                "source_sha256": source_hash,
                "model": model_name,
                "weights_url": weights_url,
            },
            text=f"{model_name} pretrained weights declared by timm {self.tag}",
            identifiers=(Identifier("timm:checkpoint-url", weights_url),),
            links=(
                Link(self.repository_url, relation="source_repository", crawl=False),
                Link(code_url, relation="model_definition", locator=model.locator, crawl=False),
                Link(weights_url, relation="weights", locator=model.locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _poolformer_urls(source: str, name: str) -> dict[str, str]:
    try:
        tree = ast.parse(source, filename=_PATH)
    except SyntaxError as error:
        raise ValueError(f"{name}: cannot parse archived config: {error.msg}") from error
    defaults = next(
        (
            statement.value
            for statement in tree.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "default_cfgs"
                for target in statement.targets
            )
        ),
        None,
    )
    if (
        not isinstance(defaults, ast.Call)
        or not isinstance(defaults.func, ast.Name)
        or defaults.func.id != "dict"
    ):
        raise ValueError(f"{name}: archived PoolFormer config has unsupported default_cfgs shape")
    result: dict[str, str] = {}
    for keyword in defaults.keywords:
        if keyword.arg not in _MODELS or not isinstance(keyword.value, ast.Call):
            continue
        if not isinstance(keyword.value.func, ast.Name) or keyword.value.func.id != "_cfg":
            continue
        url_node = next((item.value for item in keyword.value.keywords if item.arg == "url"), None)
        url = (
            url_node.value.strip()
            if isinstance(url_node, ast.Constant) and isinstance(url_node.value, str)
            else ""
        )
        expected = f"{_URL_PREFIX}{keyword.arg}.pth.tar"
        if url != expected:
            raise ValueError(f"{name}: {keyword.arg} has an unexpected checkpoint URL")
        result[keyword.arg] = url
    if set(result) != set(_MODELS):
        missing = sorted(set(_MODELS) - set(result))
        raise ValueError(f"{name}: archived config is missing checkpoint URLs for {missing}")
    return result


def _repository(value: str) -> str:
    result = _required_text(value, "repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", result):
        raise ValueError("repository must be an owner/name pair")
    return result


def _required_text(value: Any, field: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
