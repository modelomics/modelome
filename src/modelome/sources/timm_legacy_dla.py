"""Enumerate historical DLA checkpoint URLs from timm v0.6.13."""

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
_REPOSITORY = "huggingface/pytorch-image-models"
_TAG = "v0.6.13"
_PATH = "timm/models/dla.py"
_CONFIG_COUNT = 12
_URL_PREFIX = "https://github.com/"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TimmLegacyDLASourceAdapter:
    """Read literal DLA URLs from one immutable timm release module."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the 12 literal checkpoint URLs declared in timm v0.6.13's DLA "
        "pretrained config. It does not enumerate current Hub files or fetch weights."
    )

    def __init__(
        self,
        *,
        name: str = "timm-legacy-dla-v0613",
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.client = client or HttpClient(max_response_bytes=2 * 1024 * 1024)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "timm-legacy-dla-v1",
                "repository": _REPOSITORY,
                "tag": _TAG,
                "module_path": _PATH,
                "config_count": _CONFIG_COUNT,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_TAG}"

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{_REPOSITORY}"

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
            count = state.get("model_count")
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=count if isinstance(count, int) and count >= 0 else None,
            )

        source_url = (
            f"https://raw.githubusercontent.com/{_REPOSITORY}/"
            f"{quote(revision, safe='')}/{_PATH}"
        )
        source_response: HttpResponse = self.client.get(source_url)
        if source_response.status != 200:
            raise ValueError(f"{self.name}: archived config returned HTTP {source_response.status}")
        if len(source_response.body) > 2 * 1024 * 1024:
            raise ValueError(f"{self.name}: archived config exceeds 2097152 bytes")
        source_hash = content_hash(source_response.body)
        configs = _dla_urls(source_response.text(), self.name)
        records = tuple(
            self._record(model_name, weights_url, revision, source_hash)
            for model_name, weights_url in configs
        )
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": source_hash,
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, model_name: str, weights_url: str, revision: str, source_hash: str
    ) -> SourceRecord:
        model_id = f"model:{model_name}"
        locator = f"{_PATH}:default_cfgs[{model_name!r}]"
        code_url = f"{self.repository_url}/blob/{quote(revision, safe='')}/{_PATH}"
        model = ModelHint(
            local_id=model_id,
            name=model_name,
            identifiers=(Identifier("timm:pretrained-config", model_name),),
            status=ModelStatus.RELEASED,
            locator=locator,
        )
        release = ReleaseHint(
            local_id=f"release:{model_name}:v0.6.13",
            model_local_id=model_id,
            version=_TAG,
            revision=revision,
            identifiers=(Identifier("timm:checkpoint-url", weights_url),),
            metadata={
                "repository": _REPOSITORY,
                "tag": _TAG,
                "module_path": _PATH,
                "config_key": model_name,
                "weights_url": weights_url,
                "source_sha256": source_hash,
            },
            locator=locator,
        )
        return SourceRecord(
            source_record_id=f"legacy-dla:{model_name}:v0.6.13",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(code_url),
            title=f"{model_name} pretrained weights (timm v0.6.13)",
            raw={
                "repository": _REPOSITORY,
                "tag": _TAG,
                "revision": revision,
                "module_path": _PATH,
                "source_sha256": source_hash,
                "model": model_name,
                "weights_url": weights_url,
            },
            text=f"{model_name} pretrained weights declared by timm {_TAG}",
            identifiers=(Identifier("timm:checkpoint-url", weights_url),),
            links=(
                Link(self.repository_url, relation="source_repository", crawl=False),
                Link(code_url, relation="model_definition", locator=locator, crawl=False),
                Link(weights_url, relation="weights", locator=locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _dla_urls(source: str, name: str) -> tuple[tuple[str, str], ...]:
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
    entries: list[tuple[ast.expr | None, ast.expr]] = []
    if isinstance(defaults, ast.Dict):
        entries = list(zip(defaults.keys, defaults.values, strict=True))
    elif (
        isinstance(defaults, ast.Call)
        and isinstance(defaults.func, ast.Name)
        and defaults.func.id == "dict"
    ):
        entries = [(ast.Constant(item.arg), item.value) for item in defaults.keywords if item.arg]
    else:
        raise ValueError(f"{name}: archived DLA config has unsupported default_cfgs shape")

    result: list[tuple[str, str]] = []
    for key, value in entries:
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            continue
        url = _url_value(value)
        if url.startswith(_URL_PREFIX):
            result.append((key.value, url))
    if len(result) != _CONFIG_COUNT:
        raise ValueError(
            f"{name}: expected {_CONFIG_COUNT} literal DLA URLs; found {len(result)}"
        )
    if len({model_name for model_name, _ in result}) != len(result):
        raise ValueError(f"{name}: archived config contains duplicate DLA keys")
    return tuple(result)


def _url_value(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        for keyword in node.keywords:
            if keyword.arg == "url" and isinstance(keyword.value, ast.Constant):
                return keyword.value.value if isinstance(keyword.value.value, str) else ""
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "url"
                and isinstance(value, ast.Constant)
            ):
                return value.value if isinstance(value.value, str) else ""
    return ""


def _required_text(value: Any, field: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{field} is required")
    return result


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
