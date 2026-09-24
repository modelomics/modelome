"""Pinned ingestion of TorchXRayVision's source-declared model weight map."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _Checkpoint,
    _checkpoint_url,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)


@dataclass(frozen=True, slots=True)
class _XrvCheckpoint:
    handle: str
    url: str
    locator: str
    aliases: tuple[str, ...]


class TorchXRayVisionRegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Read literal ``model_urls`` checkpoint declarations without executing code."""

    coverage_limitation = (
        "Covers direct checkpoint URLs in TorchXRayVision's core model_urls map at "
        "one observed Git commit. It does not import package code, follow baseline "
        "model download helpers, infer papers, or download checkpoint bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "torchxrayvision-pretrained-registry")
        kwargs.setdefault("repository", "mlmed/torchxrayvision")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "torchxrayvision/models.py")
        kwargs.setdefault("provider_namespace", "torchxrayvision:weight")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torchxrayvision-model-urls-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal model_urls weights_url and direct aliases",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = _isoformat(self.clock())
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            if etag := _header(commit_response.headers, "etag"):
                next_state["commit_etag"] = etag
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )

        response = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/x-python,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: registry returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: registry exceeds {self.max_response_bytes} bytes")
        checkpoints = _parse_model_urls(
            response.text(), source=self.name, path=self.source_path,
            maximum=self.max_entries,
        )
        records = tuple(
            self._record(checkpoint, revision, response.body) for checkpoint in checkpoints
        )
        if not records:
            raise ValueError(f"{self.name}: model_urls contains no checkpoint entries")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, checkpoint: _XrvCheckpoint, revision: str, source: bytes):
        record = super()._record(
            _Checkpoint(checkpoint.handle, checkpoint.url, checkpoint.locator),
            revision,
            source,
        )
        model = replace(record.models[0], aliases=checkpoint.aliases)
        return replace(
            record,
            text=f"TorchXRayVision pretrained checkpoint: {checkpoint.handle}",
            models=(model,),
        )


def _parse_model_urls(
    document: str, *, source: str, path: str, maximum: int
) -> tuple[_XrvCheckpoint, ...]:
    """Accept only subscript assignments with literal maps or direct aliases."""
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: registry is not valid Python: {error.msg}") from error

    entries: dict[str, tuple[str, str]] = {}
    aliases: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not _model_urls_key(target):
            continue
        key = ast.literal_eval(target.slice)
        if not isinstance(key, str):
            raise ValueError(f"{source}: model_urls keys must be literal strings")
        if key in entries or key in aliases:
            raise ValueError(f"{source}: duplicate model_urls key {key!r}")
        value = node.value
        if isinstance(value, ast.Dict):
            weights = [
                item for field, item in zip(value.keys, value.values, strict=True)
                if isinstance(field, ast.Constant) and field.value == "weights_url"
            ]
            if (
                len(weights) != 1
                or not isinstance(weights[0], ast.Constant)
                or not isinstance(weights[0].value, str)
            ):
                continue
            url = _checkpoint_url(weights[0].value, source, key)
            entries[key] = (url, f"{path}:L{node.lineno}")
        elif _model_urls_key(value):
            target_key = ast.literal_eval(value.slice)
            if not isinstance(target_key, str):
                raise ValueError(f"{source}: aliases must use literal string keys")
            aliases[key] = target_key

    if not entries:
        raise ValueError(f"{source}: no literal model_urls weights_url entries")
    if len(entries) + len(aliases) > maximum:
        raise ValueError(f"{source}: registry exceeds {maximum} entries")

    grouped: dict[str, list[str]] = {key: [key] for key in entries}
    for alias, target in aliases.items():
        if target not in entries:
            raise ValueError(f"{source}: alias {alias!r} targets unknown model {target!r}")
        grouped[target].append(alias)

    result = []
    for key, (url, locator) in entries.items():
        names = grouped[key]
        # Keep the documented constructor handle where available; short names
        # are also retained as aliases on the resulting model record.
        canonical = next((name for name in names if name.startswith(("densenet", "resnet"))), key)
        result.append(
            _XrvCheckpoint(
                handle=canonical,
                url=url,
                locator=locator,
                aliases=tuple(name for name in names if name != canonical),
            )
        )
    return tuple(result)


def _model_urls_key(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "model_urls"
    )
