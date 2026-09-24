"""PyG's literal DimeNet and DimeNet++ QM9 checkpoint bundle locations."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpResponse
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
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
    _header,
    _isoformat,
    _nonnegative_int,
    _text,
)

_EXPECTED_BASES = {
    "DimeNet": "https://github.com/klicperajo/dimenet/raw/master/pretrained/dimenet",
    "DimeNetPlusPlus": (
        "https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/dimenet_pp"
    ),
}
_CHECKPOINT_FILES = (
    "checkpoint",
    "ckpt.data-00000-of-00002",
    "ckpt.data-00001-of-00002",
    "ckpt.index",
)


class PyGDimeNetCheckpointSourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Enumerate PyG's explicit QM9 target checkpoint bundles for DimeNet variants."""

    coverage_limitation = (
        "Covers the literal QM9 target mapping and the two exact DimeNet artifact roots "
        "in PyG's dimenet.py. Each identity points to four checkpoint files documented "
        "by the loader. It does not fetch or inspect model bytes."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "pytorch-geometric-dimenet-qm9-checkpoints")
        kwargs.setdefault("repository", "pyg-team/pytorch_geometric")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "torch_geometric/nn/models/dimenet.py")
        kwargs.setdefault("provider_namespace", "pytorch-geometric:dimenet-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pyg-dimenet-checkpoints-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": (
                    "literal QM9 target map, two pinned DimeNet roots, "
                    "four checkpoint parts"
                ),
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
        response: HttpResponse = self.client.get(
            self.raw_url(revision),
            headers={"Accept": "text/x-python,text/plain"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        target_names, roots = _parse_dimenet_registry(response.text(), self.name, self.source_path)
        model_count = len(target_names) * len(roots)
        if model_count > self.max_entries:
            raise ValueError(f"{self.name}: registry exceeds {self.max_entries} entries")
        records = tuple(
            _dimenet_record(self, variant, target, base, revision, response.body)
            for variant, base in roots.items()
            for target in target_names
        )
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


def _parse_dimenet_registry(
    document: str, source: str, path: str
) -> tuple[tuple[str, ...], dict[str, str]]:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: source is not valid Python: {error.msg}") from error
    target_maps = []
    for node in module.body:
        is_annotation = (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "qm9_target_dict"
        )
        is_assignment = (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "qm9_target_dict"
        )
        if is_annotation or is_assignment:
            target_maps.append(node.value)
    if len(target_maps) != 1 or not isinstance(target_maps[0], ast.Dict):
        raise ValueError(f"{source}: expected one literal qm9_target_dict mapping")
    targets: list[tuple[int, str]] = []
    for key_node, value_node in zip(target_maps[0].keys, target_maps[0].values, strict=True):
        if (
            not isinstance(key_node, ast.Constant)
            or not isinstance(key_node.value, int)
            or isinstance(key_node.value, bool)
            or not isinstance(value_node, ast.Constant)
            or not isinstance(value_node.value, str)
            or not value_node.value.strip()
        ):
            raise ValueError(f"{source}: QM9 target entries must be literal integer/string pairs")
        targets.append((key_node.value, value_node.value))
    if not targets or len({key for key, _ in targets}) != len(targets):
        raise ValueError(f"{source}: QM9 target mapping is empty or has duplicate indexes")
    if len({name for _, name in targets}) != len(targets):
        raise ValueError(f"{source}: QM9 target mapping has duplicate target names")

    roots: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef) or node.name not in _EXPECTED_BASES:
            continue
        urls = [
            value
            for item in node.body
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "url"
            and (value := _literal_string(item.value)) is not None
        ]
        if len(urls) != 1 or urls[0].rstrip("/") != _EXPECTED_BASES[node.name]:
            raise ValueError(
                f"{source}: {node.name} checkpoint base does not match expected literal"
            )
        roots[node.name] = urls[0].rstrip("/")
    if roots != _EXPECTED_BASES:
        raise ValueError(f"{source}: expected DimeNet and DimeNetPlusPlus checkpoint roots")
    return tuple(name for _, name in sorted(targets)), roots


def _literal_string(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Tuple):
        parts = [_literal_string(element) for element in node.elts]
        if all(part is not None for part in parts):
            return "".join(part for part in parts if part is not None)
    return None


def _dimenet_record(
    adapter: PyGDimeNetCheckpointSourceAdapter,
    variant: str,
    target: str,
    base: str,
    revision: str,
    source: bytes,
) -> SourceRecord:
    target_key = "dimenet++" if variant == "DimeNetPlusPlus" else "dimenet"
    handle = f"{target_key}:qm9:{target}"
    model_id = f"model:{handle}"
    urls = tuple(f"{base}/{target}/{filename}" for filename in _CHECKPOINT_FILES)
    source_url = adapter.blob_url(revision)
    model = ModelHint(
        local_id=model_id,
        name=f"{variant} QM9 {target} checkpoint",
        aliases=(handle,),
        identifiers=(Identifier(adapter.provider_namespace, handle),),
        status=ModelStatus.RELEASED,
        locator=f"{adapter.source_path}:{variant}.from_qm9_pretrained",
    )
    release = ReleaseHint(
        local_id=f"release:{handle}",
        model_local_id=model_id,
        revision=revision,
        identifiers=(Identifier(f"{adapter.provider_namespace}:release", handle),),
        metadata={
            "repository": adapter.repository,
            "revision": revision,
            "qm9_target": target,
            "model_variant": variant,
            "checkpoint_files": urls,
        },
        locator=f"{adapter.source_path}:{variant}.from_qm9_pretrained",
    )
    return SourceRecord(
        source_record_id=f"checkpoint:{handle}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=canonicalize_url(urls[-1]),
        title=model.name,
        raw={
            "repository": adapter.repository,
            "revision": revision,
            "source_path": adapter.source_path,
            "source_sha256": content_hash(source),
            "checkpoint_handle": handle,
            "qm9_target": target,
            "model_variant": variant,
            "weight_urls": urls,
        },
        text=f"PyG pretrained {variant} model for QM9 target {target}.",
        identifiers=(Identifier(adapter.provider_namespace, handle),),
        links=(
            Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
            Link(
                adapter.repository_url,
                "source_implementation",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            *(Link(url, "weights", crawl=False, model_local_ids=(model_id,)) for url in urls),
        ),
        models=(model,),
        releases=(release,),
    )
