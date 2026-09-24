"""PyG's source-declared SchNet QM9 pretrained archive and target identities."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

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

_ARCHIVE_URL = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"


class PyGSchNetQM9RegistrySourceAdapter(StaticJsonCheckpointRegistrySourceAdapter):
    """Expand the exact target-specific models inside PyG's declared SchNet archive."""

    coverage_limitation = (
        "Covers the literal QM9 target map and one HTTP ZIP archive declared by PyG's "
        "SchNet loader. Each target resolves to the documented best_model member. "
        "The archive is linked but never fetched or inspected."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "pytorch-geometric-schnet-qm9-checkpoints")
        kwargs.setdefault("repository", "pyg-team/pytorch_geometric")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault("source_path", "torch_geometric/nn/models/schnet.py")
        kwargs.setdefault("provider_namespace", "pytorch-geometric:schnet-qm9-checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pyg-schnet-qm9-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal QM9 target map and SchNet archive/member pattern",
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
        targets, archive = _parse_schnet_source(response.text(), self.name, self.source_path)
        if len(targets) > self.max_entries:
            raise ValueError(f"{self.name}: registry exceeds {self.max_entries} entries")
        records = tuple(
            _schnet_record(self, target, member, archive, revision, response.body)
            for target, member in targets
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


def _parse_schnet_source(
    document: str, source: str, path: str
) -> tuple[tuple[tuple[str, str], ...], str]:
    try:
        module = ast.parse(document, filename=path)
    except SyntaxError as error:
        raise ValueError(f"{source}: source is not valid Python: {error.msg}") from error
    maps: list[ast.expr | None] = []
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "qm9_target_dict"
        ) or (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "qm9_target_dict"
        ):
            maps.append(node.value)
    if len(maps) != 1 or not isinstance(maps[0], ast.Dict):
        raise ValueError(f"{source}: expected one literal qm9_target_dict mapping")
    targets: list[tuple[int, str]] = []
    for key, value in zip(maps[0].keys, maps[0].values, strict=True):
        if (
            not isinstance(key, ast.Constant)
            or not isinstance(key.value, int)
            or isinstance(key.value, bool)
            or not isinstance(value, ast.Constant)
            or not isinstance(value.value, str)
            or not value.value.strip()
        ):
            raise ValueError(f"{source}: target map must contain integer/string literals")
        targets.append((key.value, value.value))
    if not targets or len({index for index, _ in targets}) != len(targets):
        raise ValueError(f"{source}: target map is empty or has duplicate indexes")
    if len({name for _, name in targets}) != len(targets):
        raise ValueError(f"{source}: target map has duplicate names")

    classes = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SchNet"
    ]
    if len(classes) != 1:
        raise ValueError(f"{source}: expected exactly one SchNet class")
    urls = [
        node.value.value
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "url"
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if urls != [_ARCHIVE_URL]:
        raise ValueError(f"{source}: SchNet archive URL does not match expected literal")
    parsed = urlsplit(urls[0])
    if parsed.scheme != "http" or parsed.hostname != "www.quantum-machine.org":
        raise ValueError(f"{source}: expected the literal first-party SchNet archive URL")
    members = tuple(
        (name, f"trained_schnet_models/qm9_{name}/best_model")
        for _, name in sorted(targets)
    )
    return members, urls[0]


def _schnet_record(
    adapter: PyGSchNetQM9RegistrySourceAdapter,
    target: str,
    member: str,
    archive: str,
    revision: str,
    source: bytes,
) -> SourceRecord:
    handle = f"schnet:qm9:{target}"
    model_id = f"model:{handle}"
    source_url = adapter.blob_url(revision)
    model = ModelHint(
        local_id=model_id,
        name=f"SchNet QM9 {target} checkpoint",
        aliases=(handle,),
        identifiers=(Identifier(adapter.provider_namespace, handle),),
        status=ModelStatus.RELEASED,
        locator=f"{adapter.source_path}:SchNet.from_qm9_pretrained",
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
            "weight_url": archive,
            "archive_member": member,
        },
        locator=f"{adapter.source_path}:SchNet.from_qm9_pretrained",
    )
    return SourceRecord(
        source_record_id=f"checkpoint:{handle}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=canonicalize_url(archive),
        title=model.name,
        raw={
            "repository": adapter.repository,
            "revision": revision,
            "source_path": adapter.source_path,
            "source_sha256": content_hash(source),
            "checkpoint_handle": handle,
            "qm9_target": target,
            "weight_url": archive,
            "archive_member": member,
        },
        text=f"PyG pretrained SchNet model for QM9 target {target}.",
        identifiers=(Identifier(adapter.provider_namespace, handle),),
        links=(
            Link(source_url, "model_card", crawl=False, model_local_ids=(model_id,)),
            Link(
                adapter.repository_url,
                "source_implementation",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(archive, "weights", crawl=False, model_local_ids=(model_id,)),
        ),
        models=(model,),
        releases=(release,),
    )
