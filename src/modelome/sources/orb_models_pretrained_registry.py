"""Orbital Materials' first-party pretrained model URL registry."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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

_REPOSITORY = "orbital-materials/orb-models"
_BRANCH = "main"
_SOURCE_PATH = "orb_models/forcefield/pretrained.py"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WEIGHT_HOST = "orbitalmaterials-public-models.s3.us-west-1.amazonaws.com"


class OrbModelsPretrainedRegistryAdapter:
    """Read literal checkpoint URLs from Orb's published loader functions."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Indexes named loader functions whose first-party source declares a literal "
        "public checkpoint URL in the weights_path default. It reads GitHub source "
        "metadata only and never requests checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "orb-models-pretrained-registry",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "orb-models-pretrained-registry-v1",
                "repository": _REPOSITORY,
                "branch": _BRANCH,
                "source_path": _SOURCE_PATH,
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{_REPOSITORY}/commits/{_BRANCH}"

    def _source_url(self, revision: str) -> str:
        return f"https://raw.githubusercontent.com/{_REPOSITORY}/{revision}/{_SOURCE_PATH}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        payload = commit_response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage((), next_state, True, upstream_count=state.get("model_count"))

        source_url = self._source_url(revision)
        source_response = self.client.get(source_url, headers={"Accept": "text/plain"})
        if source_response.status != 200:
            raise ValueError(
                f"{self.name}: pretrained source returned HTTP {source_response.status}"
            )
        if len(source_response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        entries = _parse_pretrained_urls(source_response.text(), self.name)
        records = tuple(
            _record(model_name, url, revision, source_url, source_response.body)
            for model_name, url in entries
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(source_response.body),
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_pretrained_urls(source_text: str, source: str) -> tuple[tuple[str, str], ...]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError as error:
        raise ValueError(f"{source}: pretrained source is not valid Python") from error
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    aliases: dict[str, str] = {}
    registry: ast.Dict | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "ORB_PRETRAINED_MODELS"
                for target in node.targets
            ):
                if isinstance(node.value, ast.Dict):
                    registry = node.value
                else:
                    raise ValueError(f"{source}: ORB_PRETRAINED_MODELS is not a literal dictionary")
            elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                if isinstance(node.value, ast.Name):
                    aliases[target] = node.value.id
    if registry is None:
        raise ValueError(f"{source}: ORB_PRETRAINED_MODELS registry is missing")

    def resolve(name: str) -> str:
        visited: set[str] = set()
        while name in aliases:
            if name in visited:
                raise ValueError(f"{source}: cyclic loader alias")
            visited.add(name)
            name = aliases[name]
        return name

    rows: list[tuple[str, str]] = []
    for key, value_node in zip(registry.keys, registry.values, strict=True):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ValueError(f"{source}: registry keys must be literal strings")
        if not isinstance(value_node, ast.Name):
            raise ValueError(f"{source}: registry values must name loader functions")
        model_name = key.value
        loader_name = resolve(value_node.id)
        node = functions.get(loader_name)
        if node is None:
            raise ValueError(f"{source}: registry loader {loader_name!r} is undefined")
        positional = (*node.args.posonlyargs, *node.args.args)
        names = [argument.arg for argument in positional]
        if "weights_path" not in names:
            raise ValueError(f"{source}: {model_name!r} loader has no weights_path")
        index = names.index("weights_path")
        default_index = index - (len(positional) - len(node.args.defaults))
        if default_index < 0:
            raise ValueError(
                f"{source}: {model_name!r} loader requires an unspecified weights_path"
            )
        value = node.args.defaults[default_index]
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ValueError(f"{source}: {model_name!r} loader has no literal checkpoint URL")
        url = value.value
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != _WEIGHT_HOST
            or parsed.query
            or parsed.fragment
            or not parsed.path.endswith(".ckpt")
            or not parsed.path.startswith("/forcefields/")
        ):
            raise ValueError(f"{source}: unsafe or unexpected checkpoint URL for {model_name}")
        rows.append((model_name, url))
    if not rows:
        raise ValueError(f"{source}: no literal pretrained checkpoint URLs found")
    names = [name for name, _ in rows]
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: duplicate model name")
    return tuple(rows)


def _record(
    model_name: str, url: str, revision: str, source_url: str, source: bytes
) -> SourceRecord:
    filename = urlsplit(url).path.rsplit("/", 1)[-1]
    model_id = f"model:{model_name}"
    identity = Identifier("orb:checkpoint", model_name)
    model = ModelHint(
        model_id,
        f"Orb {model_name.replace('_', ' ')}",
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
            "checkpoint_filename": filename,
            "checkpoint_url": url,
            "repository": _REPOSITORY,
            "revision": revision,
            "source_path": _SOURCE_PATH,
            "source_sha256": content_hash(source),
            "binary_reachability_checked": False,
        },
        text=f"Orb loader {model_name} declares checkpoint URL {url}",
        identifiers=(identity,),
        links=(
            Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
            Link(source_url, "source_implementation", crawl=False),
            Link(f"https://github.com/{_REPOSITORY}", "source_repository", crawl=False),
        ),
        models=(model,),
        releases=(
            ReleaseHint(
                f"release:{model_name}",
                model_id,
                revision=revision,
                identifiers=(Identifier("orb:checkpoint:release", model_name),),
                metadata={
                    "repository": _REPOSITORY,
                    "revision": revision,
                    "model_name": model_name,
                    "checkpoint_filename": filename,
                    "checkpoint_url": url,
                    "source_path": _SOURCE_PATH,
                    "source_sha256": content_hash(source),
                    "binary_reachability_checked": False,
                },
            ),
        ),
    )
