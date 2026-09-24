"""Pinned inventory for the original M3GNet Materials Project potential."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

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
    _header,
    _isoformat,
    _text,
)
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
    _assignment_value,
    _is_mapping_assignment,
    _literal_string,
)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID = "MP-2021.2.8-EFS"
_FILES = (
    "checkpoint",
    "m3gnet.json",
    "m3gnet.index",
    "m3gnet.data-00000-of-00001",
)
_SOURCE_PATH = "m3gnet/models/_m3gnet.py"
_EXPECTED_BASE = (
    "https://raw.githubusercontent.com/materialsvirtuallab/m3gnet/main/"
    "pretrained/{model_name}/{filename}"
)


def _parse_registry(document: str, source: str) -> tuple[str, ...]:
    try:
        module = ast.parse(document, filename=_SOURCE_PATH)
    except SyntaxError as error:
        raise ValueError(f"{source}: source is not valid Python: {error.msg}") from error

    files_assignments = [
        node for node in module.body if _is_mapping_assignment(node, "MODEL_FILES")
    ]
    base_assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "GITHUB_RAW_LINK"
            for target in node.targets
        )
    ]
    if len(files_assignments) != 1 or len(base_assignments) != 1:
        raise ValueError(f"{source}: expected one MODEL_FILES and GITHUB_RAW_LINK assignment")
    base = _literal_string(_assignment_value(base_assignments[0]))
    if base != _EXPECTED_BASE:
        raise ValueError(f"{source}: unexpected M3GNet artifact URL template")

    value = _assignment_value(files_assignments[0])
    try:
        mapping = ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"{source}: MODEL_FILES must be a literal mapping") from error
    if not isinstance(mapping, dict) or tuple(mapping) != (_MODEL_ID,):
        raise ValueError(f"{source}: expected only the verified {_MODEL_ID} model")
    files = mapping[_MODEL_ID]
    if not isinstance(files, list) or tuple(files) != _FILES:
        raise ValueError(f"{source}: unexpected checkpoint file inventory")
    return _FILES


class M3GNetLegacyCheckpointSourceAdapter(StaticPythonCheckpointRegistrySourceAdapter):
    """Read M3GNet's literal legacy model-file map without importing the package."""

    coverage_limitation = (
        "Covers only the original M3GNet MP-2021.2.8-EFS TensorFlow checkpoint bundle. "
        "The first-party map declares four files for one model; the adapter records "
        "their exact URLs and does not download checkpoint bytes. The successor MatGL "
        "catalog is hosted separately on Hugging Face."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "m3gnet-legacy-mp-2021-2-8-efs")
        kwargs.setdefault("repository", "materialyzeai/m3gnet")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", _SOURCE_PATH)
        kwargs.setdefault("mapping_variable", "MODEL_FILES")
        kwargs.setdefault("provider_namespace", "m3gnet:checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "m3gnet-legacy-checkpoint-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "model_id": _MODEL_ID,
                "files": _FILES,
                "artifact_template": _EXPECTED_BASE,
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
                upstream_count=1,
            )

        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/x-python,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds response limit")
        filenames = _parse_registry(response.text(), self.name)
        record = self._record(revision, response.body, filenames)
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "source_url": self.raw_url(revision),
            "source_sha256": content_hash(response.body),
            "model_count": 1,
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(self, revision: str, source: bytes, filenames: tuple[str, ...]) -> SourceRecord:
        model_id = f"model:{_MODEL_ID}"
        provider_id = Identifier("m3gnet:checkpoint", _MODEL_ID)
        artifact_urls = tuple(
            _EXPECTED_BASE.format(model_name=_MODEL_ID, filename=filename) for filename in filenames
        )
        model = ModelHint(
            local_id=model_id,
            name=f"M3GNet {_MODEL_ID}",
            identifiers=(provider_id,),
            aliases=("M3GNet Materials Project universal potential",),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            local_id=f"release:{_MODEL_ID}",
            model_local_id=model_id,
            revision=revision,
            identifiers=(Identifier("m3gnet:release", _MODEL_ID),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": content_hash(source),
                "model_files": list(filenames),
                "artifact_urls": list(artifact_urls),
            },
            locator=f"{self.source_path}:MODEL_FILES[{_MODEL_ID!r}]",
        )
        source_url = (
            f"https://github.com/{self.repository}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )
        return SourceRecord(
            source_record_id=f"checkpoint:m3gnet:{_MODEL_ID}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(
                f"https://github.com/{self.repository}/tree/{quote(revision, safe='')}/"
                f"pretrained/{_MODEL_ID}"
            ),
            title=model.name,
            raw={
                "checkpoint_handle": _MODEL_ID,
                "model_files": list(filenames),
                "artifact_urls": list(artifact_urls),
                "source_variable": "MODEL_FILES",
            },
            text=(
                "The archived first-party M3GNet source declares this MP-2021.2.8-EFS "
                "checkpoint bundle and its four files."
            ),
            links=(
                Link(source_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
                *(
                    Link(url, "weights", locator=filename, crawl=False, model_local_ids=(model_id,))
                    for filename, url in zip(filenames, artifact_urls, strict=True)
                ),
            ),
            models=(model,),
            releases=(release,),
        )
