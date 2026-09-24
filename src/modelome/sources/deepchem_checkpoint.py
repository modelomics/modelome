"""DeepChem's explicitly declared Mol2Vec default pretrained archive."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = "deepchem/deepchem"
_SOURCE_PATH = "deepchem/feat/molecule_featurizers/mol2vec_fingerprint.py"
_VARIABLE = "DEFAULT_PRETRAINED_MODEL_URL"
_EXPECTED_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/trained_models/mol2vec_model_300dim.tar.gz"
)
_HANDLE = "mol2vec-model-300dim"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DeepChemMol2VecCheckpointSourceAdapter:
    """Read DeepChem's literal Mol2Vec archive URL from pinned source code.

    The adapter extracts one assignment with Python's AST and never imports or
    executes upstream code. The tarball is the first-party default archive used
    by ``Mol2VecFingerprint``; the adapter records its exact URL, not its bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only DeepChem's single literal Mol2Vec default pretrained archive. "
        "It is not a catalog of user-trained models or all third-party checkpoints; "
        "the archive is not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "deepchem-mol2vec-checkpoint",
        repository: str = _REPOSITORY,
        branch: str = "master",
        source_path: str = _SOURCE_PATH,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if (
            repository != _REPOSITORY
            or source_path != _SOURCE_PATH
            or not branch.strip()
            or not name.strip()
            or max_response_bytes <= 0
        ):
            raise ValueError(
                "repository/source path are fixed; name, branch, and limit are required"
            )
        self.name = name
        self.repository = repository
        self.branch = branch
        self.source_path = source_path
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "deepchem-mol2vec-checkpoint-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "variable": _VARIABLE,
                "expected_url": _EXPECTED_URL,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_url = (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )
        commit_response = self.client.get(
            commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha") if isinstance(commit, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (),
                {**state, "checked_at": checked_at},
                True,
                upstream_count=state.get("model_count"),
            )

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/x-python,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        url = _literal_url(response.text(), self.name)
        parsed = urlsplit(url)
        if (
            url != _EXPECTED_URL
            or parsed.scheme != "https"
            or parsed.hostname != "deepchemdata.s3-us-west-1.amazonaws.com"
        ):
            raise ValueError(
                f"{self.name}: Mol2Vec URL differs from its verified first-party archive"
            )
        record = self._record(revision, source_url, url, content_hash(response.body))
        return SourcePage(
            (record,),
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(response.body),
                "model_count": 1,
            },
            True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(
        self, revision: str, source_url: str, weight_url: str, source_hash: str
    ) -> SourceRecord:
        namespace = "deepchem:pretrained-checkpoint"
        model_id = f"model:{_HANDLE}"
        model = ModelHint(
            model_id,
            "DeepChem Mol2Vec 300-dimensional pretrained model",
            identifiers=(Identifier(namespace, _HANDLE),),
            aliases=("mol2vec_model_300dim", "Mol2VecFingerprint default"),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{_HANDLE}",
            model_id,
            revision=revision,
            identifiers=(Identifier(f"{namespace}:release", _HANDLE),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "source_sha256": source_hash,
                "source_variable": _VARIABLE,
                "archive_filename": "mol2vec_model_300dim.tar.gz",
                "weight_url": weight_url,
            },
        )
        blob_url = (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{_HANDLE}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(blob_url),
            title=model.name,
            raw={
                "checkpoint_handle": _HANDLE,
                "archive_filename": release.metadata["archive_filename"],
                "weight_url": weight_url,
                "source_variable": _VARIABLE,
            },
            text=(
                "DeepChem's Mol2VecFingerprint implementation declares this as its "
                "default pretrained archive."
            ),
            links=(
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _literal_url(source: str, name: str) -> str:
    try:
        module = ast.parse(source, filename=_SOURCE_PATH)
    except SyntaxError as error:
        raise ValueError(f"{name}: source is not valid Python: {error.msg}") from error
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == _VARIABLE for target in node.targets)
    ]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Constant):
        raise ValueError(f"{name}: expected one literal {_VARIABLE} assignment")
    value = assignments[0].value.value
    if not isinstance(value, str):
        raise ValueError(f"{name}: {_VARIABLE} must be a literal string")
    return value


__all__ = ["DeepChemMol2VecCheckpointSourceAdapter"]
