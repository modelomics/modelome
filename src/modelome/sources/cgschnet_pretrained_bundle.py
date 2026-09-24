"""cG-SchNet's two explicitly named pretrained models and shared DOI bundle."""

from __future__ import annotations

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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = "atomistic-machine-learning/cG-SchNet"
_SOURCE_PATH = "published_data/README.md"
_BUNDLE_URL = "https://dx.doi.org/10.14279/depositonce-14978"
_MODELS = ("comp_relenergy", "gap_relenergy")
_DOI_LINK = re.compile(r"\[DOI 10\.14279/depositonce-14978\]\((https://[^)]+)\)")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CGSchNetPretrainedBundleSourceAdapter:
    """Index the two cG-SchNet models while retaining only their shared DOI link."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the two pretrained cG-SchNet model names listed in the first-party README. "
        "Both records point to the shared DOI landing page; no per-model artifact URL or "
        "binary reachability is asserted."
    )

    def __init__(
        self,
        *,
        name: str = "cgschnet-pretrained-bundle",
        repository: str = _REPOSITORY,
        branch: str = "main",
        source_path: str = _SOURCE_PATH,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 2 * 1024 * 1024,
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
        self.name, self.repository, self.branch = name, repository, branch
        self.source_path = source_path
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "cgschnet-pretrained-bundle-v1",
                "repository": repository,
                "branch": branch,
                "source_path": source_path,
                "bundle_url": _BUNDLE_URL,
                "model_handles": _MODELS,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

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
        payload = commit_response.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked_at}, True,
                upstream_count=state.get("model_count"),
            )

        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{quote(revision, safe='')}/{quote(self.source_path, safe='/')}"
        )
        response = self.client.get(source_url, headers={"Accept": "text/markdown,text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        _validate_bundle_reference(response.text(), self.name)
        source_hash = content_hash(response.body)
        records = tuple(
            self._record(revision, source_url, source_hash, handle) for handle in _MODELS
        )
        return SourcePage(
            records,
            {
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": source_hash,
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(
        self, revision: str, source_url: str, source_hash: str, handle: str
    ) -> SourceRecord:
        model_id = f"model:{handle}"
        namespace = "cgschnet:checkpoint"
        model = ModelHint(
            model_id,
            f"cG-SchNet {handle} pretrained model",
            identifiers=(Identifier(namespace, handle),),
            aliases=(handle,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={
                "repository": self.repository,
                "revision": revision,
                "source_path": self.source_path,
                "bundle_reference_url": _BUNDLE_URL,
                "bundle_name": "two pretrained cG-SchNet models",
                "per_model_artifact_url": None,
                "binary_reachability_checked": False,
                "source_sha256": source_hash,
            },
        )
        blob_url = (
            f"{self.repository_url}/blob/{quote(revision, safe='')}/"
            f"{quote(self.source_path, safe='/')}"
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(_BUNDLE_URL),
            title=model.name,
            raw={
                "checkpoint_handle": handle,
                "bundle_reference_url": _BUNDLE_URL,
                "per_model_artifact_url": None,
            },
            text=(
                f"The cG-SchNet README identifies {handle} as one of two pretrained models "
                "in the shared DOI bundle."
            ),
            links=(
                Link(_BUNDLE_URL, "bundle_reference", crawl=False, model_local_ids=(model_id,)),
                Link(blob_url, "source_implementation", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_bundle_reference(document: str, source: str) -> None:
    matches = _DOI_LINK.findall(document)
    if matches != [_BUNDLE_URL]:
        raise ValueError(f"{source}: expected the exact cG-SchNet pretrained bundle DOI")
    if not re.search(r"zip-file\s+containing\s+two\s+pretrained\s+cG-SchNet\s+models", document):
        raise ValueError(f"{source}: expected the documented two-model bundle description")
    if not all(handle in document for handle in _MODELS):
        raise ValueError(f"{source}: expected both documented cG-SchNet model names")


__all__ = ["CGSchNetPretrainedBundleSourceAdapter"]
