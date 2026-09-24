"""First-party DDLP image/video checkpoint bundles listed in its model zoo."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_REPOSITORY = "taldatech/ddlp"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ROW = re.compile(
    r"^\|\s*(?P<model>DLPv2|DDLP|DiffuseDDLP)\s*\|\s*"
    r"(?P<dataset>OBJ3D|Traffic|PHYRE|CLEVRER) \(128x128\)\s*\|\s*"
    r"\[MEGA\.nz\]\((?P<url>https://mega\.nz/file/[^)#]+#[A-Za-z0-9_-]+)\)\s*\|$"
)
_EXPECTED = {
    ("DLPv2", "OBJ3D"): "wdMUxaQJ",
    ("DLPv2", "Traffic"): "MNljnLCZ",
    ("DDLP", "OBJ3D"): "QcsRSQRD",
    ("DDLP", "Traffic"): "9clHVLTI",
    ("DDLP", "PHYRE"): "UBcl2LgQ",
    ("DDLP", "CLEVRER"): "oUUjXY4B",
    ("DiffuseDDLP", "OBJ3D"): "kJkgAa6a",
    ("DiffuseDDLP", "Traffic"): "4J0DHBBA",
}
_CATEGORIES = {
    "DLPv2": "single-image-object-centric-decomposition",
    "DDLP": "conditional-video-prediction",
    "DiffuseDDLP": "unconditional-video-generation",
}


class DDLPVideoCheckpointSourceAdapter:
    """Index eight exact MEGA bundle IDs from the official DDLP model-zoo table."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the eight DLPv2, DDLP, and DiffuseDDLP pretrained bundles in the "
        "official README table; MEGA links identify encrypted bundles, not individual files."
    )

    def __init__(
        self,
        *,
        name: str = "ddlp-video-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = "main",
        document_path: str = "README.md",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 8,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY or branch != "main" or document_path != "README.md":
            raise ValueError("repository, branch, and document_path must identify official README")
        if not name.strip() or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_response_bytes, max_checkpoints)
        ):
            raise ValueError("name and positive limits are required")
        self.name, self.repository, self.branch = name, repository, branch
        self.document_path = document_path
        self.max_response_bytes, self.max_checkpoints = max_response_bytes, max_checkpoints
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "ddlp-video-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "admission": "exact MEGA links in the first-party pretrained-model table",
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/commits/{self.branch}"

    def raw_url(self, revision: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/{self.document_path}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit = self.client.get(self.commit_url, headers={"Accept": "application/vnd.github+json"})
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        response = self.client.get(self.raw_url(revision), headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response byte limit")
        try:
            readme = response.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        rows = _parse_model_zoo(readme, maximum=self.max_checkpoints)
        if not rows:
            raise ValueError(f"{self.name}: expected all eight model-zoo checkpoint links")
        digest = content_hash(response.body)
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            count = state.get("record_count")
            return SourcePage(
                (),
                dict(state),
                True,
                upstream_count=count
                if isinstance(count, int) and not isinstance(count, bool)
                else 0,
            )
        records = tuple(self._record(row, revision, digest) for row in rows)
        next_state = {
            "completed_revision": revision,
            "source_digest": digest,
            "record_count": len(records),
        }
        return SourcePage(
            records, next_state, True, upstream_count=len(records), authoritative_snapshot=True
        )

    def _record(self, row: Mapping[str, str], revision: str, digest: str) -> SourceRecord:
        model_type, dataset, url = row["model"], row["dataset"], row["url"]
        bundle_id = urlsplit(url).path.rsplit("/", 1)[-1]
        model_value = f"{model_type}/{dataset}/128x128"
        local_id = f"model:{model_type.casefold()}:{dataset.casefold()}:128"
        release_id = f"{model_type.casefold()}/{dataset.casefold()}-128x128"
        category = _CATEGORIES[model_type]
        locator = f"{self.document_path}: Model Zoo - Pretrained Models; {model_type} / {dataset}"
        return SourceRecord(
            source_record_id=f"taldatech-ddlp:{model_type}:{dataset}:128x128",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{model_type} {dataset} 128x128 checkpoint bundle",
            identifiers=(Identifier("taldatech:ddlp-bundle", bundle_id),),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://github.com/{self.repository}", relation="repository", crawl=False),
                Link(self.raw_url(revision), relation="model_card", crawl=False, locator=locator),
            ),
            raw={
                "record_type": "ddlp_video_checkpoint_bundle",
                "model_type": model_type,
                "dataset": dataset,
                "resolution": "128x128",
                "category": category,
                "bundle_id": bundle_id,
                "checkpoint_url": url,
                "source_revision": revision,
                "source_sha256": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(
                ModelHint(
                    local_id=local_id,
                    name=f"{model_type} {dataset} 128x128",
                    aliases=(model_value, f"{model_type} {dataset}"),
                    identifiers=(Identifier("taldatech:ddlp-model", model_value),),
                    status=ModelStatus.RELEASED,
                    locator=locator,
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_id}",
                    model_local_id=local_id,
                    version="README-listed pretrained bundle",
                    identifiers=(Identifier("taldatech:ddlp-bundle", bundle_id),),
                    metadata={
                        "bundle_id": bundle_id,
                        "checkpoint_url": url,
                        "dataset": dataset,
                        "resolution": "128x128",
                        "category": category,
                    },
                    locator=locator,
                ),
            ),
        )


def _parse_model_zoo(readme: str, *, maximum: int) -> tuple[Mapping[str, str], ...]:
    start = readme.find("## Model Zoo - Pretrained Models")
    end = readme.find("## Interactive Graphical User Interface", start + 1)
    if start < 0 or end < 0:
        return ()
    rows: list[Mapping[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in readme[start:end].splitlines():
        match = _ROW.fullmatch(line.strip())
        if match is None:
            continue
        model_type, dataset, url = (match.group(key) for key in ("model", "dataset", "url"))
        file_id = urlsplit(url).path.rsplit("/", 1)[-1]
        key = model_type, dataset
        if key not in _EXPECTED or file_id != _EXPECTED[key]:
            raise ValueError(f"unexpected DDLP checkpoint identity or MEGA asset for {key}")
        if key in seen:
            raise ValueError(f"duplicate DDLP model-zoo entry: {key}")
        seen.add(key)
        rows.append({"model": model_type, "dataset": dataset, "url": url})
        if len(rows) > maximum:
            raise ValueError(f"DDLP model-zoo checkpoint count exceeds {maximum}")
    if len(rows) != len(_EXPECTED) or seen != set(_EXPECTED):
        raise ValueError("DDLP model-zoo table is missing expected checkpoint mappings")
    return tuple(rows)


__all__ = ["DDLPVideoCheckpointSourceAdapter"]
