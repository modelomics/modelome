"""First-party SDXL checkpoint manifest and Stability Hugging Face mappings."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_REPOSITORY = "Stability-AI/generative-models"
_BRANCH = "main"
_SHA = re.compile(r"^[0-9a-f]{40}$")
_ENTRY = re.compile(
    r'"(?P<variant>SDXL-(?:base|refiner)-(?P<version>0\.9|1\.0))"\s*:\s*'
    r"\{(?P<body>.*?)\n\s*\},",
    re.DOTALL,
)
_CHECKPOINT = re.compile(
    r'"ckpt"\s*:\s*"(?:checkpoints/)?'
    r'(?P<filename>sd_xl_(?:base|refiner)_(?:0\.9|1\.0)\.safetensors)"'
)
_HF_MODEL = re.compile(
    r"\[(?P<variant>SDXL-(?:base|refiner)-(?:0\.9|1\.0))\]"
    r"\(https://huggingface\.co/(?P<repo>stabilityai/stable-diffusion-xl-"
    r"(?:base|refiner)-(?:0\.9|1\.0))/?\)"
)


class StabilitySDXLCheckpointSourceAdapter:
    """Join Stability's SDXL demo checkpoint names to first-party HF model IDs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the four base/refiner `.safetensors` checkpoints explicitly "
        "listed by Stability AI's SDXL demo and README. SDXL 0.9 remains gated "
        "under the research license; the source records that restriction and "
        "does not fetch checkpoint bytes. Other Stability model families are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "stability-sdxl-checkpoints",
        repository: str = _REPOSITORY,
        branch: str = _BRANCH,
        readme_path: str = "README.md",
        manifest_path: str = "scripts/demo/sampling.py",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_checkpoints: int = 10,
        client: HttpClient | Any | None = None,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if branch != _BRANCH:
            raise ValueError(f"branch must be {_BRANCH}")
        if readme_path != "README.md" or manifest_path != "scripts/demo/sampling.py":
            raise ValueError("readme_path and manifest_path must use the official SDXL sources")
        if not name.strip() or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (max_response_bytes, max_checkpoints)
        ):
            raise ValueError("name and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.readme_path = readme_path
        self.manifest_path = manifest_path
        self.max_response_bytes = max_response_bytes
        self.max_checkpoints = max_checkpoints
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "stability-sdxl-checkpoints-v1",
                "repository": repository,
                "branch": branch,
                "readme_path": readme_path,
                "manifest_path": manifest_path,
                "max_response_bytes": max_response_bytes,
                "max_checkpoints": max_checkpoints,
                "mapping": "exact-first-party-variant-and-checkpoint-filename",
            }
        )

    @property
    def commit_url(self) -> str:
        return f"https://api.github.com/repos/{self.repository}/commits/{self.branch}"

    def raw_url(self, revision: str, path: str) -> str:
        return f"https://raw.githubusercontent.com/{self.repository}/{revision}/{path}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        commit_response = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if commit_response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit_response.status}")
        commit = commit_response.json()
        revision = commit.get("sha") if isinstance(commit, Mapping) else None
        if not isinstance(revision, str) or not _SHA.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid repository revision")
        contents: dict[str, bytes] = {}
        for path in (self.readme_path, self.manifest_path):
            response = self.client.get(
                self.raw_url(revision, path), headers={"Accept": "text/plain"}
            )
            if response.status != 200:
                raise ValueError(f"{self.name}: {path} returned HTTP {response.status}")
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: {path} exceeds response byte limit")
            contents[path] = response.body
        try:
            readme = contents[self.readme_path].decode("utf-8")
            manifest = contents[self.manifest_path].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: official source files are not UTF-8") from exc
        entries = _parse_sdxl_sources(readme, manifest)
        if not entries or len(entries) > self.max_checkpoints:
            raise ValueError(
                f"{self.name}: expected 1..{self.max_checkpoints} SDXL checkpoint mappings"
            )
        digest = content_hash({path: content_hash(body) for path, body in contents.items()})
        if revision == state.get("completed_revision") and digest == state.get("source_digest"):
            next_state = dict(state)
            return SourcePage(
                (),
                next_state,
                True,
                upstream_count=_nonnegative(state.get("record_count")),
            )
        records = tuple(self._record(entry, revision, digest) for entry in entries)
        next_state = {
            "completed_revision": revision,
            "source_digest": digest,
            "record_count": len(records),
        }
        return SourcePage(
            records,
            next_state,
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, entry: Mapping[str, str], revision: str, digest: str) -> SourceRecord:
        variant = entry["variant"]
        version = entry["version"]
        filename = entry["filename"]
        repo_id = entry["repo"]
        url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='._-')}"
        parsed = urlsplit(url)
        if parsed.hostname != "huggingface.co" or not parsed.path.endswith(f"/{filename}"):
            raise ValueError(f"{self.name}: constructed invalid checkpoint URL")
        role = "base" if "-base-" in variant else "refiner"
        model_id = f"sdxl-{role}"
        model_local_id = f"model:{model_id}"
        model_name = f"Stable Diffusion XL {role}"
        source_locator = (
            f"{self.manifest_path}: {variant} -> {filename}; {self.readme_path}: {repo_id}"
        )
        model = ModelHint(
            local_id=model_local_id,
            name=model_name,
            aliases=(variant, filename),
            identifiers=(
                Identifier("huggingface:model", repo_id),
                Identifier("stabilityai:sdxl-variant", model_id),
            ),
            status=ModelStatus.RELEASED,
            locator=source_locator,
        )
        access = "research-gated" if version == "0.9" else "public-open-rail"
        release_key = f"{version}/{filename}"
        return SourceRecord(
            source_record_id=f"stability-sdxl-checkpoint:{release_key}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"{variant} checkpoint",
            identifiers=(
                Identifier("stabilityai:sdxl-checkpoint", release_key),
                Identifier("huggingface:model", repo_id),
            ),
            links=(
                Link(url, relation="weights", crawl=False),
                Link(f"https://huggingface.co/{repo_id}", relation="model_card", crawl=False),
                Link(
                    self.raw_url(revision, self.readme_path),
                    relation="source_readme",
                    crawl=False,
                    locator=source_locator,
                ),
                Link(
                    self.raw_url(revision, self.manifest_path),
                    relation="source_manifest",
                    crawl=False,
                    locator=source_locator,
                ),
            ),
            raw={
                "record_type": "stability_sdxl_checkpoint",
                "variant": variant,
                "version": version,
                "filename": filename,
                "huggingface_repo": repo_id,
                "checkpoint_url": url,
                "access_status": access,
                "source_revision": revision,
                "source_digest": digest,
                "checkpoint_bytes_fetched": False,
            },
            models=(model,),
            releases=(
                ReleaseHint(
                    local_id=f"release:{release_key}",
                    model_local_id=model_local_id,
                    version=version,
                    identifiers=(Identifier("stabilityai:sdxl-checkpoint", release_key),),
                    metadata={
                        "filename": filename,
                        "checkpoint_url": url,
                        "access_status": access,
                    },
                    locator=source_locator,
                ),
            ),
        )


def _parse_sdxl_sources(readme: str, manifest: str) -> tuple[Mapping[str, str], ...]:
    repo_by_variant = {
        match.group("variant"): match.group("repo") for match in _HF_MODEL.finditer(readme)
    }
    result: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for match in _ENTRY.finditer(manifest):
        variant = match.group("variant")
        checkpoint = _CHECKPOINT.search(match.group("body"))
        repo = repo_by_variant.get(variant)
        if checkpoint is None or repo is None:
            continue
        filename = checkpoint.group("filename")
        expected_role = "base" if "-base-" in variant else "refiner"
        if f"_{expected_role}_" not in filename or variant in seen:
            raise ValueError("official SDXL manifest has an inconsistent or duplicate entry")
        seen.add(variant)
        result.append(
            {
                "variant": variant,
                "version": match.group("version"),
                "filename": filename,
                "repo": repo,
            }
        )
    return tuple(result)


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


__all__ = ["StabilitySDXLCheckpointSourceAdapter"]
