"""ConceptNet Numberbatch's first-party release artifact index."""

from __future__ import annotations

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
_TABLE_ROW = re.compile(
    r"^\|\s*(?:\*\*)?(?P<version>[0-9.]+)(?:\*\*)?\s*\|(.*)\|$"
)
_MARKDOWN_REFERENCE = re.compile(r"\[([^\]]+)\]\[([^\]]+)\]")
_REFERENCE = re.compile(r"^\[(?P<name>[a-z0-9-]+)\]:\s+(?P<url>\S+)\s*$", re.M)
_ALLOWED_HOST = "conceptnet.s3.amazonaws.com"
_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
_REPOSITORY = "commonsense/conceptnet-numberbatch"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ConceptNetNumberbatchSourceAdapter:
    """Enumerate explicitly linked release files from the official README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers artifacts listed in the official README release table; the 17.02 "
        "multilingual link is skipped because it points to a 17.04 file."
    )

    def __init__(
        self,
        *,
        name: str = "conceptnet-numberbatch",
        repository: str = _REPOSITORY,
        branch: str = "master",
        provider_namespace: str = "conceptnet-numberbatch:embedding",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = 100,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY or not name.strip() or not branch.strip():
            raise ValueError("name, official repository, and branch are required")
        if not provider_namespace.strip() or min(max_response_bytes, max_entries) <= 0:
            raise ValueError("namespace and response/entry limits are required")
        self.name, self.repository, self.branch = name, repository, branch
        self.provider_namespace = provider_namespace
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "conceptnet-numberbatch-v1",
                "repository": repository,
                "branch": branch,
                "readme": "README.md",
                "table": "Downloads",
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision = state.get("catalog_revision")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            commit = self.client.get(
                f"https://api.github.com/repos/{self.repository}/commits/"
                f"{quote(self.branch, safe='')}",
                headers={"Accept": "application/vnd.github+json"},
            )
            if commit.status != 200:
                raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
            payload = commit.json()
            revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid catalog revision")

        readme = self.client.get(
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/README.md",
            headers={"Accept": "text/plain"},
        )
        if readme.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {readme.status}")
        if len(readme.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        try:
            readme_text = readme.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc

        records = _parse_readme(readme_text, self.name, self.provider_namespace)
        if not records:
            raise ValueError(f"{self.name}: no release artifacts found")
        if len(records) > self.max_entries:
            raise ValueError(f"{self.name}: release artifact count exceeds configured limit")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        return SourcePage(
            records=tuple(self._record(item, revision) for item in records),
            next_state={"catalog_revision": revision, "checked_at": checked_at},
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, item: Mapping[str, str], revision: str) -> SourceRecord:
        version, scope, artifact = item["version"], item["scope"], item["artifact"]
        identity = f"{version}:{scope}:{artifact}"
        model_id = f"model:{identity}"
        url = item["url"]
        model = ModelHint(
            model_id,
            f"ConceptNet Numberbatch {version} {scope} {artifact}",
            identifiers=(Identifier(self.provider_namespace, identity),),
            aliases=(urlsplit(url).path.rsplit("/", 1)[-1],),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{identity}",
            model_id,
            version=version,
            identifiers=(Identifier(f"{self.provider_namespace}:release", identity),),
            metadata={
                "scope": scope,
                "artifact_format": artifact,
                "license_url": _LICENSE_URL,
                "catalog_revision": revision,
                "weight_urls": (url,),
            },
        )
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=model.name,
            raw={
                **item,
                "license_url": _LICENSE_URL,
                "catalog_revision": revision,
            },
            text=(
                f"Official ConceptNet Numberbatch {version} {scope} embeddings "
                f"distributed under CC BY-SA 4.0."
            ),
            identifiers=(Identifier(self.provider_namespace, identity),),
            links=(
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                Link(
                    f"https://github.com/{self.repository}",
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(_LICENSE_URL, "license", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_readme(
    readme: str, source: str, namespace: str
) -> tuple[dict[str, str], ...]:
    del namespace  # Parsing stays independent of source identity configuration.
    references = {match["name"]: match["url"] for match in _REFERENCE.finditer(readme)}
    rows: list[dict[str, str]] = []
    in_download_table = False
    for line in readme.splitlines():
        if line.strip() == "## Downloads":
            in_download_table = True
            continue
        if in_download_table and line.startswith("## "):
            break
        if not in_download_table:
            continue
        match = _TABLE_ROW.fullmatch(line.strip())
        if not match:
            continue
        version = match["version"]
        cells = [cell.strip() for cell in match.group(2).split("|")]
        if len(cells) != 3:
            continue
        for scope, cell in zip(
            ("multilingual", "english-only", "hdf5"), cells, strict=True
        ):
            linked = _MARKDOWN_REFERENCE.search(cell)
            if not linked:
                continue
            url = references.get(linked.group(2))
            if not url:
                continue
            parsed = urlsplit(url)
            filename = parsed.path.rsplit("/", 1)[-1]
            if parsed.hostname != _ALLOWED_HOST or parsed.scheme not in {"http", "https"}:
                continue
            if not filename.endswith((".txt.gz", ".h5")):
                continue
            # A table entry must point at the release it names. The upstream
            # 17.02 multilingual row currently points at 17.04 by mistake.
            if filename.endswith(".txt.gz") and version not in filename:
                continue
            artifact = "text-vectors" if filename.endswith(".txt.gz") else "hdf5"
            if artifact == "hdf5":
                scope = "mini-hdf5" if filename.endswith("mini.h5") else "full-hdf5"
            rows.append(
                {
                    "version": version,
                    "scope": scope,
                    "artifact": artifact,
                    "url": url,
                    "source": source,
                }
            )
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in rows:
        key = item["version"], item["scope"], item["artifact"]
        if key in unique and unique[key]["url"] != item["url"]:
            raise ValueError(f"{source}: conflicting URLs for {key}")
        unique[key] = item
    return tuple(unique[key] for key in sorted(unique))
