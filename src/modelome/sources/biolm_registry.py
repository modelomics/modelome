"""Meta FAIR's first-party biomedical language model archive table."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]
_REPOSITORY = "facebookresearch/bio-lm"
_README_PATH = "README.md"
_MAX_ENTRIES = 100
_ARCHIVE_LINK = re.compile(r"\[download\]\((https://[^)]+)\)", re.IGNORECASE)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class BioLMRegistrySourceAdapter:
    """Enumerate exact FAIR biomedical LM names and both published archives."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the model rows and exact Transformers/fairseq tar.gz links in "
        "facebookresearch/bio-lm README. It does not inspect archive members or "
        "cover downstream fine-tuned models."
    )

    def __init__(
        self,
        *,
        name: str = "facebook-bio-lm-first-party-archives",
        repository: str = _REPOSITORY,
        branch: str = "main",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_entries: int = _MAX_ENTRIES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if not name.strip() or not branch.strip() or max_response_bytes <= 0 or max_entries <= 0:
            raise ValueError("name, branch, and positive limits are required")
        self.name = name
        self.repository = repository
        self.branch = branch
        self.max_response_bytes = max_response_bytes
        self.max_entries = max_entries
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "facebook-bio-lm-first-party-archive-table-v1",
            "repository": repository,
            "branch": branch,
            "readme_path": _README_PATH,
            "max_response_bytes": max_response_bytes,
            "max_entries": max_entries,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        readme_url = (
            f"https://raw.githubusercontent.com/{self.repository}/"
            f"{self.branch}/{_README_PATH}"
        )
        response = self.client.get(readme_url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        digest = hashlib.sha256(response.body).hexdigest()
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if digest == state.get("completed_revision"):
            return SourcePage(
                (), {**state, "checked_at": checked}, True,
                upstream_count=state.get("checkpoint_count"),
            )
        models = _parse_model_table(response.text(), self.max_entries)
        records = tuple(_record(model, readme_url, self.name) for model in models)
        return SourcePage(
            records,
            {"completed_revision": digest, "checked_at": checked,
             "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_model_table(readme: str, limit: int) -> list[dict[str, str]]:
    in_models = False
    saw_models_section = False
    parsed: dict[str, dict[str, str]] = {}
    seen_urls: set[str] = set()
    for line in readme.splitlines():
        if line.strip().casefold() == "## models":
            in_models = True
            saw_models_section = True
            continue
        if in_models and line.startswith("## "):
            break
        if not in_models or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0].casefold() == "model" or set(cells[0]) <= {"-", ":"}:
            continue
        model_name, model_size, description = cells[:3]
        urls = _ARCHIVE_LINK.findall(line)
        if len(urls) != 2:
            continue
        transformers_url, fairseq_url = urls
        if model_name in parsed:
            raise ValueError(f"Bio-LM README repeats model {model_name}")
        _validate_archive(transformers_url, "-hf.tar.gz")
        _validate_archive(fairseq_url, "-fairseq.tar.gz")
        if transformers_url in seen_urls or fairseq_url in seen_urls:
            raise ValueError(f"Bio-LM README repeats archive URL for {model_name}")
        expected_stem = model_name
        transformer_filename = urlsplit(transformers_url).path.rsplit("/", 1)[-1]
        fairseq_filename = urlsplit(fairseq_url).path.rsplit("/", 1)[-1]
        if (
            transformer_filename != f"{expected_stem}-hf.tar.gz"
            or fairseq_filename != f"{expected_stem}-fairseq.tar.gz"
        ):
            raise ValueError(f"Bio-LM archive filenames do not match model {model_name}")
        parsed[model_name] = {
            "name": model_name,
            "size": model_size,
            "description": description,
            "transformers_url": transformers_url,
            "fairseq_url": fairseq_url,
        }
        seen_urls.update((transformers_url, fairseq_url))
        if len(parsed) > limit:
            raise ValueError("Bio-LM model table exceeds entry limit")
    if not saw_models_section or not parsed:
        raise ValueError("Bio-LM README Models section has no admitted model rows")
    return [parsed[name] for name in sorted(parsed)]


def _validate_archive(url: str, suffix: str) -> None:
    parts = urlsplit(url)
    if (
        parts.scheme != "https" or parts.netloc != "dl.fbaipublicfiles.com"
        or not parts.path.startswith("/biolm/") or not parts.path.endswith(suffix)
        or parts.query or parts.fragment
    ):
        raise ValueError(f"Bio-LM has a non-admitted archive URL: {url}")


def _record(model: dict[str, str], readme_url: str, source_name: str) -> SourceRecord:
    name = model["name"]
    local_id = f"model:{name}"
    model_hint = ModelHint(
        local_id, name,
        identifiers=(Identifier("facebook-bio-lm:model", name),),
        aliases=(name,), status=ModelStatus.RELEASED,
    )
    release_hints: list[ReleaseHint] = []
    links: list[Link] = [
        Link(readme_url, "model_card", crawl=False, model_local_ids=(local_id,)),
    ]
    for format_name, url in (
        ("transformers", model["transformers_url"]),
        ("fairseq", model["fairseq_url"]),
    ):
        release_hints.append(ReleaseHint(
            f"archive:{format_name}", local_id,
            identifiers=(Identifier("facebook-bio-lm:archive-url", url),),
            metadata={
                "format": format_name,
                "archive_format": "tar.gz",
                "archive_url": url,
                "declared_model_size": model["size"],
                "description": model["description"],
            },
            locator=f"Bio-LM README model table: {name} ({format_name})",
        ))
        links.append(Link(url, "weights", crawl=False, model_local_ids=(local_id,)))
    canonical_url = model["transformers_url"]
    return SourceRecord(
        source_record_id=f"{source_name}:model:{name}",
        kind=ArtifactKind.CATALOG_RECORD,
        canonical_url=canonicalize_url(canonical_url),
        title=f"Biomedical language model {name}",
        raw={"model": model, "readme_url": readme_url},
        text=f"{name}: {model['description']}",
        identifiers=(Identifier("facebook-bio-lm:model", name),),
        links=tuple(links),
        models=(model_hint,),
        releases=tuple(release_hints),
    )


__all__ = ["BioLMRegistrySourceAdapter"]
