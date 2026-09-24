"""TorchMD-Net's documented AceFF checkpoint family from Acellera's collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
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
_COLLECTION_URL = "https://huggingface.co/collections/Acellera/aceff-machine-learning-potentials"
_TORCHMD_README = "https://github.com/torchmd/torchmd-net"
_MODELS = {
    "AceFF-1.0": "aceff_v1.0.ckpt",
    "AceFF-1.1": "aceff_v1.1.ckpt",
    "AceFF-2.0": "aceff_v2.0.ckpt",
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _CollectionModels(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.repositories: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href", "")
        prefix = "/Acellera/"
        if href and href.startswith(prefix):
            repository = href[len(prefix):].split("/", 1)[0]
            if repository.startswith("AceFF-"):
                self.repositories.add(repository)


class TorchMDNetAceFFCheckpointSourceAdapter:
    """Index the three AceFF releases collected by TorchMD-Net's model partner."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the three AceFF model repositories in Acellera's official collection "
        "referenced by TorchMD-Net. The one-file-per-release map is verified against the "
        "individual repository listings and TorchMD-Net loading examples. Model bytes are "
        "not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "torchmdnet-aceff-checkpoints",
        collection_url: str = _COLLECTION_URL,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if collection_url != _COLLECTION_URL or not name.strip() or max_response_bytes <= 0:
            raise ValueError(
                "collection URL is fixed; name and positive response limit are required"
            )
        self.name, self.collection_url = name, collection_url
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torchmdnet-aceff-checkpoints-v1",
                "collection_url": collection_url,
                "torchmd_readme": _TORCHMD_README,
                "models": _MODELS,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(
            self.collection_url, headers={"Accept": "text/html,application/xhtml+xml"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: source returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: source exceeds {self.max_response_bytes} bytes")
        source_hash = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if source_hash == state.get("source_sha256"):
            return SourcePage(
                (), {**state, "checked_at": checked_at}, True,
                upstream_count=state.get("model_count"),
            )
        repositories = _parse_collection(response.text(), self.name)
        records = tuple(
            self._record(version, filename, source_hash)
            for version, filename in _MODELS.items()
        )
        return SourcePage(
            records,
            {
                "checked_at": checked_at,
                "source_url": self.collection_url,
                "source_sha256": source_hash,
                "repository_ids": repositories,
                "model_count": len(records),
            },
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, version: str, filename: str, source_hash: str) -> SourceRecord:
        repository = f"Acellera/{version}"
        model_id = f"model:{version.lower()}"
        namespace = "torchmdnet:aceff-checkpoint"
        file_url = f"https://huggingface.co/{repository}/blob/main/{filename}"
        model_url = f"https://huggingface.co/{repository}"
        model = ModelHint(
            model_id,
            f"{version} neural network potential",
            identifiers=(Identifier(namespace, version),),
            aliases=(filename, repository),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{version.lower()}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", version),),
            metadata={
                "repository": repository,
                "checkpoint_filename": filename,
                "checkpoint_url": file_url,
                "source_collection": self.collection_url,
                "torchmdnet_usage_documentation": _TORCHMD_README,
                "source_sha256": source_hash,
                "binary_reachability_checked": False,
            },
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{version.lower()}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=f"{version} checkpoint",
            raw={
                "model_version": version,
                "repository": repository,
                "checkpoint_filename": filename,
                "checkpoint_url": file_url,
            },
            text=(
                f"TorchMD-Net documents AceFF inference using {repository} and {filename}; "
                "Acellera lists the model in its official AceFF collection."
            ),
            links=(
                Link(file_url, "checkpoint", crawl=False, model_local_ids=(model_id,)),
                Link(model_url, "model_repository", crawl=False, model_local_ids=(model_id,)),
                Link(_TORCHMD_README, "torchmdnet_usage", crawl=False, model_local_ids=(model_id,)),
                Link(
                    self.collection_url,
                    "source_collection",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _parse_collection(document: str, source: str) -> tuple[str, ...]:
    parser = _CollectionModels()
    parser.feed(document)
    if parser.repositories != set(_MODELS):
        raise ValueError(f"{source}: expected the exact three-model AceFF collection")
    for version, filename in _MODELS.items():
        repository = f"https://huggingface.co/Acellera/{version}"
        file_url = f"{repository}/blob/main/{filename}"
        if urlsplit(file_url).hostname != "huggingface.co":
            raise ValueError(f"{source}: invalid checkpoint URL for {version}")
    return tuple(sorted(parser.repositories))


__all__ = ["TorchMDNetAceFFCheckpointSourceAdapter"]
