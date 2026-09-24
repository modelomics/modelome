"""Atari-PB pretrained model files from the authors' Zenodo record."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient, HttpResponse
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

_RECORD_ID = "16981616"
_DOI = "10.5281/zenodo.16981616"
_RECORD_URL = f"https://zenodo.org/records/{_RECORD_ID}"
_API_URL = f"https://zenodo.org/api/records/{_RECORD_ID}"
_REPOSITORY_URL = "https://github.com/dojeon-ai/Atari-PB"
_ALGORITHMS = frozenset(
    {
        "atc",
        "bc",
        "cql_dist",
        "cql_mse",
        "curl",
        "dt",
        "idm",
        "mae",
        "r3m",
        "siammae",
        "spr",
        "spr_idm",
    }
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AtariPbCheckpointAdapter:
    """Enumerate exact `.pth` files in Atari-PB's fixed public Zenodo record.

    The Atari-PB repository links to this record as its main-experiment model
    weights. The record API supplies each file's own download URL and checksum.
    The separate `Far-OOD.zip` dataset is intentionally excluded.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the twelve algorithm checkpoint files in Atari-PB Zenodo "
        "record 16981616. It excludes dataset archives, other experiment outputs, "
        "and checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "atari-pb-checkpoints",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_files: int = 100,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        for label, value in (
            ("max_response_bytes", max_response_bytes),
            ("max_files", max_files),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.max_files = max_files
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "atari-pb-zenodo-checkpoints-v1",
                "record_id": _RECORD_ID,
                "algorithms": sorted(_ALGORITHMS),
                "max_response_bytes": max_response_bytes,
                "max_files": max_files,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(
            _API_URL, headers={"Accept": "application/json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo record returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Zenodo record exceeds response limit")
        payload = response.json()
        files = self._files(payload)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(item) for item in files)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "record_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _files(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: Zenodo response is not an object")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("doi") != _DOI:
            raise ValueError(f"{self.name}: response DOI does not match Atari-PB record")
        raw_files = payload.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes)):
            raise ValueError(f"{self.name}: Zenodo files must be an array")
        if len(raw_files) > self.max_files:
            raise ValueError(f"{self.name}: Zenodo record exceeds file limit")
        found: dict[str, dict[str, Any]] = {}
        for raw in raw_files:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.name}: Zenodo file entry is not an object")
            filename = raw.get("key")
            if not isinstance(filename, str) or not filename.endswith(".pth"):
                continue
            algorithm = filename.removesuffix(".pth")
            if algorithm not in _ALGORITHMS:
                raise ValueError(f"{self.name}: unexpected checkpoint filename {filename!r}")
            if algorithm in found:
                raise ValueError(f"{self.name}: duplicate checkpoint {filename!r}")
            links = raw.get("links")
            download_url = links.get("download") if isinstance(links, Mapping) else None
            if not isinstance(download_url, str):
                raise ValueError(f"{self.name}: download URL missing for {filename!r}")
            _validate_download_url(download_url, filename, self.name)
            checksum = raw.get("checksum")
            size = raw.get("size")
            if not isinstance(checksum, str) or not checksum:
                raise ValueError(f"{self.name}: checksum missing for {filename!r}")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: invalid size for {filename!r}")
            found[algorithm] = {
                "algorithm": algorithm,
                "filename": filename,
                "url": canonicalize_url(download_url),
                "checksum": checksum,
                "size": size,
            }
        if set(found) != _ALGORITHMS:
            missing = sorted(_ALGORITHMS - set(found))
            raise ValueError(f"{self.name}: expected checkpoint inventory is incomplete: {missing}")
        return [found[key] for key in sorted(found)]

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        algorithm = str(item["algorithm"])
        filename = str(item["filename"])
        local_id = f"checkpoint:atari-pb:{algorithm}"
        identifier = Identifier("atari-pb:checkpoint", algorithm)
        model = ModelHint(
            local_id=local_id,
            name=f"Atari-PB {algorithm} pretrained model",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        release = ReleaseHint(
            local_id=f"release:atari-pb:{algorithm}",
            model_local_id=local_id,
            version="1",
            identifiers=(Identifier("atari-pb:zenodo-file", filename),),
            metadata={
                "algorithm": algorithm,
                "checkpoint_filename": filename,
                "checksum": item["checksum"],
                "size_bytes": item["size"],
                "weight_url": item["url"],
                "record_doi": _DOI,
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(str(item["url"])),
            title=f"Atari-PB {algorithm} checkpoint",
            raw={"algorithm": algorithm, **dict(item)},
            text=f"Official Atari-PB pretrained model checkpoint {filename}.",
            identifiers=(identifier,),
            links=(
                Link(_RECORD_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(str(item["url"]), "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_download_url(url: str, filename: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query not in {"", "download=1"}
        or parsed.fragment
        or _RECORD_ID not in parsed.path.split("/")
        or filename not in parsed.path.split("/")
    ):
        raise ValueError(f"{source}: invalid record-scoped download URL for {filename!r}")


__all__ = ["AtariPbCheckpointAdapter"]
