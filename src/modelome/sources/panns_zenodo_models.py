"""Metadata-only inventory for released PANNs AudioSet checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import unquote, urlsplit

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

_TITLE = (
    "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern "
    "Recognition (Pretrained Models)"
)
_REPOSITORY_URL = "https://github.com/qiuqiangkong/audioset_tagging_cnn"
_VERSIONS = (("3576403", "v1"), ("3987831", "v3"))


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PannsZenodoModelsAdapter:
    """Enumerate PANNs .pth files from its two published Zenodo versions.

    The records are the paper's official model releases. The adapter reads JSON
    metadata and file checksums only; it never requests checkpoint bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers .pth checkpoint attachments in PANNs Zenodo versions v1 (record "
        "3576403) and v3 (record 3987831). It excludes paper_statistics.zip and "
        "models distributed through unrelated repositories."
    )

    def __init__(
        self,
        *,
        name: str = "panns-zenodo-models",
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
        max_files_per_record: int = 100,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        for label, value in (
            ("max_files_per_record", max_files_per_record),
            ("max_response_bytes", max_response_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.max_files_per_record = max_files_per_record
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {
                "adapter": "panns-zenodo-model-files-v1",
                "records": _VERSIONS,
                "max_files_per_record": max_files_per_record,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        entries: list[tuple[str, str, Mapping[str, Any], str]] = []
        record_digests: dict[str, str] = {}
        record_file_counts: dict[str, int] = {}
        for record_id, version in _VERSIONS:
            api_url = f"https://zenodo.org/api/records/{record_id}"
            response: HttpResponse = self.client.get(
                api_url, headers={"Accept": "application/json"}
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: Zenodo record {record_id} returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: Zenodo record {record_id} exceeds response limit")
            digest = content_hash(response.body)
            record_digests[record_id] = digest
            payload = response.json()
            files = self._files(payload, record_id)
            record_file_counts[record_id] = len(files)
            entries.extend((record_id, version, item, digest) for item in files)

        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        digest = content_hash(record_digests)
        model_count = len(entries)
        if digest == state.get("records_sha256"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=state.get("model_count"),
            )
        records = tuple(self._record(*entry) for entry in entries)
        return SourcePage(
            records=records,
            next_state={
                "records_sha256": digest,
                "record_digests": record_digests,
                "record_file_counts": record_file_counts,
                "checked_at": checked_at,
                "model_count": model_count,
            },
            complete=True,
            upstream_count=model_count,
            authoritative_snapshot=True,
        )

    def _files(self, payload: Any, record_id: str) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(payload, Mapping) or str(payload.get("id", "")) != record_id:
            raise ValueError(f"{self.name}: response is not Zenodo record {record_id}")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{self.name}: Zenodo record {record_id} has no metadata")
        expected_doi = f"10.5281/zenodo.{record_id}"
        if metadata.get("doi") != expected_doi or metadata.get("title") != _TITLE:
            raise ValueError(f"{self.name}: Zenodo record {record_id} identity mismatch")
        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            raise ValueError(f"{self.name}: Zenodo record {record_id} has no file list")
        if len(raw_files) > self.max_files_per_record:
            raise ValueError(f"{self.name}: Zenodo record {record_id} exceeds file-count limit")
        result: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, Mapping):
                raise ValueError(f"{self.name}: Zenodo file entry is not an object")
            filename = item.get("key")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"{self.name}: Zenodo file has no name")
            if not filename.endswith(".pth"):
                continue
            if filename in seen:
                raise ValueError(f"{self.name}: duplicate checkpoint filename {filename}")
            seen.add(filename)
            links = item.get("links")
            file_url = links.get("self") if isinstance(links, Mapping) else None
            _validate_file_url(file_url, record_id, filename, self.name)
            checksum = item.get("checksum")
            size = item.get("size")
            if not isinstance(checksum, str) or not checksum.startswith("md5:"):
                raise ValueError(f"{self.name}: checkpoint checksum is missing or invalid")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: checkpoint size is invalid")
            result.append(item)
        if not result:
            raise ValueError(f"{self.name}: Zenodo record {record_id} contains no .pth files")
        return tuple(result)

    def _record(
        self,
        record_id: str,
        version: str,
        item: Mapping[str, Any],
        record_digest: str,
    ) -> SourceRecord:
        filename = str(item["key"])
        file_url = str(item["links"]["self"])
        slug = filename.removesuffix(".pth")
        model_id = f"model:{slug}"
        identifier = Identifier("panns:model", slug)
        model = ModelHint(
            local_id=model_id,
            name=slug,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=filename,
        )
        version_doi = f"10.5281/zenodo.{record_id}"
        release = ReleaseHint(
            local_id=f"release:zenodo:{record_id}:{slug}",
            model_local_id=model_id,
            version=version,
            identifiers=(
                Identifier("doi", version_doi),
                Identifier("panns:zenodo-file", f"{record_id}/{filename}"),
            ),
            metadata={
                "zenodo_record_id": record_id,
                "zenodo_record_doi": version_doi,
                "version": version,
                "filename": filename,
                "weight_url": file_url,
                "checksum": item["checksum"],
                "size_bytes": item["size"],
                "record_metadata_sha256": record_digest,
            },
            locator=filename,
        )
        record_url = f"https://zenodo.org/records/{record_id}"
        return SourceRecord(
            source_record_id=f"zenodo:{record_id}:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(file_url),
            title=slug,
            raw={
                "record_id": record_id,
                "version": version,
                "filename": filename,
                "weight_url": file_url,
                "checksum": item["checksum"],
                "size_bytes": item["size"],
            },
            text=f"PANNs AudioSet pretrained checkpoint {filename} (Zenodo {version}).",
            identifiers=(identifier,),
            links=(
                Link(record_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(file_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_file_url(url: Any, record_id: str, filename: str, source: str) -> None:
    if not isinstance(url, str):
        raise ValueError(f"{source}: checkpoint content URL is missing")
    parsed = urlsplit(url)
    path = unquote(parsed.path)
    expected_api_path = f"/api/records/{record_id}/files/{filename}/content"
    expected_page_path = f"/records/{record_id}/files/{filename}"
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or path not in {expected_api_path, expected_page_path}
        or parsed.query not in {"", "download=1"}
        or parsed.fragment
    ):
        raise ValueError(f"{source}: invalid Zenodo checkpoint URL")


__all__ = ["PannsZenodoModelsAdapter"]
