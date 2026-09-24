"""nnU-Net v1 pretrained bundles from the authors' fixed Zenodo release."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
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

Clock = Callable[[], datetime]
_RECORD_ID = "3734294"
_TASK_FILE = re.compile(r"^(?P<task>Task[0-9]{3,}_[A-Za-z0-9][A-Za-z0-9_.-]*)\.zip$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_DOI = re.compile(r"^10\.5281/zenodo\.3734294$", re.IGNORECASE)
_MAX_FILES = 500


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NnUNetV1PretrainedRegistryAdapter:
    """Enumerate task bundles in the nnU-Net v1 authors' Zenodo release.

    nnU-Net's maintained documentation says v2 has no pretrained-model catalog
    and directs v1 users to legacy weights. This adapter reads only the authors'
    fixed Zenodo record 3734294, which contains task-named zip bundles with
    record-specific download URLs and checksums. It does not discover unrelated
    nnU-Net forks or download archive bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the nnU-Net v1 pretrained task bundles in Zenodo record 3734294. "
        "It does not cover nnU-Net v2, later third-party nnU-Net bundles, or "
        "BioImage.IO/Hugging Face mirrors. Archive contents are not downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "nnunet-v1-pretrained-models",
        record_id: str = _RECORD_ID,
        max_record_bytes: int = 4 * 1024 * 1024,
        max_files: int = _MAX_FILES,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        if record_id != _RECORD_ID:
            raise ValueError(
                f"{self.name}: only official nnU-Net v1 record {_RECORD_ID} is supported"
            )
        self.record_id = record_id
        self.record_url = f"https://zenodo.org/records/{record_id}"
        self.api_url = f"https://zenodo.org/api/records/{record_id}"
        self.max_record_bytes = _positive_int(max_record_bytes, "max_record_bytes", self.name)
        self.max_files = _positive_int(max_files, "max_files", self.name)
        self.client = client or HttpClient(max_response_bytes=self.max_record_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "nnunet-v1-zenodo-release-v1",
                "record_id": record_id,
                "max_record_bytes": self.max_record_bytes,
                "max_files": self.max_files,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        headers = {"Accept": "application/json"}
        if etag := _text(state.get("etag")):
            headers["If-None-Match"] = etag
        response: HttpResponse = self.client.get(self.api_url, headers=headers)
        checked_at = _isoformat(self.clock())
        if response.status == 304:
            if not _text(state.get("completed_content_hash")):
                raise ValueError(f"{self.name}: Zenodo returned 304 without a prior snapshot")
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at},
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo record returned HTTP {response.status}")
        if len(response.body) > self.max_record_bytes:
            raise ValueError(
                f"{self.name}: Zenodo record exceeds {self.max_record_bytes} bytes"
            )
        payload = response.json()
        metadata, files = self._record(payload)
        record_hash = content_hash(response.body)
        if record_hash == _text(state.get("completed_content_hash")):
            return SourcePage(
                records=(),
                next_state={
                    **state,
                    "checked_at": checked_at,
                    "completed_content_hash": record_hash,
                },
                complete=True,
                upstream_count=len(files),
            )

        models = []
        releases = []
        links = []
        records = []
        for item in files:
            task = item["task"]
            local_id = f"nnunet:{task}#model"
            models.append(
                ModelHint(
                    local_id=local_id,
                    name=task,
                    identifiers=(Identifier("nnunet:model", task),),
                    status=ModelStatus.RELEASED,
                    locator=f"files.{item['filename']}",
                )
            )
            releases.append(
                ReleaseHint(
                    local_id=f"{local_id}:release:{item['checksum']}",
                    model_local_id=local_id,
                    version=metadata["version"],
                    revision=item["checksum"],
                    identifiers=(
                        Identifier(
                            "nnunet:task-archive",
                            f"{metadata['doi']}#{item['filename']}@{item['checksum']}",
                        ),
                    ),
                    metadata={
                        "archive_url": item["url"],
                        "archive_checksum": item["checksum"],
                        "archive_checksum_type": "md5",
                        "archive_size": item["size"],
                        "release_doi": metadata["doi"],
                        "nnunet_generation": "v1",
                    },
                    locator=f"files.{item['filename']}",
                )
            )
            links.append(
                Link(
                    item["url"],
                    relation="weights",
                    locator=f"files.{item['filename']}",
                    crawl=False,
                    model_local_ids=(local_id,),
                )
            )
            records.append(item)
        etag = _header(response.headers, "etag") or None
        next_state = {
            "checked_at": checked_at,
            "completed_content_hash": record_hash,
            "model_count": len(files),
            "release_doi": metadata["doi"],
        }
        if etag:
            next_state["etag"] = etag
        record = SourceRecord(
            source_record_id=f"{self.name}:zenodo-{self.record_id}",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self.record_url,
            title=metadata["title"],
            raw={
                "record_id": self.record_id,
                "record_url": self.record_url,
                "api_url": self.api_url,
                "doi": metadata["doi"],
                "version": metadata["version"],
                "record_hash": record_hash,
                "response_url": response.url or self.api_url,
                "files": records,
            },
            text="\n".join(item["task"] for item in files),
            identifiers=(Identifier("doi", metadata["doi"]),),
            links=tuple(links),
            models=tuple(models),
            releases=tuple(releases),
        )
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=True,
            upstream_count=len(files),
            authoritative_snapshot=True,
        )

    def _record(self, payload: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: Zenodo response is not an object")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{self.name}: Zenodo response is missing metadata")
        doi = _text(metadata.get("doi"))
        if not _DOI.fullmatch(doi):
            raise ValueError(f"{self.name}: response DOI does not match the supported record")
        title = _required_text(metadata.get("title"), "Zenodo record title")
        version = _text(metadata.get("version")) or "1"
        raw_files = payload.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: Zenodo files must be an array")
        if len(raw_files) > self.max_files:
            raise ValueError(f"{self.name}: Zenodo record exceeds {self.max_files} files")
        files = []
        seen = set()
        for raw in raw_files:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.name}: Zenodo file entry is not an object")
            filename = _required_text(raw.get("key") or raw.get("filename"), "file key")
            match = _TASK_FILE.fullmatch(filename)
            if match is None:
                continue
            if filename in seen:
                raise ValueError(f"{self.name}: duplicate task archive {filename!r}")
            seen.add(filename)
            checksum = _text(raw.get("checksum")).casefold()
            if checksum.startswith("md5:"):
                checksum = checksum[4:]
            if not _MD5.fullmatch(checksum):
                raise ValueError(f"{self.name}: invalid MD5 for {filename!r}")
            links = raw.get("links")
            if isinstance(links, Mapping) and _text(links.get("self")):
                download_url = self._download_url(
                    _text(links.get("self")), filename, content_endpoint=True
                )
            else:
                download_url = (
                    _text(links.get("download"))
                    if isinstance(links, Mapping)
                    else _text(raw.get("links"))
                )
                download_url = self._download_url(download_url, filename)
            size = raw.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: invalid size for {filename!r}")
            files.append(
                {
                    "filename": filename,
                    "task": match.group("task"),
                    "checksum": checksum,
                    "size": size,
                    "url": download_url,
                }
            )
        if not files:
            raise ValueError(f"{self.name}: Zenodo record contains no nnU-Net task archives")
        return {"doi": doi, "title": title, "version": version}, sorted(
            files, key=lambda item: item["filename"]
        )

    def _download_url(
        self, value: str, filename: str, *, content_endpoint: bool = False
    ) -> str:
        url = canonicalize_url(value)
        parts = urlsplit(url)
        path_parts = parts.path.split("/")
        if (
            parts.scheme != "https"
            or parts.hostname != "zenodo.org"
            or self.record_id not in path_parts
            or filename not in path_parts
            or parts.query not in {"", "download=1"}
            or parts.fragment
            or (
                content_endpoint
                and path_parts
                != ["", "api", "records", self.record_id, "files", filename, "content"]
            )
        ):
            raise ValueError(f"{self.name}: download URL for {filename!r} is not in its record")
        return url


def _required_text(value: Any, label: str) -> str:
    result = value.strip() if isinstance(value, str) else ""
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _positive_int(value: Any, label: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source}: {label} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _header(headers: Mapping[str, Any], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return _text(value)
    return ""


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = ["NnUNetV1PretrainedRegistryAdapter"]
