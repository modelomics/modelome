"""Pinned discovery of exact nnU-Net Zenodo bundle identities."""

from __future__ import annotations

import re
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

_RECORDS = {
    "4003545": {
        "doi": "10.5281/zenodo.4003545",
        "generation": "v1",
        "filename": re.compile(r"^(?P<model>Task[0-9]{3,}_[A-Za-z0-9][A-Za-z0-9_.-]*)\.zip$"),
    },
    "8362371": {
        "doi": "10.5281/zenodo.8362371",
        "generation": "v2",
        "filename": re.compile(
            r"^nnUNetTrainer__nnUNetPlans__(?P<model>3d_fullres_resenc(?:_192x192x192_b24|_bs80))_exported\.zip$"
        ),
    },
}
_MD5 = re.compile(r"^[0-9a-f]{32}$")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NnUNetZenodoBundleRegistryAdapter:
    """Enumerate only author-declared files from two fixed Zenodo records.

    Record 4003545 is the v1 dataset bundle inventory, a later version of the
    record already covered by the older adapter. Record 8362371 is the two-model
    AutoPET II nnU-Net v2 release linked by the first-party competition guide.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers file metadata in fixed Zenodo records 4003545 (nnU-Net v1) and "
        "8362371 (two AutoPET II nnU-Net v2 configurations). It does not discover "
        "a general v2 model zoo, inspect archives, or download checkpoint bytes."
    )

    def __init__(
        self,
        *,
        record_id: str,
        name: str = "nnunet-zenodo-bundle-registry",
        max_record_bytes: int = 4 * 1024 * 1024,
        max_files: int = 500,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if record_id not in _RECORDS:
            raise ValueError("only the verified nnU-Net Zenodo records are supported")
        self.name = name
        self.record_id = record_id
        self.spec = _RECORDS[record_id]
        self.record_url = f"https://zenodo.org/records/{record_id}"
        self.api_url = f"https://zenodo.org/api/records/{record_id}"
        self.max_record_bytes = _positive_int(max_record_bytes, "max_record_bytes")
        self.max_files = _positive_int(max_files, "max_files")
        self.client = client or HttpClient(max_response_bytes=self.max_record_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "nnunet-fixed-zenodo-bundles-v1",
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
                records=(), next_state={**state, "checked_at": checked_at},
                complete=True, upstream_count=_nonnegative_int(state.get("model_count")),
            )
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo record returned HTTP {response.status}")
        if len(response.body) > self.max_record_bytes:
            raise ValueError(f"{self.name}: Zenodo record exceeds {self.max_record_bytes} bytes")
        metadata, bundles = self._parse_record(response.json())
        record_hash = content_hash(response.body)
        if record_hash == _text(state.get("completed_content_hash")):
            return SourcePage(
                records=(),
                next_state={**state, "checked_at": checked_at,
                            "completed_content_hash": record_hash},
                complete=True, upstream_count=len(bundles),
            )

        models = []
        releases = []
        links = []
        for bundle in bundles:
            local_id = f"nnunet:{self.spec['generation']}:{bundle['model']}#model"
            model_identifier = Identifier(
                "nnunet:model"
                if self.spec["generation"] == "v1"
                else "nnunet:v2:model",
                bundle["model"],
            )
            models.append(
                ModelHint(
                    local_id=local_id, name=bundle["model"],
                    identifiers=(model_identifier,), status=ModelStatus.RELEASED,
                    locator=f"files.{bundle['filename']}",
                )
            )
            releases.append(
                ReleaseHint(
                    local_id=f"{local_id}:release:{bundle['checksum']}",
                    model_local_id=local_id,
                    version=metadata["version"],
                    revision=bundle["checksum"],
                    identifiers=(Identifier(
                        "nnunet:task-archive",
                        f"{metadata['doi']}#{bundle['filename']}@{bundle['checksum']}",
                    ),),
                    metadata={
                        "archive_url": bundle["url"],
                        "archive_checksum": bundle["checksum"],
                        "archive_checksum_type": "md5",
                        "archive_size": bundle["size"],
                        "release_doi": metadata["doi"],
                        "nnunet_generation": self.spec["generation"],
                    },
                    locator=f"files.{bundle['filename']}",
                )
            )
            links.append(Link(
                bundle["url"], relation="weights", locator=f"files.{bundle['filename']}",
                crawl=False, model_local_ids=(local_id,),
            ))
        next_state = {
            "checked_at": checked_at,
            "completed_content_hash": record_hash,
            "model_count": len(bundles),
            "release_doi": metadata["doi"],
        }
        if etag := _header(response.headers, "etag"):
            next_state["etag"] = etag
        record = SourceRecord(
            source_record_id=f"{self.name}:zenodo-{self.record_id}",
            kind=ArtifactKind.CATALOG_RECORD, canonical_url=self.record_url,
            title=metadata["title"],
            raw={"record_id": self.record_id, "record_url": self.record_url,
                 "api_url": self.api_url, "doi": metadata["doi"],
                 "version": metadata["version"], "record_hash": record_hash,
                 "response_url": response.url or self.api_url, "files": bundles},
            text="\n".join(bundle["model"] for bundle in bundles),
            identifiers=(Identifier("doi", metadata["doi"]),),
            links=tuple(links), models=tuple(models), releases=tuple(releases),
        )
        return SourcePage(
            records=(record,), next_state=next_state, complete=True,
            upstream_count=len(bundles), authoritative_snapshot=True,
        )

    def _parse_record(self, payload: Any) -> tuple[dict[str, str], list[dict[str, Any]]]:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("metadata"), Mapping):
            raise ValueError(f"{self.name}: Zenodo response is missing metadata")
        metadata = payload["metadata"]
        doi = _text(metadata.get("doi"))
        if doi.casefold() != self.spec["doi"].casefold():
            raise ValueError(f"{self.name}: response DOI does not match the supported record")
        title = _required_text(metadata.get("title"), "Zenodo record title")
        version = _text(metadata.get("version")) or "1"
        raw_files = payload.get("files")
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes, bytearray)):
            raise ValueError(f"{self.name}: Zenodo files must be an array")
        if len(raw_files) > self.max_files:
            raise ValueError(f"{self.name}: Zenodo record exceeds {self.max_files} files")
        bundles = []
        seen = set()
        for raw in raw_files:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{self.name}: Zenodo file entry is not an object")
            filename = _required_text(raw.get("key") or raw.get("filename"), "file key")
            match = self.spec["filename"].fullmatch(filename)
            if match is None:
                continue
            if filename in seen:
                raise ValueError(f"{self.name}: duplicate model archive {filename!r}")
            seen.add(filename)
            checksum = _text(raw.get("checksum")).casefold()
            if checksum.startswith("md5:"):
                checksum = checksum[4:]
            if not _MD5.fullmatch(checksum):
                raise ValueError(f"{self.name}: invalid MD5 for {filename!r}")
            raw_links = raw.get("links")
            if isinstance(raw_links, Mapping) and _text(raw_links.get("self")):
                file_url = self._download_url(
                    _text(raw_links.get("self")), filename, content_endpoint=True
                )
            else:
                file_url = (
                    _text(raw_links.get("download"))
                    if isinstance(raw_links, Mapping) else ""
                )
                file_url = self._download_url(file_url, filename)
            size = raw.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(f"{self.name}: invalid size for {filename!r}")
            bundles.append({"filename": filename, "model": match.group("model"),
                            "checksum": checksum, "size": size, "url": file_url})
        if not bundles:
            raise ValueError(f"{self.name}: Zenodo record contains no recognised model bundles")
        return {"doi": doi, "title": title, "version": version}, sorted(
            bundles, key=lambda item: item["filename"]
        )

    def _download_url(
        self, value: str, filename: str, *, content_endpoint: bool = False
    ) -> str:
        url = canonicalize_url(value)
        parts = urlsplit(url)
        path_parts = parts.path.split("/")
        if (
            parts.scheme != "https" or parts.hostname != "zenodo.org"
            or self.record_id not in path_parts
            or filename not in path_parts
            or parts.query not in {"", "download=1"} or parts.fragment
            or (
                content_endpoint
                and path_parts
                != ["", "api", "records", self.record_id, "files", filename, "content"]
            )
        ):
            raise ValueError(f"{self.name}: download URL for {filename!r} is not in its record")
        return url


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _required_text(value: Any, label: str) -> str:
    result = _text(value)
    if not result:
        raise ValueError(f"{label} must be a non-empty string")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _header(headers: Mapping[str, Any], name: str) -> str:
    return next((_text(value) for key, value in headers.items()
                 if str(key).casefold() == name.casefold() and _text(value)), "")


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
