"""Metadata-only index of MaRS RGB/SAR encoder weights on first-party Zenodo."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

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

_REPOSITORY_URL = "https://github.com/WanderRainy/MaRS"
_RECORD_ID = "17800805"
_RECORD_URL = f"https://zenodo.org/records/{_RECORD_ID}"
_API_URL = f"https://zenodo.org/api/records/{_RECORD_ID}"
_TITLE = (
    "MaRS: A Multi-Modality Very-High-Resolution Remote Sensing Foundation Model with "
    "Cross-Granularity Meta-Modality Learning"
)
_EXPECTED_FILES: dict[str, tuple[str, int, str]] = {
    "mars_base_rgb_encoder_only.pth": (
        "MaRS-Base-RGB-Encoder",
        347_703_439,
        "md5:5ab1b978f783cbded01427c50b0c0dbb",
    ),
    "mars_base_sar_encoder_only.pth": (
        "MaRS-Base-SAR-Encoder",
        347_687_055,
        "md5:8789da9bc9a44853d50d92371a1b4a6c",
    ),
    "mars_large_rgb_encoder_only.pth": (
        "MaRS-Large-RGB-Encoder",
        634_126_336,
        "md5:60574c1b87a7fb9006521fb5c031ee2c",
    ),
    "mars_large_sar_encoder_only.pth": (
        "MaRS-Large-SAR-Encoder",
        639_631_360,
        "md5:6f04c3b8120f0331f22059961b9db6e7",
    ),
}
_DATASET_FILE = "MaRS16M-Demo.zip"


class MarsZenodoCheckpointsSourceAdapter:
    """Enumerate the four source-declared MaRS encoder weights, never file bytes."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only four encoder-only `.pth` files in Zenodo record 17800805. "
        "The record also contains a dataset archive, which is excluded. Checkpoint "
        "links, file sizes, and MD5 values come from Zenodo metadata; no weight bytes are fetched."
    )

    def __init__(
        self,
        *,
        name: str = "mars-zenodo-checkpoints",
        client: Any | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_files: int = 20,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        if isinstance(max_files, bool) or not isinstance(max_files, int) or max_files < 1:
            raise ValueError("max_files must be a positive integer")
        self.name = name.strip()
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.max_files = max_files
        self.checkpoint_signature = content_hash(
            {
                "adapter": "mars-zenodo-checkpoints-v1",
                "record_id": _RECORD_ID,
                "files": sorted(_EXPECTED_FILES),
                "max_files": max_files,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_API_URL, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo record returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Zenodo metadata exceeds configured byte limit")
        digest = content_hash(response.body)
        if digest == state.get("record_sha256"):
            return SourcePage((), dict(state), True, upstream_count=state.get("model_count"))
        files = _parse_record(response.json(), self.name, self.max_files)
        records = tuple(
            self._record(filename, size, checksum, digest)
            for filename, size, checksum in files
        )
        return SourcePage(
            records,
            {"record_sha256": digest, "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, size: int, checksum: str, digest: str) -> SourceRecord:
        model_id, _, _ = _EXPECTED_FILES[filename]
        url = f"https://zenodo.org/records/{_RECORD_ID}/files/{filename}?download=1"
        model_identifier = Identifier("mars:model", model_id)
        file_identifier = Identifier("mars:zenodo-file", f"{_RECORD_ID}/{filename}")
        release = ReleaseHint(
            local_id=f"{model_id}@zenodo-{_RECORD_ID}",
            model_local_id=model_id,
            identifiers=(file_identifier,),
            metadata={
                "zenodo_record_id": _RECORD_ID,
                "filename": filename,
                "size_bytes": size,
                "checksum": checksum,
                "record_metadata_sha256": digest,
            },
            locator=filename,
        )
        return SourceRecord(
            source_record_id=f"{self.name}:{filename}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=model_id,
            raw={
                "zenodo_record_id": _RECORD_ID,
                "filename": filename,
                "size_bytes": size,
                "checksum": checksum,
                "record_metadata_sha256": digest,
            },
            identifiers=(file_identifier, model_identifier),
            links=(
                Link(
                    url,
                    "weights",
                    locator="Zenodo record file metadata",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(_RECORD_URL, "source_record", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_documentation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(
                ModelHint(
                    local_id=model_id,
                    name=model_id.replace("-", " "),
                    identifiers=(model_identifier,),
                    status=ModelStatus.RELEASED,
                ),
            ),
            releases=(release,),
        )


def _parse_record(payload: Any, source: str, maximum: int) -> tuple[tuple[str, int, str], ...]:
    if not isinstance(payload, Mapping) or str(payload.get("id", "")) != _RECORD_ID:
        raise ValueError(f"{source}: response is not Zenodo record {_RECORD_ID}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{source}: Zenodo record metadata is missing")
    if metadata.get("doi") != f"10.5281/zenodo.{_RECORD_ID}" or metadata.get("title") != _TITLE:
        raise ValueError(f"{source}: Zenodo record identity mismatch")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or len(raw_files) > maximum:
        raise ValueError(f"{source}: Zenodo file list is missing or over limit")
    found: dict[str, tuple[int, str]] = {}
    dataset_found = False
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise ValueError(f"{source}: Zenodo file entry is not an object")
        key = item.get("key")
        if key == _DATASET_FILE:
            if dataset_found:
                raise ValueError(f"{source}: duplicate dataset file entry")
            dataset_found = True
            continue
        if key not in _EXPECTED_FILES or key in found:
            raise ValueError(f"{source}: unexpected or duplicate file entry {key!r}")
        _, expected_size, expected_checksum = _EXPECTED_FILES[key]
        size = item.get("size")
        checksum = item.get("checksum")
        links = item.get("links")
        file_url = links.get("self") if isinstance(links, Mapping) else None
        expected_url = f"https://zenodo.org/api/records/{_RECORD_ID}/files/{key}/content"
        if isinstance(size, bool) or size != expected_size:
            raise ValueError(f"{source}: size mismatch for {key}")
        if checksum != expected_checksum:
            raise ValueError(f"{source}: checksum mismatch for {key}")
        if file_url != expected_url:
            raise ValueError(f"{source}: unexpected metadata URL for {key}")
        parsed = urlsplit(file_url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != "zenodo.org"
            or unquote(parsed.path)
            != expected_url.removeprefix("https://zenodo.org")
        ):
            raise ValueError(f"{source}: invalid metadata URL for {key}")
        found[key] = (size, checksum)
    if set(found) != set(_EXPECTED_FILES) or not dataset_found:
        raise ValueError(f"{source}: expected four checkpoints and the declared dataset archive")
    return tuple((filename, *found[filename]) for filename in _EXPECTED_FILES)


__all__ = ["MarsZenodoCheckpointsSourceAdapter"]
