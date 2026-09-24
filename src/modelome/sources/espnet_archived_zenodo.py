"""Metadata-only index for one archived ESPnet checkpoint removed from table.csv."""

from __future__ import annotations

from collections.abc import Mapping
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

_RECORD_ID = "4065140"
_DOI = "10.5281/zenodo.4065140"
_RECORD_URL = f"https://zenodo.org/records/{_RECORD_ID}"
_API_URL = f"https://zenodo.org/api/records/{_RECORD_ID}"
_MODEL = "kan-bayashi/csj_asr_train_asr_conformer_raw_char_sp_valid.acc.ave"
_FILENAME = "asr_train_asr_conformer_raw_char_sp_valid.acc.ave.zip"
_REPOSITORY_URL = "https://github.com/espnet/espnet"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EspnetArchivedZenodoCheckpointAdapter:
    """Enumerate the named 2020 CSJ ASR release omitted from ESPnet's live CSV.

    ESPnet issue #55 points users to Zenodo record 4065140 after noting that the
    checkpoint was no longer registered in ``table.csv``. The adapter reads
    only Zenodo metadata and never downloads the archive.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the archived CSJ Conformer checkpoint in Zenodo record "
        "4065140, documented by ESPnet issue 55. It does not enumerate other "
        "retired ESPnet models or download the checkpoint archive."
    )

    def __init__(
        self,
        *,
        name: str = "espnet-archived-csj-conformer",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "espnet-archived-zenodo-checkpoint-v1",
                "record_id": _RECORD_ID,
                "filename": _FILENAME,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(_API_URL, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: Zenodo record returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: Zenodo record exceeds response limit")
        payload = response.json()
        item = self._file(payload)
        url = str(item["url"])
        digest = content_hash(response.body)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        model_id = f"model:{_MODEL}"
        identifier = Identifier("espnet:model", _MODEL)
        model = ModelHint(
            local_id=model_id,
            name=_MODEL,
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=_FILENAME,
        )
        release = ReleaseHint(
            local_id=f"release:{_MODEL}",
            model_local_id=model_id,
            version="2020-10-04",
            identifiers=(Identifier("espnet:zenodo-file", _FILENAME),),
            metadata={
                "record_id": _RECORD_ID,
                "record_doi": _DOI,
                "filename": _FILENAME,
                "weight_url": url,
                "checksum": item["checksum"],
                "size_bytes": item["size"],
                "corpus": "csj",
                "task": "asr",
                "sample_rate_hz": 16000,
                "language": "ja",
                "espnet_version": "0.8.0",
                "recipe_commit": "51352aee9ae318640e128a645e722d1f7524edb1",
            },
            locator=_FILENAME,
        )
        record = SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=_MODEL,
            raw={"record_id": _RECORD_ID, "record_sha256": digest, **item},
            text="Archived ESPnet CSJ Japanese ASR Conformer checkpoint.",
            identifiers=(identifier,),
            links=(
                Link(_RECORD_URL, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(
                    _REPOSITORY_URL,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
                Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )
        return SourcePage(
            records=(record,),
            next_state={"checked_at": checked_at, "record_sha256": digest, "model_count": 1},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _file(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: Zenodo response is not an object")
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("doi") != _DOI:
            raise ValueError(f"{self.name}: response DOI does not match ESPnet record")
        title = metadata.get("title")
        expected_title = (
            "ESPnet2 pretrained model, "
            "kan-bayashi/csj_asr_train_asr_conformer_raw_char_sp_valid.acc.ave, "
            "fs=16k, lang=jp"
        )
        if title != expected_title:
            raise ValueError(f"{self.name}: unexpected Zenodo record title")
        files = payload.get("files")
        if not isinstance(files, list):
            raise ValueError(f"{self.name}: Zenodo files must be an array")
        matches = [
            item for item in files if isinstance(item, Mapping) and item.get("key") == _FILENAME
        ]
        if len(matches) != 1:
            raise ValueError(f"{self.name}: expected exactly one ESPnet checkpoint file")
        item = matches[0]
        file_links = item.get("links")
        download_url = None
        if isinstance(file_links, Mapping):
            # Current Zenodo Records API exposes its file-content URL as `self`;
            # older responses may still expose the equivalent as `download`.
            download_url = (
                file_links.get("self") or file_links.get("download") or file_links.get("content")
            )
        _validate_download_url(download_url, self.name)
        checksum = item.get("checksum")
        size = item.get("size")
        if not isinstance(checksum, str) or not checksum:
            raise ValueError(f"{self.name}: checkpoint checksum is missing")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{self.name}: checkpoint size is invalid")
        return {
            "filename": _FILENAME,
            "url": canonicalize_url(download_url),
            "checksum": checksum,
            "size": size,
        }


def _validate_download_url(url: Any, source: str) -> None:
    if not isinstance(url, str):
        raise ValueError(f"{source}: checkpoint download URL is missing")
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    is_record_page_file = parts == ["records", "4065140", "files", _FILENAME]
    is_api_file_content = parts == ["api", "records", "4065140", "files", _FILENAME, "content"]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "zenodo.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query not in {"", "download=1"}
        or parsed.fragment
        or not (is_record_page_file or is_api_file_content)
    ):
        raise ValueError(f"{source}: invalid Zenodo checkpoint URL")


__all__ = ["EspnetArchivedZenodoCheckpointAdapter"]
