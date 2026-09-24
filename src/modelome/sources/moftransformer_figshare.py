"""MOFTransformer's first-party pretrained and fine-tuned Figshare checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
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

_ARTICLE_ID = 21155506
_ARTICLE_URL = "https://figshare.com/articles/dataset/MOFTransformer/21155506"
_API_URL = f"https://api.figshare.com/v2/articles/{_ARTICLE_ID}"
_FILE_URL_PREFIX = "https://ndownloader.figshare.com/files/"
_CHECKPOINTS = frozenset(
    {"best_mtp_moc_vfp.ckpt", "finetuned_bandgap.ckpt", "finetuned_h2_uptake.ckpt"}
)


class MOFTransformerFigshareAdapter:
    """Index exact checkpoint file objects from the authors' public Figshare article."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the three `.ckpt` files identified as the pretrained MOFTransformer and "
        "two fine-tuned models in the authors' public Figshare article. Dataset archives are "
        "excluded. The Figshare API supplies exact file IDs, URLs, sizes, and MD5 values; no "
        "file contents are downloaded."
    )

    def __init__(
        self,
        *,
        name: str = "moftransformer-figshare-checkpoints",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {"adapter": "moftransformer-figshare-v1", "article_id": _ARTICLE_ID,
             "checkpoints": sorted(_CHECKPOINTS)}
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(_API_URL, headers={"Accept": "application/json"})
        if response.status != 200:
            raise ValueError(f"{self.name}: Figshare API returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: article metadata exceeds {self.max_response_bytes} bytes"
            )
        article = response.json()
        if not isinstance(article, Mapping) or article.get("id") != _ARTICLE_ID:
            raise ValueError(f"{self.name}: API response is not the expected Figshare article")
        if article.get("is_public") is not True or article.get("download_disabled") is True:
            raise ValueError(f"{self.name}: article is not publicly downloadable")
        if article.get("title") != "MOFTransformer":
            raise ValueError(f"{self.name}: unexpected Figshare article title")
        article_version = article.get("version")
        if not isinstance(article_version, int) or article_version <= 0:
            raise ValueError(f"{self.name}: missing Figshare article version")
        file_rows = _parse_files(article.get("files"), article_version, self.name)
        records = tuple(self._record(row) for row in file_rows)
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return SourcePage(
            records,
            {"checked_at": checked_at, "article_id": _ARTICLE_ID,
             "article_url": _ARTICLE_URL, "source_sha256": content_hash(response.body),
             "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, file_row: Mapping[str, Any]) -> SourceRecord:
        file_id = file_row["id"]
        filename = file_row["name"]
        url = file_row["download_url"]
        model_id = f"model:{file_id}"
        identifier = Identifier("moftransformer:figshare-file", str(file_id))
        model = ModelHint(
            model_id, f"MOFTransformer checkpoint {filename}", aliases=(filename,),
            identifiers=(identifier,), status=ModelStatus.RELEASED,
        )
        md5 = file_row["computed_md5"]
        release = ReleaseHint(
            f"figshare-file:{file_id}", model_id,
            revision=f"figshare-article-{_ARTICLE_ID}-v{file_row['article_version']}",
            identifiers=(Identifier("figshare:file", str(file_id)),),
            metadata={"provider": "Figshare", "article_id": _ARTICLE_ID,
                      "article_version": file_row["article_version"], "file_id": file_id,
                      "filename": filename, "size_bytes": file_row["size"],
                      "md5": md5, "binary_reachability_checked": False},
        )
        return SourceRecord(
            source_record_id=f"figshare-file:{file_id}", kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url), title=model.name,
            raw={"article_id": _ARTICLE_ID, "file_id": file_id, "filename": filename,
                 "download_url": url, "size_bytes": file_row["size"], "md5": md5},
            text=f"MOFTransformer checkpoint file: {filename}", identifiers=(identifier,),
            links=(Link(_ARTICLE_URL, "model_card", crawl=False, model_local_ids=(model_id,)),
                   Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                   Link("https://github.com/hspark1212/MOFTransformer",
                        "source_implementation", crawl=False)),
            models=(model,), releases=(release,),
        )


def _parse_files(
    files: Any, article_version: int, source: str
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(files, list):
        raise ValueError(f"{source}: Figshare article has no file list")
    matches: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: malformed Figshare file metadata")
        name = row.get("name")
        if name not in _CHECKPOINTS:
            continue
        if name in matches:
            raise ValueError(f"{source}: duplicate checkpoint filename {name!r}")
        file_id = row.get("id")
        size = row.get("size")
        download_url = row.get("download_url")
        md5 = row.get("computed_md5")
        if not isinstance(file_id, int) or file_id <= 0 or not isinstance(size, int) or size <= 0:
            raise ValueError(f"{source}: invalid file ID or size for {name!r}")
        parsed = urlsplit(download_url) if isinstance(download_url, str) else None
        if (
            parsed is None
            or parsed.scheme != "https"
            or parsed.netloc != "ndownloader.figshare.com"
            or parsed.path != f"/files/{file_id}"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"{source}: invalid Figshare download URL for {name!r}")
        if not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5):
            raise ValueError(f"{source}: missing exact checksum for {name!r}")
        row = dict(row)
        row["article_version"] = article_version
        matches[name] = row
    if matches.keys() != _CHECKPOINTS:
        missing = sorted(_CHECKPOINTS - matches.keys())
        raise ValueError(f"{source}: Figshare article is missing checkpoint rows {missing}")
    return tuple(matches[name] for name in sorted(matches))


__all__ = ["MOFTransformerFigshareAdapter"]
