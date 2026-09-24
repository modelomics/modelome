"""PMTransformer's first-party pretrained checkpoint on Figshare."""

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

_ARTICLE_ID = 22698655
_ARTICLE_URL = "https://figshare.com/articles/dataset/PMTransformer_pre-trained_model/22698655"
_API_URL = f"https://api.figshare.com/v2/articles/{_ARTICLE_ID}"
_FILE_NAME = "pmtransformer.ckpt"


class PMTransformerFigshareAdapter:
    """Index the exact PMTransformer checkpoint row from its authors' Figshare record."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the single pmtransformer.ckpt file in the authors' public Figshare article. "
        "The companion MOFTransformer checkpoint and JSON archive are excluded because they "
        "are covered elsewhere or are not model checkpoints. Metadata only; no file bytes are read."
    )

    def __init__(
        self,
        *,
        name: str = "pmtransformer-figshare-checkpoint",
        client: HttpClient | Any | None = None,
        max_response_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if not name.strip() or max_response_bytes <= 0:
            raise ValueError("name and positive response limit are required")
        self.name = name
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.checkpoint_signature = content_hash(
            {"adapter": "pmtransformer-figshare-v1", "article_id": _ARTICLE_ID,
             "filename": _FILE_NAME}
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
        if article.get("title") != "PMTransformer pre-trained model":
            raise ValueError(f"{self.name}: unexpected Figshare article title")
        version = article.get("version")
        if not isinstance(version, int) or version <= 0:
            raise ValueError(f"{self.name}: missing Figshare article version")
        row = _parse_file(article.get("files"), version, self.name)
        record = self._record(row)
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return SourcePage(
            (record,),
            {"checked_at": checked_at, "article_id": _ARTICLE_ID,
             "article_url": _ARTICLE_URL, "source_sha256": content_hash(response.body),
             "checkpoint_count": 1},
            True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(self, row: Mapping[str, Any]) -> SourceRecord:
        file_id = row["id"]
        url = row["download_url"]
        model_id = f"model:{file_id}"
        identifier = Identifier("pmtransformer:figshare-file", str(file_id))
        model = ModelHint(
            model_id, "PMTransformer pretrained checkpoint", aliases=(_FILE_NAME,),
            identifiers=(identifier,), status=ModelStatus.RELEASED,
        )
        md5 = row["computed_md5"]
        release = ReleaseHint(
            f"figshare-file:{file_id}", model_id,
            revision=f"figshare-article-{_ARTICLE_ID}-v{row['article_version']}",
            identifiers=(Identifier("figshare:file", str(file_id)),),
            metadata={"provider": "Figshare", "article_id": _ARTICLE_ID,
                      "article_version": row["article_version"], "file_id": file_id,
                      "filename": _FILE_NAME, "size_bytes": row["size"], "md5": md5,
                      "binary_reachability_checked": False},
        )
        return SourceRecord(
            source_record_id=f"figshare-file:{file_id}", kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url), title=model.name,
            raw={"article_id": _ARTICLE_ID, "file_id": file_id, "filename": _FILE_NAME,
                 "download_url": url, "size_bytes": row["size"], "md5": md5},
            text=f"PMTransformer checkpoint file: {_FILE_NAME}", identifiers=(identifier,),
            links=(Link(_ARTICLE_URL, "model_card", crawl=False, model_local_ids=(model_id,)),
                   Link(url, "weights", crawl=False, model_local_ids=(model_id,)),
                   Link("https://github.com/hspark1212/MOFTransformer",
                        "source_implementation", crawl=False)),
            models=(model,), releases=(release,),
        )


def _parse_file(files: Any, version: int, source: str) -> Mapping[str, Any]:
    if not isinstance(files, list):
        raise ValueError(f"{source}: Figshare article has no file list")
    matches = []
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError(f"{source}: malformed Figshare file metadata")
        if row.get("name") == _FILE_NAME:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"{source}: expected exactly one {_FILE_NAME} row")
    row = matches[0]
    file_id, size, url, md5 = (
        row.get("id"), row.get("size"), row.get("download_url"), row.get("computed_md5")
    )
    if not isinstance(file_id, int) or file_id <= 0 or not isinstance(size, int) or size <= 0:
        raise ValueError(f"{source}: invalid file ID or size for {_FILE_NAME}")
    parsed = urlsplit(url) if isinstance(url, str) else None
    if (
        parsed is None or parsed.scheme != "https" or parsed.netloc != "ndownloader.figshare.com"
        or parsed.path != f"/files/{file_id}" or parsed.query or parsed.fragment
    ):
        raise ValueError(f"{source}: invalid Figshare download URL for {_FILE_NAME}")
    if not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError(f"{source}: missing exact checksum for {_FILE_NAME}")
    result = dict(row)
    result["article_version"] = version
    return result


__all__ = ["PMTransformerFigshareAdapter"]
