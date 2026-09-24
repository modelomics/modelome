"""Exact pretrained model files published by the SDSS SSL project at NERSC."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
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

_CATALOG_URL = (
    "https://portal.nersc.gov/project/dasrepo/"
    "self-supervised-learning-sdss/models.html"
)
_ASSET_ROOT = (
    "https://portal.nersc.gov/project/dasrepo/"
    "self-supervised-learning-sdss/checkpoints/"
)
_MODEL_FILES = {
    "pretrained_paper_model.pth.tar": (
        "SDSS self-supervised pretrained encoder",
        "CNN encoder trained with the project's contrastive self-supervised method.",
    ),
    "photoz_finetuned_model.pth.tar": (
        "SDSS photo-z fine-tuned encoder",
        "CNN encoder fine-tuned to estimate redshift using spectroscopic labels.",
    ),
    "photoz_supervised_baseline_model.pth.tar": (
        "SDSS supervised photo-z baseline",
        "Fully supervised CNN baseline trained for redshift estimation.",
    ),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _Anchors(HTMLParser):
    def __init__(self, maximum: int) -> None:
        super().__init__(convert_charrefs=True)
        self.maximum = maximum
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)
            if len(self.hrefs) > self.maximum:
                raise ValueError(f"catalog exceeds {self.maximum} links")


class SdssSslCheckpointsSourceAdapter:
    """Record the three exact checkpoint files listed by the project authors."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the three named pretrained checkpoint files listed on the SDSS "
        "SSL project's NERSC models page. No additional files or versions are "
        "inferred; file bytes are never downloaded, and the source page does "
        "not provide a revisioned manifest or checksums."
    )

    def __init__(
        self,
        *,
        name: str = "sdss-ssl-checkpoints",
        url: str = _CATALOG_URL,
        max_response_bytes: int = 2 * 1024 * 1024,
        max_links: int = 2_000,
        client: Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        self.name = name.strip()
        if _valid_page_url(url) is None or canonicalize_url(url) != _CATALOG_URL:
            raise ValueError(f"catalog URL must be {_CATALOG_URL}")
        self.url = _CATALOG_URL
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self.max_links = _positive_int(max_links, "max_links")
        self.client = client or HttpClient(max_response_bytes=self.max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "sdss-ssl-checkpoints-v1",
                "url": self.url,
                "expected_files": sorted(_MODEL_FILES),
                "max_response_bytes": self.max_response_bytes,
                "max_links": self.max_links,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            self.url, headers={"Accept": "text/html,application/xhtml+xml"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: model page returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(
                f"{self.name}: model page exceeds {self.max_response_bytes} bytes"
            )
        parser = _Anchors(self.max_links)
        parser.feed(response.text())
        parser.close()
        observed: dict[str, str] = {}
        for href in parser.hrefs:
            asset_url = _valid_asset_url(href)
            if asset_url is None:
                continue
            filename = urlsplit(asset_url).path.rsplit("/", 1)[-1]
            if filename not in _MODEL_FILES:
                continue
            if filename in observed and observed[filename] != asset_url:
                raise ValueError(f"{self.name}: duplicate checkpoint filename {filename}")
            observed[filename] = asset_url
        missing = set(_MODEL_FILES) - set(observed)
        if missing:
            raise ValueError(
                f"{self.name}: expected model checkpoint links missing: "
                f"{', '.join(sorted(missing))}"
            )

        checked_at = self.clock().astimezone(UTC).isoformat()
        document_hash = content_hash(response.body)
        records = tuple(
            self._record(filename, observed[filename], document_hash)
            for filename in _MODEL_FILES
        )
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "catalog_url": self.url,
                "catalog_sha256": document_hash,
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, filename: str, asset_url: str, catalog_hash: str) -> SourceRecord:
        name, description = _MODEL_FILES[filename]
        identity = f"{self.name}:{filename}"
        model_id = f"model:{filename}"
        model_identifier = Identifier("sdss-ssl:checkpoint", filename)
        model = ModelHint(
            local_id=model_id,
            name=name,
            aliases=(filename,),
            identifiers=(model_identifier,),
            status=ModelStatus.RELEASED,
            locator=f"{self.url}#model:{filename}",
        )
        release = ReleaseHint(
            local_id=f"release:{filename}",
            model_local_id=model_id,
            identifiers=(Identifier("sdss-ssl:release-file", filename),),
            metadata={
                "checkpoint_filename": filename,
                "checkpoint_url": asset_url,
                "catalog_url": self.url,
                "catalog_sha256": catalog_hash,
            },
            locator=f"{self.url}#file:{filename}",
        )
        return SourceRecord(
            source_record_id=identity,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=asset_url,
            title=name,
            raw={
                "catalog_url": self.url,
                "catalog_sha256": catalog_hash,
                "checkpoint_filename": filename,
                "checkpoint_url": asset_url,
                "description": description,
            },
            text=f"{name}. {description}",
            identifiers=(model_identifier,),
            links=(
                Link(asset_url, relation="weights", crawl=False),
                Link(self.url, relation="source_model_catalog", crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


def _valid_page_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    return value.strip()


def _valid_asset_url(value: str) -> str | None:
    if value.startswith("/"):
        from urllib.parse import urljoin

        value = urljoin(_CATALOG_URL, value)
    valid = _valid_page_url(value)
    if valid is None or not valid.startswith(_ASSET_ROOT):
        return None
    if not valid.endswith(".pth.tar") or "?" in valid or "#" in valid:
        return None
    return valid


def _positive_int(value: int, label: str) -> int:
    result = int(value)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result
