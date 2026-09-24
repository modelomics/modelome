"""Bounded extractor for declared model files in bioRxiv/medRxiv TDM MECA packages.

This module is intentionally not registered in the default source catalog. The
first-party TDM buckets are requester-pays bulk archives, so callers must opt in
with an AWS client and a fixed monthly prefix. Listing checkpoints are scoped to
that immutable month prefix.
"""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import quote, urlsplit

from modelome.models import ArtifactKind, Identifier, Link, SourcePage, SourceRecord
from modelome.normalize import canonicalize_url, content_hash

_SERVERS = {
    "biorxiv": ("biorxiv-src-monthly", "www.biorxiv.org"),
    "medrxiv": ("medrxiv-src-monthly", "www.medrxiv.org"),
}
_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_RESOURCE_CUE_RE = re.compile(
    r"\b(?:pre[- ]?trained|trained|downloadable|released)\s+(?:deep[- ]learning\s+)?models?\b|"
    r"\b(?:model\s+)?(?:weights?|checkpoints?|parameters?)\b|"
    r"\bweights?\s+for\s+(?:the\s+)?(?:trained\s+)?models?\b",
    re.IGNORECASE,
)
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


class BioRxivTdmSourceAdapter:
    """Page one first-party MECA S3 prefix using requester-pays reads.

    The adapter uses S3's opaque continuation token, processes no more than
    ``max_objects_per_page`` objects per call, and spools each archive to disk
    after the in-memory threshold. Archives and decompressed XML have separate
    byte limits. Supply a boto3-compatible client; boto3 is not a project
    dependency and no bucket access occurs at import time.
    """

    def __init__(
        self,
        *,
        server: str,
        prefix: str,
        client: Any,
        max_objects_per_page: int = 25,
        max_archive_bytes: int = 256 * 1024 * 1024,
        max_manifest_bytes: int = 4 * 1024 * 1024,
        max_article_xml_bytes: int = 64 * 1024 * 1024,
        max_zip_entries: int = 20_000,
        max_supplement_links: int = 2_000,
    ) -> None:
        self.server = _required_server(server)
        self.bucket, self.host = _SERVERS[self.server]
        self.prefix = _required_prefix(prefix)
        self.client = client
        self.max_objects_per_page = _positive(max_objects_per_page, "max_objects_per_page")
        if self.max_objects_per_page > 1000:
            raise ValueError("max_objects_per_page must not exceed the S3 ListObjectsV2 limit")
        self.max_archive_bytes = _positive(max_archive_bytes, "max_archive_bytes")
        self.max_manifest_bytes = _positive(max_manifest_bytes, "max_manifest_bytes")
        self.max_article_xml_bytes = _positive(max_article_xml_bytes, "max_article_xml_bytes")
        self.max_zip_entries = _positive(max_zip_entries, "max_zip_entries")
        self.max_supplement_links = _positive(max_supplement_links, "max_supplement_links")
        self.name = f"{self.server}-tdm:{self.prefix}"
        self.checkpoint_signature = content_hash(
            {
                "adapter": "biorxiv-tdm-meca-supplementary-v1",
                "server": self.server,
                "bucket": self.bucket,
                "prefix": self.prefix,
                "max_objects_per_page": self.max_objects_per_page,
                "max_archive_bytes": self.max_archive_bytes,
                "max_manifest_bytes": self.max_manifest_bytes,
                "max_article_xml_bytes": self.max_article_xml_bytes,
                "max_zip_entries": self.max_zip_entries,
                "max_supplement_links": self.max_supplement_links,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state.get("bucket", self.bucket) != self.bucket:
            raise ValueError(f"{self.name}: checkpoint bucket does not match source")
        if state.get("prefix", self.prefix) != self.prefix:
            raise ValueError(f"{self.name}: checkpoint prefix does not match source")
        if state.get("complete") is True:
            return SourcePage(records=(), next_state=dict(state), complete=True, upstream_count=0)
        token = state.get("continuation_token")
        if token is not None and (not isinstance(token, str) or not token):
            raise ValueError(f"{self.name}: invalid continuation_token")
        request: dict[str, Any] = {
            "Bucket": self.bucket,
            "Prefix": self.prefix,
            "MaxKeys": self.max_objects_per_page,
            "RequestPayer": "requester",
        }
        if token is not None:
            request["ContinuationToken"] = token
        page = self.client.list_objects_v2(**request)
        if not isinstance(page, Mapping):
            raise ValueError(f"{self.name}: S3 listing response must be an object")
        contents = page.get("Contents", ())
        if not isinstance(contents, list):
            raise ValueError(f"{self.name}: S3 listing Contents must be an array")
        if len(contents) > self.max_objects_per_page:
            raise ValueError(f"{self.name}: S3 listing exceeds configured object limit")
        truncated = page.get("IsTruncated")
        if not isinstance(truncated, bool):
            raise ValueError(f"{self.name}: S3 listing is missing IsTruncated")
        next_token = page.get("NextContinuationToken")
        if truncated and (not isinstance(next_token, str) or not next_token):
            raise ValueError(f"{self.name}: truncated S3 listing is missing continuation token")

        records: list[SourceRecord] = []
        for item in contents:
            if not isinstance(item, Mapping) or not isinstance(item.get("Key"), str):
                raise ValueError(f"{self.name}: S3 object entry is malformed")
            key = item["Key"]
            if not key.startswith(self.prefix):
                raise ValueError(f"{self.name}: S3 object key escaped configured prefix")
            if not key.casefold().endswith(".meca"):
                continue
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=key,
                RequestPayer="requester",
            )
            body = response.get("Body")
            if body is None or not hasattr(body, "read"):
                raise ValueError(f"{self.name}: S3 object body is missing or not readable")
            content_length = response.get("ContentLength")
            if content_length is not None and (
                not isinstance(content_length, int) or content_length < 0
            ):
                raise ValueError(f"{self.name}: invalid S3 ContentLength")
            if content_length is not None and content_length > self.max_archive_bytes:
                raise ValueError(f"{self.name}: MECA archive exceeds configured byte limit")
            try:
                with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as archive:
                    _copy_bounded(body, archive, self.max_archive_bytes, self.name)
                    archive.seek(0)
                    records.extend(self._archive_records(key, archive))
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()

        raw_seen = _nonnegative(state.get("raw_objects_seen", 0), "raw_objects_seen")
        raw_seen += len(contents)
        next_state = {
            "complete": not truncated,
            "raw_objects_seen": raw_seen,
            "prefix": self.prefix,
            "bucket": self.bucket,
        }
        if truncated:
            next_state["continuation_token"] = next_token
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=not truncated,
            upstream_count=raw_seen,
        )

    def _archive_records(self, key: str, archive: BinaryIO) -> tuple[SourceRecord, ...]:
        try:
            with zipfile.ZipFile(archive) as package:
                members = package.infolist()
                if len(members) > self.max_zip_entries:
                    raise ValueError(f"{self.name}: MECA archive exceeds entry limit")
                names = {info.filename for info in members if not info.is_dir()}
                manifest_name = next(
                    (
                        name
                        for name in names
                        if PurePosixPath(name).name.casefold() == "manifest.xml"
                    ),
                    None,
                )
                if manifest_name is None:
                    raise ValueError(f"{self.name}: MECA package has no manifest.xml")
                _read_xml_member(package, manifest_name, self.max_manifest_bytes, self.name)
                xml_members = sorted(
                    name
                    for name in names
                    if name.startswith("content/") and name.casefold().endswith(".xml")
                )
                if not xml_members:
                    raise ValueError(f"{self.name}: MECA package has no content XML")
                if len(xml_members) > 64:
                    raise ValueError(f"{self.name}: MECA package has too many content XML files")
                output: list[SourceRecord] = []
                for xml_name in xml_members:
                    article_bytes = _read_xml_member(
                        package, xml_name, self.max_article_xml_bytes, self.name
                    )
                    article = _safe_xml(article_bytes, self.name)
                    if _local_name(article.tag) != "article":
                        continue
                    output.extend(self._article_records(key, xml_name, article, names))
                return tuple(output)
        except zipfile.BadZipFile as error:
            raise ValueError(f"{self.name}: invalid MECA ZIP package {key!r}") from error

    def _article_records(
        self,
        key: str,
        xml_name: str,
        article: ET.Element,
        members: set[str],
    ) -> tuple[SourceRecord, ...]:
        article_id = next(
            (node for node in article.iter() if _local_name(node.tag) == "article-id"
             and _text(node.get("pub-id-type")).casefold() == "doi"),
            None,
        )
        doi = _normalize_doi(_element_text(article_id))
        if not doi:
            return ()
        title = next(
            (
                _element_text(node)
                for node in article.iter()
                if _local_name(node.tag) == "article-title"
            ),
            doi,
        ) or doi
        supplement_nodes = [
            node for node in article.iter()
            if _local_name(node.tag) in {"supplementary-material", "inline-supplementary-material"}
        ]
        links: list[Link] = []
        evidence: list[dict[str, str]] = []
        for node in supplement_nodes:
            context = " ".join(" ".join(node.itertext()).split())
            if not _RESOURCE_CUE_RE.search(context):
                continue
            hrefs: list[str] = []
            for candidate in node.iter():
                href = _text(candidate.get(_XLINK_HREF) or candidate.get("href"))
                if href:
                    hrefs.append(href)
            for href in dict.fromkeys(hrefs):
                direct = _web_url(href)
                if direct:
                    url = canonicalize_url(direct)
                    locator = f"{xml_name}:{_local_name(node.tag)}[{len(evidence)}]"
                else:
                    member = _resolve_content_member(href, members)
                    if member is None:
                        continue
                    # S3 object URIs preserve the precise MECA provenance. The
                    # member is in the evidence locator because S3 cannot address
                    # a single ZIP member as an object URL.
                    url = f"s3://{self.bucket}/{quote(key, safe='/')}"
                    locator = f"{xml_name}:{member}"
                evidence.append({"url": url, "locator": locator})
                links.append(Link(url, relation="model_artifact", locator=locator, crawl=False))
                if len(links) > self.max_supplement_links:
                    raise ValueError(f"{self.name}: package exceeds supplementary link limit")
        if not links:
            return ()
        server_url = canonicalize_url(f"https://{self.host}/content/{quote(doi, safe='/.:_-()')}")
        links.insert(0, Link(server_url, relation="preprint", locator=xml_name, crawl=False))
        return (
            SourceRecord(
                source_record_id=(
                    f"{self.server}:tdm:{doi}:"
                    f"{content_hash({'key': key, 'xml': xml_name})[:20]}"
                ),
                kind=ArtifactKind.PAPER,
                canonical_url=server_url,
                title=title,
                raw={
                    "tdm_bucket": self.bucket,
                    "tdm_key": key,
                    "jats_member": xml_name,
                    "jats_doi": doi,
                    "supplementary_model_resources": evidence,
                },
                identifiers=(Identifier("doi", doi),),
                links=tuple(links),
            ),
        )


def _copy_bounded(source: Any, destination: BinaryIO, limit: int, label: str) -> int:
    total = 0
    while True:
        block = source.read(min(1024 * 1024, limit + 1 - total))
        if not block:
            return total
        total += len(block)
        if total > limit:
            raise ValueError(f"{label}: MECA archive exceeds configured byte limit")
        destination.write(block)


def _read_xml_member(
    package: zipfile.ZipFile, name: str, limit: int, label: str
) -> bytes:
    info = package.getinfo(name)
    if info.file_size > limit:
        raise ValueError(f"{label}: ZIP member {name!r} exceeds configured byte limit")
    with package.open(info) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"{label}: ZIP member {name!r} exceeds configured byte limit")
    return data


def _safe_xml(data: bytes, label: str) -> ET.Element:
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", data, flags=re.IGNORECASE):
        raise ValueError(f"{label}: unsafe XML declaration in MECA package")
    try:
        return ET.fromstring(data)
    except ET.ParseError as error:
        raise ValueError(f"{label}: malformed XML in MECA package: {error}") from None


def _resolve_content_member(href: str, members: set[str]) -> str | None:
    path = PurePosixPath(href.split("?", 1)[0].split("#", 1)[0])
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return None
    candidates = ("/".join(path.parts), f"content/{'/'.join(path.parts)}")
    return next((item for item in candidates if item in members), None)


def _normalize_doi(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    match = _DOI_RE.fullmatch(text.rstrip(".,;:"))
    return match.group(0).casefold() if match else ""


def _web_url(value: str) -> str | None:
    parts = urlsplit(value)
    if parts.scheme in {"http", "https"} and parts.hostname:
        return value
    return None


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _element_text(node: ET.Element | None) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def _required_server(value: str) -> str:
    server = value.strip().casefold()
    if server not in _SERVERS:
        raise ValueError("server must be 'biorxiv' or 'medrxiv'")
    return server


def _required_prefix(value: str) -> str:
    prefix = value.strip()
    if not prefix or not prefix.endswith("/") or prefix.startswith("/"):
        raise ValueError("prefix must be a nonempty relative S3 prefix ending in '/'")
    if ".." in PurePosixPath(prefix).parts:
        raise ValueError("prefix must not contain parent path segments")
    return prefix


def _positive(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a positive integer") from None
    if result <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return result


def _nonnegative(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a nonnegative integer") from None
    if result < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
