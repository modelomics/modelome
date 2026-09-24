from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

from modelome.extract import IntroductionCueExtractor
from modelome.models import ArtifactKind, Identifier, Link, SourceRecord
from modelome.normalize import canonicalize_url
from modelome.storage import Database

DEFAULT_MAX_TEXT_BYTES = 16 * 1024 * 1024
_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class InstitutionalTextOutcome:
    """Receipt for one user-authorized, locally supplied publisher-text import."""

    source: str
    doi: str
    access_scope: str
    text_bytes: int
    stats: dict[str, int]


def ingest_institutional_text(
    database: Database,
    *,
    doi: str,
    title: str,
    text_path: str | Path | None = None,
    text: str | None = None,
    access_confirmed: bool,
    landing_url: str | None = None,
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
) -> InstitutionalTextOutcome:
    """Ingest manually obtained text after an authorized user confirms its use.

    This is deliberately an on-demand local import, not a browser automation or
    text-and-data-mining client. Source text stays in the private evidence store;
    the metadata exporter excludes it.

    Either ``text_path`` (a local UTF-8 file) or ``text`` (the already-extracted
    string) must be provided, never both.
    """

    if not access_confirmed:
        raise ValueError("institutional text import requires --access-confirmed")
    if isinstance(max_text_bytes, bool) or max_text_bytes < 1:
        raise ValueError("max_text_bytes must be positive")
    if (text_path is None) == (text is None):
        raise ValueError("provide exactly one of text_path or text")

    normalized_doi = _normalize_doi(doi)
    normalized_title = _required_text(title, "title")

    if text is not None:
        payload = text.encode("utf-8")
        if len(payload) > max_text_bytes:
            raise ValueError(f"text string exceeds {max_text_bytes} byte limit")
        if b"\0" in payload:
            raise ValueError("institutional import accepts extracted UTF-8 text, not binary data")
        if not text.strip():
            raise ValueError("institutional text must contain non-whitespace text")
    else:
        path = Path(text_path)  # type: ignore[union-attr]
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValueError(f"could not stat text file {path}") from error
        if not path.is_file():
            raise ValueError(f"text path is not a regular file: {path}")
        if size < 1:
            raise ValueError("institutional text file must not be empty")
        if size > max_text_bytes:
            raise ValueError(f"institutional text file exceeds {max_text_bytes} byte limit: {path}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"could not read text file {path}") from error
        if len(payload) > max_text_bytes:
            raise ValueError(f"institutional text file exceeds {max_text_bytes} byte limit: {path}")
        if b"\0" in payload:
            raise ValueError("institutional import accepts extracted UTF-8 text, not a binary file")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("institutional text file must be UTF-8") from error
        if not text.strip():
            raise ValueError("institutional text file must contain non-whitespace text")

    doi_url = canonicalize_url(f"https://doi.org/{quote(normalized_doi, safe='/():._-')}")
    links: tuple[Link, ...] = ()
    if landing_url is not None and landing_url.strip():
        normalized_landing_url = _web_url(landing_url)
        if normalized_landing_url != doi_url:
            links = (
                Link(
                    normalized_landing_url,
                    relation="institutional_landing_page",
                    locator="operator:landing_url",
                    crawl=False,
                ),
            )

    record = SourceRecord(
        source_record_id=normalized_doi,
        kind=ArtifactKind.PAPER,
        canonical_url=doi_url,
        title=normalized_title,
        text=text,
        raw={
            "access_scope": "institutional-restricted",
            "access_method": "authorized-user-manual-import",
            "export_policy": "metadata-only; source text excluded",
            "text_encoding": "utf-8",
            "text_bytes": len(payload),
        },
        identifiers=(Identifier("doi", normalized_doi),),
        links=links,
    )
    source = "institutional-text"
    stats = database.ingest_page(
        source,
        (record,),
        {
            "mode": "on-demand-authorized-user-import",
            "last_doi": normalized_doi,
        },
        extractor=IntroductionCueExtractor(),
        complete=False,
    )
    return InstitutionalTextOutcome(
        source=source,
        doi=normalized_doi,
        access_scope="institutional-restricted",
        text_bytes=len(payload),
        stats=stats,
    )


def _normalize_doi(value: str) -> str:
    candidate = _required_text(value, "doi").casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
            break
    candidate = candidate.strip().rstrip('.,;:!?)"]}')
    if not _DOI_RE.fullmatch(candidate):
        raise ValueError("doi must be a valid DOI")
    return candidate


def _web_url(value: str) -> str:
    candidate = _required_text(value, "landing_url")
    try:
        parts = urlsplit(candidate)
        port = parts.port
    except ValueError as error:
        raise ValueError("landing_url must be an absolute HTTP(S) URL") from error
    if (
        parts.scheme.casefold() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
    ):
        raise ValueError("landing_url must be an absolute HTTP(S) URL without credentials")
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("landing_url port is out of range")
    return canonicalize_url(candidate)


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


__all__ = ["DEFAULT_MAX_TEXT_BYTES", "InstitutionalTextOutcome", "ingest_institutional_text"]
