from __future__ import annotations

import re
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient
from modelome.institutional import InstitutionalTextOutcome, ingest_institutional_text
from modelome.normalize import canonicalize_url, extract_urls

DEFAULT_MAX_TEXT_BYTES = 16 * 1024 * 1024
DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)

_EXPLICIT_MODEL_MARKERS = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "zenodo.org",
    "figshare.com",
    "huggingface.co",
    "osf.io",
    "codeocean.com",
    "kaggle.com",
    "modelscope.cn",
    "bioimage.io",
)

_GENERIC_HOSTS = frozenset(
    {
        "doi.org",
        "creativecommons.org",
        "orcid.org",
        "scholar.google.com",
        "google.com",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "europepmc.org",
        "crossref.org",
        "twitter.com",
        "x.com",
        "facebook.com",
        "linkedin.com",
        "w3.org",
    }
)

_PAREN_PAIRS = (("(", ")"), ("[", "]"), ("{", "}"))


@dataclass(frozen=True, slots=True)
class TextArtifactInventory:
    """Model-artifact hints recovered from locally extracted paper text."""

    urls: tuple[str, ...]
    artifact_urls: tuple[str, ...]
    doi: str | None


def extract_doi(text: str) -> str | None:
    """Locate the first DOI in free text and strip trailing junk."""
    match = DOI_RE.search(text)
    if not match:
        return None
    doi = match.group(0).casefold()
    while True:
        changed = False
        while doi and doi[-1] in ".,;:)>]}\"'”’\u2026":
            doi = doi[:-1]
            changed = True
        for opener, closer in _PAREN_PAIRS:
            if (
                doi.startswith(opener)
                and doi.endswith(closer)
                and doi.count(opener) == doi.count(closer)
            ):
                doi = doi[len(opener) : -len(closer)]
                changed = True
        if doi.endswith(")") and doi.count(")") > doi.count("("):
            doi = doi[:-1]
            changed = True
        if not changed:
            return doi or None


def inventory_text_artifacts(text: str) -> TextArtifactInventory:
    """Extract, classify, and deduplicate every URL found in plain paper text."""
    seen: list[str] = []
    artifact: list[str] = []
    doi = extract_doi(text)
    if doi:
        artifact.append(f"https://doi.org/{quote(doi, safe='/():._-')}")
    for raw in extract_urls(text):
        try:
            url = canonicalize_url(raw)
        except ValueError:
            continue
        if any(
            url.casefold().startswith(scheme)
            for scheme in ("mailto:", "javascript:", "data:", "file:", "ftp:")
        ):
            continue
        if url not in seen:
            seen.append(url)
        lowered = url.casefold()
        if any(host in lowered for host in _GENERIC_HOSTS):
            continue
        if any(marker in lowered for marker in _EXPLICIT_MODEL_MARKERS) and url not in artifact:
            artifact.append(url)
    return TextArtifactInventory(
        urls=tuple(seen),
        artifact_urls=tuple(artifact),
        doi=doi,
    )


def _normalize_doi(doi: str) -> str:
    """Return a normalized DOI string, or raise."""
    candidate = doi.casefold().removeprefix("https://doi.org/").removeprefix("http://doi.org/").removeprefix("doi:")
    candidate = candidate.strip()
    if not candidate:
        raise ValueError("doi must not be empty")
    return candidate


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes, *, max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES
) -> str:
    """Extract UTF-8 text from raw PDF content, with a basic fallback parser."""
    parts: list[str] = []
    for match in re.finditer(rb"\(([^()]{2,})\)\s*Tj", pdf_bytes):
        try:
            parts.append(match.group(1).decode("latin-1", errors="ignore"))
        except UnicodeDecodeError:
            continue
    for match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, flags=re.DOTALL):
        stream = match.group(1)
        try:
            decompressed = zlib.decompress(stream)
        except (zlib.error, ValueError):
            continue
        for text_match in re.finditer(rb"\(([^()]{2,})\)\s*Tj", decompressed):
            try:
                parts.append(text_match.group(1).decode("latin-1", errors="ignore"))
            except UnicodeDecodeError:
                continue
        if sum(len(p) for p in parts) > max_text_bytes:
            break
    return "\n".join(parts)


def read_text_file(path: Path, *, max_bytes: int = DEFAULT_MAX_TEXT_BYTES) -> str:
    """Read and validate a local UTF-8 text file (never a PDF)."""
    if not path.is_file():
        raise ValueError(f"text path is not a regular file: {path}")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError(f"text file exceeds {max_bytes} byte limit: {path}")
    if b"\x00" in raw:
        raise ValueError("accepts extracted UTF-8 text, not a binary file")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("text file must be UTF-8") from error
    if not text.strip():
        raise ValueError("text must not be empty")
    return text


def fetch_oa_pdf_url(
    doi: str,
    *,
    client: Any | None = None,
) -> tuple[str, str] | None:
    """Return the first OA PDF locatable via OpenAlex for a DOI."""
    http = client or HttpClient()
    doi_norm = _normalize_doi(doi)
    api_url = f"https://api.openalex.org/works/doi:{doi_norm}"
    try:
        response = http.get(api_url)
        data = response.json()
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    locations = data.get("locations")
    if not isinstance(locations, (list, tuple)):
        return None
    for location in locations:
        if not isinstance(location, Mapping):
            continue
        pdf_url = location.get("pdf_url")
        version = str(location.get("version") or "").casefold()
        if pdf_url and isinstance(pdf_url, str) and pdf_url.casefold().startswith("https://"):
            return (pdf_url, f"oa/{version}")
    return None


def resolve_lawful_access_routes(
    doi: str,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Report lawful full-text access routes for a DOI, without bypassing paywalls."""
    doi_norm = _normalize_doi(doi)
    http = client or HttpClient()
    result: dict[str, Any] = {
        "doi": doi_norm,
        "open_access": None,
        "institutional": None,
        "oa_pdf_url": None,
        "oa_version": None,
        "landing_url": f"https://doi.org/{quote(doi_norm, safe='/():._-')}",
        "advice": (
            "If no OA copy exists: use --institutional-access while on UC Berkeley's network "
            "to download the PDF via EZproxy (manual step; then `modelome ingest-text` to import "
            "the extracted text)"
        ),
    }
    try:
        oa = fetch_oa_pdf_url(doi_norm, client=http)
        if oa:
            result["oa_pdf_url"], result["oa_version"] = oa
            result["open_access"] = True
            result["advice"] = (
                "OA copy located; download PDF and ingest with `modelome ingest-text`"
            )
        else:
            result["open_access"] = False
    except Exception:
        result["open_access"] = None
    return result


def ingest_locally_authorized_text(
    database: Any,
    *,
    doi: str,
    title: str,
    text: str,
    access_confirmed: bool,
    landing_url: str | None = None,
) -> InstitutionalTextOutcome:
    """Delegate to the institutional import guardrail after extracting text."""
    return ingest_institutional_text(
        database,
        doi=doi,
        title=title,
        text=text,
        access_confirmed=access_confirmed,
        landing_url=landing_url,
    )
