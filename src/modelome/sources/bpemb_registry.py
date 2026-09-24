"""Pinned BPEmb language-index reader for official pretrained vector archives."""

from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE = re.compile(r"^[a-z0-9]{2,12}$")
_LANGUAGE_LINK = re.compile(
    r"\[[^\]]+\]\(https?://nlp\.h-its\.org/bpemb/(?P<lang>[a-z0-9]{2,12})/?\)"
)
_VECTOR_ARCHIVE = re.compile(
    r"^(?P<lang>[a-z0-9]{2,12})\.wiki\.bpe\.vs(?P<vs>[0-9]+)\."
    r"d(?P<dim>[0-9]+)\.w2v\.(?P<format>bin|txt)\.tar\.gz$"
)
_REPOSITORY = "bheinzerling/bpemb"
_README = "README.md"
_CODE_HOSTS = {"bpemb.h-its.org", "nlp.h-its.org"}
_MONOLINGUAL_HOST = "https://nlp.h-its.org/bpemb/"
_MULTI_HOST = "https://bpemb.h-its.org/multi/multi/"
_LANGUAGES_PER_PAGE = 10


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _ArchiveLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.urls.add(html.unescape(href))


class BPEmbPretrainedVectorRegistrySourceAdapter:
    """Enumerate BPEmb's distinct monolingual and multilingual vector weights.

    README language links define the pages to inspect. Those first-party pages
    publish direct text and binary archives with exact language, vocabulary,
    and dimension identities; segmentation model files and plots are excluded.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers direct BPEmb vector archives linked from the project's monolingual "
        "language pages and multilingual page. Segmentation-only SentencePiece "
        "files, visualizations, and model bytes are excluded."
    )

    def __init__(
        self,
        *,
        name: str = "bpemb-pretrained-vectors",
        repository: str = _REPOSITORY,
        branch: str = "master",
        provider_namespace: str = "bpemb:pretrained-vector",
        max_response_bytes: int = 4 * 1024 * 1024,
        max_entries: int = 10000,
        max_languages: int = 300,
        languages_per_page: int = _LANGUAGES_PER_PAGE,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if (
            not name.strip()
            or repository != _REPOSITORY
            or not branch.strip()
            or not provider_namespace.strip()
        ):
            raise ValueError("name, official repository, branch, and namespace are required")
        if min(max_response_bytes, max_entries, max_languages, languages_per_page) <= 0:
            raise ValueError("response, entry, language, and page limits must be positive")
        if languages_per_page > max_languages:
            raise ValueError("languages per page cannot exceed the language limit")
        self.name, self.repository, self.branch = name, repository, branch
        self.provider_namespace = provider_namespace
        self.max_response_bytes, self.max_entries = max_response_bytes, max_entries
        self.max_languages, self.languages_per_page = max_languages, languages_per_page
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "bpemb-pretrained-vector-registry-v1",
                "repository": repository,
                "branch": branch,
                "readme": _README,
                "monolingual_page_base": _MONOLINGUAL_HOST,
                "multilingual_page": _MULTI_HOST,
                "archive_pattern": _VECTOR_ARCHIVE.pattern,
                "languages_per_page": languages_per_page,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision = state.get("catalog_revision")
        languages = state.get("languages")
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            revision, languages = self._load_language_index()
        if not isinstance(languages, list) or not languages:
            raise ValueError(f"{self.name}: no supported language pages found")
        if (
            len(languages) > self.max_languages
            or any(not isinstance(language, str) or not _LANGUAGE.fullmatch(language)
                   for language in languages)
            or len(set(languages)) != len(languages)
            or "en" not in languages
            or "multi" not in languages
        ):
            raise ValueError(f"{self.name}: invalid language index in cursor")
        offset = state.get("language_offset", 0)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise ValueError(f"{self.name}: invalid language cursor")
        if offset > len(languages):
            raise ValueError(f"{self.name}: language cursor exceeds language count")
        if offset == len(languages):
            model_count = state.get("model_count")
            if (
                not isinstance(model_count, int)
                or isinstance(model_count, bool)
                or model_count < 0
                or model_count > self.max_entries
            ):
                raise ValueError(f"{self.name}: invalid terminal model count")
            return SourcePage(
                (),
                {**state, "language_offset": len(languages)},
                True,
                upstream_count=model_count,
            )
        batch = languages[offset : offset + self.languages_per_page]
        rows: dict[str, dict[str, Any]] = {}
        for language in batch:
            page_url = (
                f"{_MONOLINGUAL_HOST}{quote(language, safe='')}/"
                if language != "multi"
                else _MULTI_HOST
            )
            response = self.client.get(page_url, headers={"Accept": "text/html"})
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: language page {language} returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(f"{self.name}: language page {language} exceeds response limit")
            for variant, urls in _vector_rows(response.body, page_url, language, self.name):
                key = f"{language}:vs{variant[0]}:d{variant[1]}"
                rows.setdefault(
                    key, {"language": language, "vs": variant[0], "dim": variant[1], "urls": set()}
                )["urls"].update(urls)
        previous_count = state.get("model_count", 0)
        if (
            not isinstance(previous_count, int)
            or isinstance(previous_count, bool)
            or previous_count < 0
        ):
            raise ValueError(f"{self.name}: invalid cumulative model count")
        cumulative_count = previous_count + len(rows)
        if cumulative_count > self.max_entries:
            raise ValueError(f"{self.name}: vector count exceeds configured limit")
        records = tuple(self._record(row, revision) for _, row in sorted(rows.items()))
        next_offset = offset + len(batch)
        next_state = {
            "catalog_revision": revision,
            "languages": languages,
            "language_offset": next_offset,
            "checked_at": self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "model_count": cumulative_count,
        }
        return SourcePage(
            records,
            next_state,
            next_offset >= len(languages),
            upstream_count=cumulative_count if next_offset >= len(languages) else None,
            # Page cursors can resume in a later SyncEngine run, whose run_id
            # differs from the runs that ingested earlier batches. A terminal
            # page therefore cannot safely tombstone unseen prior-batch rows.
            authoritative_snapshot=False,
        )

    def _load_language_index(self) -> tuple[str, list[str]]:
        commit = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if commit.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {commit.status}")
        payload = commit.json()
        revision = payload.get("sha") if isinstance(payload, Mapping) else None
        if not isinstance(revision, str) or not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: invalid catalog revision")
        readme_url = f"https://raw.githubusercontent.com/{self.repository}/{revision}/{_README}"
        readme = self.client.get(readme_url, headers={"Accept": "text/plain"})
        if readme.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {readme.status}")
        if len(readme.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds response limit")
        try:
            text = readme.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{self.name}: README is not UTF-8") from exc
        languages = list(dict.fromkeys(_LANGUAGE_LINK.findall(text)))
        # MultiBPEmb is documented as a separate page and uses its own direct
        # archive family, so include it alongside the monolingual index.
        if "multi" not in languages:
            languages.append("multi")
        if not languages or "en" not in languages:
            raise ValueError(f"{self.name}: README has no valid language links")
        return revision, languages

    def _record(self, row: Mapping[str, Any], revision: str) -> SourceRecord:
        lang, vs, dim = row["language"], row["vs"], row["dim"]
        model_id = f"model:{lang}:vs{vs}:d{dim}"
        page_url = _MULTI_HOST if lang == "multi" else f"{_MONOLINGUAL_HOST}{lang}/"
        archive_urls = tuple(sorted(row["urls"]))
        model = ModelHint(
            model_id,
            f"BPEmb {lang} BPE subword vectors",
            identifiers=(Identifier(self.provider_namespace, f"{lang}:vs{vs}:d{dim}"),),
            aliases=(f"{lang}.wiki.bpe.vs{vs}.d{dim}",),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{lang}:vs{vs}:d{dim}",
            model_id,
            version=f"vs{vs}-d{dim}",
            identifiers=(
                Identifier(f"{self.provider_namespace}:release", f"{lang}:vs{vs}:d{dim}"),
            ),
            metadata={
                "language": lang,
                "vocabulary_size": vs,
                "dimension": dim,
                "weight_urls": archive_urls,
                "catalog_revision": revision,
            },
        )
        return SourceRecord(
            source_record_id=model_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(
                next(url for url in archive_urls if ".bin.tar.gz" in url)
            ),
            title=f"BPEmb {lang} {vs} {dim}-dimensional subword vectors",
            raw={
                "language": lang,
                "vocabulary_size": vs,
                "dimension": dim,
                "weight_urls": archive_urls,
                "language_page": page_url,
                "catalog_revision": revision,
            },
            text=(
                f"BPEmb pretrained subword embedding archive for {lang}, "
                f"vocabulary {vs}, dimension {dim}."
            ),
            identifiers=(Identifier(self.provider_namespace, f"{lang}:vs{vs}:d{dim}"),),
            links=(
                *(
                    Link(url, "weights", crawl=False, model_local_ids=(model_id,))
                    for url in archive_urls
                ),
                Link(page_url, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(
                    self.repository_url,
                    "source_implementation",
                    crawl=False,
                    model_local_ids=(model_id,),
                ),
            ),
            models=(model,),
            releases=(release,),
        )


def _vector_rows(
    body: bytes, page_url: str, expected_language: str, source: str
) -> tuple[tuple[tuple[int, int], tuple[str, ...]], ...]:
    try:
        page_text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: language page is not UTF-8") from exc
    parser = _ArchiveLinks()
    parser.feed(page_text)
    vectors: dict[tuple[int, int], set[str]] = defaultdict(set)
    for href in parser.urls:
        url = urljoin(page_url, href)
        parsed = urlsplit(url)
        match = _VECTOR_ARCHIVE.fullmatch(parsed.path.rsplit("/", 1)[-1])
        if not match:
            continue
        if parsed.scheme != "https" or parsed.hostname not in _CODE_HOSTS:
            continue
        if match.group("lang") != expected_language:
            raise ValueError(f"{source}: language page declares another language asset")
        key = int(match.group("vs")), int(match.group("dim"))
        vectors[key].add(url)
    rows: list[tuple[tuple[int, int], tuple[str, ...]]] = []
    for variant, urls in sorted(vectors.items()):
        formats = {
            _VECTOR_ARCHIVE.fullmatch(urlsplit(url).path.rsplit("/", 1)[-1]).group("format")
            for url in urls
        }
        if "bin" not in formats:
            continue
        rows.append((variant, tuple(sorted(urls))))
    return tuple(rows)
