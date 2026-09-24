"""Piper's first-party voice/quality index with exact model and config assets."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

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

Clock = Callable[[], datetime]
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LANGUAGE_CODE = re.compile(r"\(([`']?)(?P<code>[a-z]{2}_[A-Z]{2})\1(?:[,)]|$)")
_MODEL_ROW = re.compile(
    r"^\s{8}\*\s*(?P<quality>x_low|low|medium|high)\s+-\s+"
    r"\[\[model\]\((?P<model>https://[^)]+)\)\]\s+"
    r"\[\[config\]\((?P<config>https://[^)]+)\)\]\s*$"
)
_VOICE_ROW = re.compile(r"^\s{4}\*\s+(?P<voice>[^*].*?)\s*$")
_ASSET_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PiperVoiceCatalogSourceAdapter:
    """Read named Piper voice variants and paired ONNX/config files.

    The source is Piper's archived first-party ``VOICES.md`` index. Links pin
    assets to the ``rhasspy/piper-voices`` v1.0.0 dataset revision; only URL
    metadata is fetched, never ONNX bytes.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers exact voice/quality rows in rhasspy/piper VOICES.md and their "
        "pinned piper-voices v1.0.0 assets. It does not enumerate user-submitted "
        "voices or later assets absent from that archived index."
    )

    def __init__(
        self,
        *,
        name: str = "piper-voice-catalog",
        repository: str = "rhasspy/piper",
        branch: str = "master",
        document_path: str = "VOICES.md",
        max_response_bytes: int = 2 * 1024 * 1024,
        max_rows: int = 2_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if repository != "rhasspy/piper" or document_path != "VOICES.md":
            raise ValueError("Piper voice catalog requires rhasspy/piper VOICES.md")
        if not name.strip() or not branch.strip():
            raise ValueError("source name and branch must not be empty")
        for label, value in (("max_response_bytes", max_response_bytes), ("max_rows", max_rows)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch.strip()
        self.document_path = document_path
        self.max_response_bytes = max_response_bytes
        self.max_rows = max_rows
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "piper-voice-catalog-v1",
                "repository": repository,
                "branch": self.branch,
                "document_path": document_path,
                "max_response_bytes": max_response_bytes,
                "max_rows": max_rows,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    @property
    def commit_url(self) -> str:
        return (
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}"
        )

    def raw_url(self, revision: str) -> str:
        return (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/{self.document_path}"
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        revision, commit_response = self._revision()
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        if revision == _text(state.get("completed_revision")):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative(state.get("model_count")),
            )
        response: HttpResponse = self.client.get(
            self.raw_url(revision), headers={"Accept": "text/markdown,text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: voice index returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: voice index exceeds response limit")
        rows = _parse_voice_rows(response.text(), source=self.name, maximum=self.max_rows)
        records = tuple(self._record(row, revision, response.body) for row in rows)
        if not records:
            raise ValueError(f"{self.name}: voice index contains no model rows")
        next_state: dict[str, Any] = {
            "completed_revision": revision,
            "checked_at": checked_at,
            "document_url": self.raw_url(revision),
            "document_sha256": content_hash(response.body),
            "model_count": len(records),
        }
        if etag := _header(commit_response.headers, "etag"):
            next_state["commit_etag"] = etag
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _revision(self) -> tuple[str, HttpResponse]:
        response: HttpResponse = self.client.get(
            self.commit_url, headers={"Accept": "application/vnd.github+json"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = _text(payload.get("sha")) if isinstance(payload, Mapping) else ""
        if not _COMMIT.fullmatch(revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        return revision, response

    def _record(self, row: _VoiceRow, revision: str, document: bytes) -> SourceRecord:
        identity = f"{row.language}/{row.voice}/{row.quality}"
        local_id = f"model:{identity}"
        identifier = Identifier("piper:voice", identity)
        model = ModelHint(
            local_id=local_id,
            name=f"Piper {row.language} {row.voice} ({row.quality})",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=row.locator,
        )
        release = ReleaseHint(
            local_id=f"release:{identity}",
            model_local_id=local_id,
            version="v1.0.0",
            revision="v1.0.0",
            identifiers=(Identifier("piper:voice-release", identity),),
            metadata={
                "language": row.language,
                "voice": row.voice,
                "quality": row.quality,
                "asset_revision": "v1.0.0",
                "model_url": row.model_url,
                "config_url": row.config_url,
            },
            locator=row.locator,
        )
        card_url = f"{self.repository_url}/blob/{revision}/{self.document_path}"
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(card_url),
            title=model.name,
            raw={
                "repository": self.repository,
                "revision": revision,
                "document_sha256": content_hash(document),
                "language": row.language,
                "voice": row.voice,
                "quality": row.quality,
                "model_url": row.model_url,
                "config_url": row.config_url,
            },
            text=(
                f"Piper text-to-speech voice {row.voice} for {row.language} "
                f"at {row.quality} quality."
            ),
            identifiers=(identifier,),
            links=(
                Link(card_url, "model_card", locator=row.locator, crawl=False),
                Link(self.repository_url, "source_implementation", crawl=False),
                Link(row.model_url, "weights", locator=row.locator, crawl=False),
                Link(row.config_url, "model_artifact", locator=row.locator, crawl=False),
            ),
            models=(model,),
            releases=(release,),
        )


class _VoiceRow:
    __slots__ = ("language", "voice", "quality", "model_url", "config_url", "locator")

    def __init__(
        self, language: str, voice: str, quality: str, model_url: str, config_url: str, locator: str
    ) -> None:
        self.language = language
        self.voice = voice
        self.quality = quality
        self.model_url = model_url
        self.config_url = config_url
        self.locator = locator


def _parse_voice_rows(document: str, *, source: str, maximum: int) -> tuple[_VoiceRow, ...]:
    language: str | None = None
    voice: str | None = None
    rows: list[_VoiceRow] = []
    for line_number, line in enumerate(document.splitlines(), start=1):
        if len(line) - len(line.lstrip()) <= 2 and (match := _LANGUAGE_CODE.search(line)):
            language = match.group("code")
            voice = None
            continue
        if not line.startswith("      ") and (match := _VOICE_ROW.match(line)):
            voice = match.group("voice").strip()
            continue
        model_match = _MODEL_ROW.match(line)
        if model_match is None:
            continue
        if language is None or voice is None:
            raise ValueError(f"{source}: voice quality appears outside a language/voice group")
        model_url = model_match.group("model")
        config_url = model_match.group("config")
        _validate_asset_url(model_url, "onnx", source)
        _validate_asset_url(config_url, "onnx.json", source)
        rows.append(
            _VoiceRow(
                language,
                voice,
                model_match.group("quality"),
                model_url,
                config_url,
                f"VOICES.md:L{line_number}",
            )
        )
        if len(rows) > maximum:
            raise ValueError(f"{source}: voice index exceeds {maximum} rows")
    if not rows:
        raise ValueError(f"{source}: no exact model/config voice rows were recognized")
    identities = [(row.language, row.voice, row.quality) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{source}: duplicate language/voice/quality identity")
    return tuple(rows)


def _validate_asset_url(url: str, suffix: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "huggingface.co"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query not in {"download=true", "download=true.json"}
        or parsed.fragment
        or not parsed.path.startswith("/rhasspy/piper-voices/resolve/v1.0.0/")
        or not parsed.path.endswith(suffix)
    ):
        raise ValueError(f"{source}: invalid pinned Piper asset URL")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _nonnegative(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next(
        (value for key, value in headers.items() if key.casefold() == name.casefold()), None
    )


__all__ = ["PiperVoiceCatalogSourceAdapter"]
