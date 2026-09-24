"""Project checkpoint-like release assets from preserved GitHub event payloads.

This is a local projection over GH Archive event records. It makes no API calls
and only sees release assets included in an observed ``ReleaseEvent`` payload.
It is therefore an activity-based discovery adjunct, not a catalog of every
release or asset on GitHub.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    SourceRecord,
)

_CHECKPOINT_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".model",
        ".onnx",
        ".engine",
        ".rtxplan",
        ".plan",
        ".gguf",
        ".ggml",
        ".h5",
        ".keras",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".msgpack",
        ".pdparams",
        ".ptl",
        ".pte",
        ".params",
        ".pkl",
        ".weights",
    }
)
_ARCHIVE_SUFFIXES = frozenset({".zip", ".tar", ".tar.gz", ".tgz", ".7z", ".tar.zst"})
_GENERIC_MODEL_NAME_TOKENS = frozenset(
    {
        "ai",
        "asset",
        "artifact",
        "best",
        "binary",
        "checkpoint",
        "data",
        "download",
        "export",
        "file",
        "final",
        "latest",
        "llm",
        "ml",
        "model",
        "network",
        "new",
        "output",
        "payload",
        "params",
        "release",
        "update",
        "version",
        "weights",
    }
)
_MODEL_CONTEXT_RE = re.compile(
    r"\b(?:model|checkpoint|weights?|pretrained|neural|llm|transformer|embedding|"
    r"fine[ -]?tuned)\b",
    re.IGNORECASE,
)
_NON_MODEL_FILE_RE = re.compile(
    r"\b(?:checksum|client|config|dataset|firmware|installer|license|manifest|"
    r"metadata|optimizer|readme|scheduler|server|source|test|tokenizer|vocab)\b",
    re.IGNORECASE,
)
_STRONG_MODEL_SUFFIXES = frozenset(
    {".safetensors", ".gguf", ".ggml", ".keras", ".tflite", ".mlmodel", ".pte"}
)


def project_github_release_assets(
    event_record: SourceRecord,
    *,
    max_assets: int = 100,
) -> tuple[SourceRecord, ...]:
    """Return bounded weight-file candidates found in one GH Archive event row.

    An absent or empty ``payload.release.assets`` list is normal. The function
    validates repository identity and each browser download URL before creating
    records, and does not claim that an asset contains valid model weights.
    """

    if not isinstance(event_record, SourceRecord):
        raise TypeError("event_record must be a SourceRecord")
    if (
        isinstance(max_assets, bool)
        or not isinstance(max_assets, int)
        or not 1 <= max_assets <= 10_000
    ):
        raise ValueError("max_assets must be an integer from 1 to 10000")
    raw = event_record.raw
    if not isinstance(raw, Mapping) or raw.get("record_type") != "gharchive_public_event":
        return ()
    event = raw.get("event")
    repository = raw.get("repository")
    if not isinstance(event, Mapping) or not isinstance(repository, Mapping):
        return ()
    if event.get("type") != "ReleaseEvent":
        return ()
    payload = event.get("payload")
    if not isinstance(payload, Mapping) or payload.get("action") != "published":
        return ()
    release = payload.get("release")
    if not isinstance(release, Mapping):
        return ()

    repository_id = _positive_decimal(repository.get("id"))
    repository_name = _repository_name(repository.get("name"))
    release_id = _positive_decimal(release.get("id"))
    tag_name = _text(release.get("tag_name"))
    release_url = _release_url(release.get("html_url"), repository_name)
    if not all((repository_id, repository_name, release_id, tag_name, release_url)):
        return ()
    assets = release.get("assets")
    if not _is_sequence(assets):
        return ()
    if len(assets) > max_assets:
        raise ValueError(
            f"release event has {len(assets)} assets, above max_assets {max_assets}; "
            "projection would be incomplete"
        )

    result: list[SourceRecord] = []
    release_context = " ".join(_text(release.get(field)) for field in ("name", "body", "tag_name"))
    for asset in assets:
        if not isinstance(asset, Mapping):
            continue
        asset_id = _positive_decimal(asset.get("id"))
        name = _text(asset.get("name"))
        suffix = _asset_suffix(name)
        # Dots are common in model names (for example, ``qwen2.5``). Treat a
        # final short dotted token as a file extension, but allow dotted model
        # identities that end in a longer descriptive token.
        extensionless = bool(name) and not suffix and not re.search(r"\.[A-Za-z0-9]{1,8}$", name)
        checkpoint_file = suffix in _CHECKPOINT_SUFFIXES
        archive_file = suffix in _ARCHIVE_SUFFIXES
        if not asset_id or not name or not (checkpoint_file or archive_file or extensionless):
            continue
        download_url = _asset_url(asset.get("browser_download_url"), repository_name)
        if not download_url:
            continue
        model_name, model_locator = _model_candidate(
            name,
            suffix=suffix,
            asset_label=_text(asset.get("label")),
            release_name=_text(release.get("name")),
            release_tag=tag_name,
            context=" ".join((release_context, _text(asset.get("label")))),
        )
        if archive_file and model_locator not in {
            "$.payload.release.assets[id].name",
            "$.payload.release.assets[id].label",
        }:
            model_name = ""
        if extensionless and (
            not _MODEL_CONTEXT_RE.search(" ".join((release_context, _text(asset.get("label")))))
            or model_locator
            not in {
                "$.payload.release.assets[id].name",
                "$.payload.release.assets[id].label",
            }
        ):
            model_name = ""
        if not checkpoint_file and not model_name:
            continue
        models = (
            (
                ModelHint(
                    local_id="release-asset-model",
                    name=model_name,
                    status=ModelStatus.CANDIDATE,
                    confidence=0.2,
                    locator=model_locator,
                ),
            )
            if model_name
            else ()
        )
        result.append(
            SourceRecord(
                source_record_id=(f"github-release-asset:{repository_id}:{release_id}:{asset_id}"),
                kind=ArtifactKind.WEIGHTS,
                canonical_url=download_url,
                title=name,
                published_at=_text(release.get("published_at")) or None,
                identifiers=(
                    Identifier("github:repository-id", repository_id),
                    Identifier("github:repository", repository_name),
                    Identifier("github:release-id", release_id),
                    Identifier("github:release-tag", tag_name),
                    Identifier("github:release-asset-id", asset_id),
                ),
                links=(Link(release_url, relation="source_release", crawl=False),),
                raw={
                    "record_type": "github_release_asset_candidate",
                    "asset": dict(asset),
                    "release": {
                        "id": release_id,
                        "tag_name": tag_name,
                        "html_url": release_url,
                    },
                    "repository": {
                        "id": repository_id,
                        "name": repository_name,
                    },
                    "event_id": _text(event.get("id")),
                    "event_created_at": _text(event.get("created_at")),
                    "discovery_basis": "github_release_event_payload",
                    "is_verified_model_checkpoint": False,
                    "asset_container_type": "archive" if archive_file else "single_file",
                    "model_hint_basis": (
                        "descriptive_checkpoint_filename_and_release_context"
                        if model_name
                        else None
                    ),
                },
                models=models,
            )
        )
    return tuple(result)


def _is_checkpoint_name(value: str) -> bool:
    lowered = value.casefold()
    return any(lowered.endswith(suffix) for suffix in _CHECKPOINT_SUFFIXES)


def _asset_suffix(value: str) -> str:
    lowered = value.casefold()
    suffixes = _CHECKPOINT_SUFFIXES | _ARCHIVE_SUFFIXES
    return next(
        (suffix for suffix in sorted(suffixes, key=len, reverse=True) if lowered.endswith(suffix)),
        "",
    )


def _model_candidate(
    filename: str,
    *,
    suffix: str,
    asset_label: str,
    release_name: str,
    release_tag: str,
    context: str,
) -> tuple[str, str]:
    stem = filename[: -len(suffix)] if suffix else filename
    if _NON_MODEL_FILE_RE.search(stem):
        return "", ""
    if suffix not in _STRONG_MODEL_SUFFIXES and not _MODEL_CONTEXT_RE.search(context):
        return "", ""
    for candidate_text, locator in (
        (stem, "$.payload.release.assets[id].name"),
        (asset_label, "$.payload.release.assets[id].label"),
        (release_name, "$.payload.release.name"),
        (release_tag, "$.payload.release.tag_name"),
    ):
        name = _descriptive_name(candidate_text)
        if name and (
            suffix in _STRONG_MODEL_SUFFIXES or _context_identifies_candidate(name, context)
        ):
            return name, locator
    return "", ""


def _descriptive_name(value: str) -> str:
    parts = re.split(r"[\s_-]+", value.strip())
    meaningful: list[str] = []
    for part in parts:
        clean = re.sub(r"[^a-zA-Z0-9.+]", "", part)
        folded = clean.casefold()
        if not clean or folded in _GENERIC_MODEL_NAME_TOKENS:
            continue
        if re.fullmatch(r"v\d+(?:\.\d+)*", folded):
            continue
        if meaningful and re.fullmatch(r"\d+(?:\.\d+)*", folded):
            meaningful.append(clean)
            continue
        if meaningful and re.fullmatch(r"\d+(?:\.\d+)?[bmk]", folded):
            meaningful.append(clean)
            continue
        tokens = re.findall(r"[a-z0-9]+", folded)
        if any(
            token not in _GENERIC_MODEL_NAME_TOKENS
            and sum(character.isalpha() for character in token) >= 2
            for token in tokens
        ):
            meaningful.append(clean)
    return " ".join(meaningful)


def _context_identifies_candidate(name: str, context: str) -> bool:
    candidate_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", name.casefold())
        if token not in _GENERIC_MODEL_NAME_TOKENS and len(token) >= 2
    }
    context_tokens = set(re.findall(r"[a-z0-9]+", context.casefold()))
    return bool(candidate_tokens & context_tokens)


def _release_url(value: Any, repository_name: str) -> str:
    url = _text(value)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.casefold().startswith(f"/{repository_name.casefold()}/releases/")
    ):
        return ""
    return url


def _asset_url(value: Any, repository_name: str) -> str:
    url = _text(value)
    parsed = urlsplit(url)
    prefix = f"/{repository_name}/releases/download/".casefold()
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.casefold().startswith(prefix)
    ):
        return ""
    return url


def _repository_name(value: Any) -> str:
    name = _text(value)
    parts = name.split("/")
    if len(parts) != 2 or not all(parts):
        return ""
    return name


def _positive_decimal(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value) if value > 0 else ""
    if isinstance(value, str) and value.isdecimal() and int(value) > 0:
        return str(int(value))
    return ""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


__all__ = ["project_github_release_assets"]
