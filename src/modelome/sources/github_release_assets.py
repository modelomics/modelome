"""Project checkpoint-like release assets from preserved GitHub event payloads.

This is a local projection over GH Archive event records. It makes no API calls
and only sees release assets included in an observed ``ReleaseEvent`` payload.
It is therefore an activity-based discovery adjunct, not a catalog of every
release or asset on GitHub.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from modelome.models import ArtifactKind, Identifier, Link, SourceRecord

_CHECKPOINT_SUFFIXES = frozenset(
    {
        ".safetensors",
        ".pt",
        ".pth",
        ".bin",
        ".ckpt",
        ".model",
        ".onnx",
        ".gguf",
        ".ggml",
        ".h5",
        ".keras",
        ".tflite",
        ".pb",
        ".mlmodel",
        ".msgpack",
    }
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
    if isinstance(max_assets, bool) or not isinstance(max_assets, int) or max_assets < 1:
        raise ValueError("max_assets must be a positive integer")
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

    result: list[SourceRecord] = []
    for asset in assets[:max_assets]:
        if not isinstance(asset, Mapping):
            continue
        asset_id = _positive_decimal(asset.get("id"))
        name = _text(asset.get("name"))
        if not asset_id or not name or not _is_checkpoint_name(name):
            continue
        download_url = _asset_url(asset.get("browser_download_url"), repository_name)
        if not download_url:
            continue
        result.append(
            SourceRecord(
                source_record_id=(
                    f"github-release-asset:{repository_id}:{release_id}:{asset_id}"
                ),
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
                },
            )
        )
    return tuple(result)


def _is_checkpoint_name(value: str) -> bool:
    lowered = value.casefold()
    return any(lowered.endswith(suffix) for suffix in _CHECKPOINT_SUFFIXES)


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
