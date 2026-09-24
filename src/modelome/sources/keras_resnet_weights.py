"""Exact Keras Applications ResNet-family ImageNet weight links."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

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

_REPOSITORY = "keras-team/keras"
_PATH = "keras/src/applications/resnet.py"
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_VARIANT = re.compile(r"^(?:resnet(?:50|101|152)(?:v2)?|resnext(?:50|101))$")


class KerasResNetWeightsSourceAdapter:
    """Read Keras's literal ResNet-family manifest without downloading weights."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only ResNet, ResNetV2, and ResNeXt ImageNet top/no-top assets "
        "declared in keras-team/keras/keras/src/applications/resnet.py."
    )

    def __init__(
        self,
        *,
        name: str = "keras-resnet-imagenet-weights",
        repository: str = _REPOSITORY,
        branch: str = "master",
        source_path: str = _PATH,
        max_source_bytes: int = 2 * 1024 * 1024,
        max_records: int = 32,
        client: HttpClient | Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if not isinstance(branch, str) or not branch.strip():
            raise ValueError("branch must be non-empty text")
        if repository != _REPOSITORY:
            raise ValueError(f"repository must be {_REPOSITORY}")
        if source_path != _PATH:
            raise ValueError(f"source_path must be {_PATH}")
        for value, field in ((max_source_bytes, "max_source_bytes"), (max_records, "max_records")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        self.name = name.strip()
        self.repository = repository
        self.branch = branch.strip()
        self.source_path = source_path
        self.max_source_bytes = max_source_bytes
        self.max_records = max_records
        self.client = client or HttpClient(max_response_bytes=max_source_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "keras-resnet-weights-v1",
                "repository": repository,
                "branch": branch,
                "path": source_path,
                "max_source_bytes": max_source_bytes,
                "max_records": max_records,
            }
        )

    @property
    def repository_url(self) -> str:
        return f"https://github.com/{self.repository}"

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response: HttpResponse = self.client.get(
            f"https://api.github.com/repos/{self.repository}/commits/{quote(self.branch, safe='')}",
            headers={"Accept": "application/vnd.github+json"},
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: commit endpoint returned HTTP {response.status}")
        payload = response.json()
        revision = payload.get("sha", "") if isinstance(payload, Mapping) else ""
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{self.name}: commit endpoint did not return a SHA-1 revision")
        checked_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        if revision == state.get("completed_revision"):
            next_state = dict(state)
            next_state["checked_at"] = checked_at
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=True,
                upstream_count=_nonnegative_int(state.get("model_count")),
            )
        source_url = (
            f"https://raw.githubusercontent.com/{self.repository}/{revision}/"
            f"{quote(self.source_path, safe='/')}"
        )
        source_response: HttpResponse = self.client.get(
            source_url, headers={"Accept": "text/x-python,text/plain"}
        )
        if source_response.status != 200:
            raise ValueError(f"{self.name}: source file returned HTTP {source_response.status}")
        if len(source_response.body) > self.max_source_bytes:
            raise ValueError(f"{self.name}: source file exceeds {self.max_source_bytes} bytes")
        records = _parse_manifest(source_response.text(), source_url, revision, self.name)
        if len(records) > self.max_records:
            raise ValueError(f"{self.name}: source exceeds {self.max_records} checkpoints")
        if not records:
            raise ValueError(f"{self.name}: source contains no supported checkpoints")
        return SourcePage(
            records=records,
            next_state={
                "completed_revision": revision,
                "checked_at": checked_at,
                "source_url": source_url,
                "source_sha256": content_hash(source_response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )


def _parse_manifest(
    text: str, source_url: str, revision: str, source: str
) -> tuple[SourceRecord, ...]:
    try:
        tree = ast.parse(text, filename=_PATH)
    except SyntaxError as error:
        raise ValueError(f"{source}: cannot parse ResNet source: {error.msg}") from error
    manifest: dict[str, Any] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "BASE_WEIGHTS_PATH",
                "WEIGHTS_HASHES",
            }:
                with suppress(ValueError, TypeError):
                    manifest[target.id] = ast.literal_eval(statement.value)
    base = manifest.get("BASE_WEIGHTS_PATH")
    hashes = manifest.get("WEIGHTS_HASHES")
    if not isinstance(base, str) or not base.startswith("https://"):
        raise ValueError(f"{source}: missing literal HTTPS BASE_WEIGHTS_PATH")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"{source}: missing literal WEIGHTS_HASHES mapping")

    records = []
    for variant, pair in sorted(hashes.items()):
        if not isinstance(variant, str) or not _VARIANT.fullmatch(variant):
            raise ValueError(f"{source}: unexpected ResNet variant {variant!r}")
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, str) and _MD5.fullmatch(value) for value in pair)
        ):
            raise ValueError(f"{source}: {variant} must declare top and no-top MD5 hashes")
        for include_top, suffix, checksum in (
            (True, "_weights_tf_dim_ordering_tf_kernels.h5", pair[0]),
            (False, "_weights_tf_dim_ordering_tf_kernels_notop.h5", pair[1]),
        ):
            filename = variant + suffix
            url = canonicalize_url(base + filename)
            model_key = f"keras-applications:resnet-family:{variant}"
            model_local_id = f"model:{content_hash(model_key)[:24]}"
            locator = f"{_PATH}:WEIGHTS_HASHES[{variant!r}]"
            model = ModelHint(
                local_id=model_local_id,
                name=variant,
                aliases=("Keras Applications", "ImageNet"),
                identifiers=(Identifier("keras-applications:model", model_key),),
                status=ModelStatus.RELEASED,
                locator=locator,
            )
            release = ReleaseHint(
                local_id=f"release:{content_hash(url)[:24]}",
                model_local_id=model_local_id,
                revision=revision,
                identifiers=(Identifier("keras-applications:checkpoint", url),),
                metadata={
                    "variant": variant,
                    "include_top": include_top,
                    "filename": filename,
                    "checksum": checksum,
                    "checksum_algorithm": "md5",
                    "url": url,
                    "revision": revision,
                },
                locator=locator,
            )
            records.append(
                SourceRecord(
                    source_record_id=f"keras-resnet:{content_hash(url)[:24]}",
                    kind=ArtifactKind.WEIGHTS,
                    canonical_url=url,
                    title=(
                        f"Keras Applications {variant} ImageNet weights "
                        f"({'top' if include_top else 'no top'})"
                    ),
                    raw={
                        "variant": variant,
                        "include_top": include_top,
                        "url": url,
                        "checksum": checksum,
                        "checksum_algorithm": "md5",
                        "source_url": source_url,
                        "revision": revision,
                    },
                    text=f"{variant} ImageNet pretrained weights; include_top={include_top}.",
                    identifiers=(Identifier("keras-applications:checkpoint", url),),
                    links=(
                        Link(
                            url,
                            relation="model_artifact",
                            locator=locator,
                            crawl=False,
                            model_local_ids=(model_local_id,),
                        ),
                        Link(
                            source_url,
                            relation="preset_definition",
                            locator=locator,
                            crawl=False,
                            model_local_ids=(model_local_id,),
                        ),
                        Link(
                            f"https://github.com/{_REPOSITORY}",
                            relation="source_repository",
                            crawl=False,
                        ),
                    ),
                    models=(model,),
                    releases=(release,),
                )
            )
    return tuple(records)


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
