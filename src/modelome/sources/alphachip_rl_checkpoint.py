"""AlphaChip's documented pretrained reinforcement-learning policy checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any

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

_DOCS_URL = "https://github.com/google-research/circuit_training/blob/main/README.md"
_CHECKPOINT_URL = (
    "https://storage.googleapis.com/rl-infra-public/circuit-training/"
    "tpu_checkpoint_20240815.tar.gz"
)
_REPOSITORY_URL = "https://github.com/google-research/circuit_training"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class _PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []

    def handle_data(self, data: str) -> None:
        self.text.append(data)


class AlphaChipRlCheckpointAdapter:
    """Index the single pretrained AlphaChip policy archive cited in its README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the AlphaChip pretrained checkpoint archive documented in the "
        "project README. It does not enumerate archive members or inspect bytes."
    )

    def __init__(
        self,
        *,
        name: str = "alphachip-rl-checkpoint",
        max_response_bytes: int = 2 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if (
            isinstance(max_response_bytes, bool)
            or not isinstance(max_response_bytes, int)
            or max_response_bytes < 1
        ):
            raise ValueError("max_response_bytes must be a positive integer")
        self.name = name.strip()
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "alphachip-rl-checkpoint-v1",
                "readme_url": _DOCS_URL,
                "checkpoint_url": _CHECKPOINT_URL,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(_DOCS_URL, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: project README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: project README exceeds {self.max_response_bytes} bytes")
        parser = _PageText()
        parser.feed(response.text())
        parser.close()
        if _CHECKPOINT_URL not in " ".join(parser.text):
            raise ValueError(f"{self.name}: documented checkpoint URL is missing")
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        record = self._record()
        return SourcePage(
            records=(record,),
            next_state={"checked_at": checked_at, "readme_sha256": content_hash(response.body)},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        )

    def _record(self) -> SourceRecord:
        local_id = "checkpoint:alphachip-tpu-20240815"
        identifier = Identifier("google-research:alphachip-policy", "tpu_checkpoint_20240815")
        model = ModelHint(
            local_id=local_id,
            name="AlphaChip pretrained placement policy",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator="tpu_checkpoint_20240815",
        )
        release = ReleaseHint(
            local_id="release:alphachip-tpu-20240815",
            model_local_id=local_id,
            identifiers=(Identifier("google-research:alphachip-checkpoint", "20240815"),),
            metadata={
                "archive_filename": "tpu_checkpoint_20240815.tar.gz",
                "pretraining_description": "pre-trained on 20 TPU blocks",
                "weight_url": _CHECKPOINT_URL,
            },
            locator="tpu_checkpoint_20240815.tar.gz",
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(_CHECKPOINT_URL),
            title="AlphaChip pretrained policy checkpoint archive",
            raw={
                "archive_filename": "tpu_checkpoint_20240815.tar.gz",
                "weight_url": _CHECKPOINT_URL,
            },
            text="AlphaChip reinforcement-learning checkpoint pre-trained on 20 TPU blocks.",
            identifiers=(identifier,),
            links=(
                Link(_DOCS_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(_CHECKPOINT_URL, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["AlphaChipRlCheckpointAdapter"]
