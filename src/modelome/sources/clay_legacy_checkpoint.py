"""Project Clay's specifically documented non-Hub v0 checkpoint."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_DOC_URL = "https://clay-foundation.github.io/model/clay-v0/model_embeddings.html"
_S3_URI = "s3://clay-model-ckpt/v0/clay-small-70MT-1100T-10E.ckpt"
_MODEL_ID = "clay-v0-small-70mt-1100t-10e"
_COMMAND = re.compile(
    r"aws\s+s3\s+cp\s+" + re.escape(_S3_URI) + r"(?:\s|$)", re.IGNORECASE
)


class ClayLegacyCheckpointSourceAdapter:
    """Track the one Clay v0 checkpoint explicitly named on first-party docs."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the Clay v0 checkpoint and S3 URI explicitly shown in the "
        "official Clay documentation. It does not enumerate the bucket, download "
        "weights, claim public HTTP access, or duplicate Clay v1.5 Hugging Face files."
    )

    def __init__(
        self,
        *,
        name: str = "clay-legacy-checkpoint",
        url: str = _DOC_URL,
        max_response_bytes: int = 2 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if url != _DOC_URL:
            raise ValueError(f"documentation URL must be {_DOC_URL}")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.url = _DOC_URL
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "clay-legacy-checkpoint-v1",
                "documentation_url": self.url,
                "s3_uri": _S3_URI,
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(self.url, headers={"Accept": "text/html"})
        if response.status != 200:
            raise ValueError(f"{self.name}: documentation returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: documentation exceeds configured byte limit")
        body = response.body.decode("utf-8")
        if not _COMMAND.search(body):
            raise ValueError(f"{self.name}: documented Clay v0 checkpoint command was not found")
        digest = hashlib.sha256(response.body).hexdigest()
        if state.get("completed_digest") == digest:
            return SourcePage((), dict(state), True, upstream_count=1)
        record = _record(self.name, digest)
        return SourcePage(
            (record,),
            {"completed_digest": digest, "checkpoint_count": 1},
            True,
            upstream_count=1,
        )


def _record(source: str, source_digest: str) -> SourceRecord:
    return SourceRecord(
        source_record_id=f"{source}:{_MODEL_ID}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=_DOC_URL,
        title="Clay v0 small pretrained checkpoint (70M samples, 1100T, 10 epochs)",
        raw={
            "model_id": _MODEL_ID,
            "s3_uri": _S3_URI,
            "documentation_url": _DOC_URL,
            "documentation_sha256": source_digest,
        },
        identifiers=(Identifier("clay:model", _MODEL_ID),),
        links=(
            Link(
                _DOC_URL,
                relation="documentation",
                locator="Clay v0 embedding-generation instructions",
                crawl=False,
                model_local_ids=(_MODEL_ID,),
            ),
        ),
        models=(
            ModelHint(
                local_id=_MODEL_ID,
                name="Clay v0 small",
                identifiers=(Identifier("clay:model", _MODEL_ID),),
                locator="clay-small-70MT-1100T-10E.ckpt",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{_MODEL_ID}@v0",
                model_local_id=_MODEL_ID,
                revision="v0",
                identifiers=(Identifier("clay:checkpoint", _S3_URI),),
                metadata={"checkpoint_uri": _S3_URI},
                locator="official documentation AWS CLI copy command",
            ),
        ),
    )
