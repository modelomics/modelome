"""Exact DQN checkpoints listed by Facebook Research's PHYRE downloader."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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

_SCRIPT_URL = (
    "https://raw.githubusercontent.com/facebookresearch/phyre/main/"
    "agents/download_dqn_ckps.sh"
)
_REPOSITORY_URL = "https://github.com/facebookresearch/phyre"
_ARTIFACT_ROOT = "https://dl.fbaipublicfiles.com/phyre/"
_TEMPLATES = (
    "ball_cross_template",
    "ball_within_template",
    "two_balls_cross_template",
    "two_balls_within_template",
)
_SEEDS = tuple(str(seed) for seed in range(10))
_CHECKPOINT = "ckpt.00100000"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class PhyreDqnCheckpointAdapter:
    """Index the 40 final-split DQN checkpoints from PHYRE's official script."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the 40 final-split PHYRE DQN checkpoints emitted by the "
        "official download script (four evaluation setups and ten seeds). It "
        "excludes results JSON files, dev splits, other agents, and checkpoint bytes."
    )

    def __init__(
        self,
        *,
        name: str = "phyre-dqn-checkpoints",
        max_response_bytes: int = 64 * 1024,
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
                "adapter": "phyre-dqn-checkpoints-v1",
                "script_url": _SCRIPT_URL,
                "templates": _TEMPLATES,
                "seeds": _SEEDS,
                "checkpoint": _CHECKPOINT,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(
            _SCRIPT_URL, headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: download script returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: download script exceeds response limit")
        paths = self._paths(response.text())
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(template, seed, path) for template, seed, path in paths)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "script_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _paths(self, script: str) -> tuple[tuple[str, str, str], ...]:
        # Fail closed when upstream changes the finite loops or URL composition.
        expected = (
            "for seed in $(seq 0 9); do",
            "for tpl in " + " ".join(_TEMPLATES) + "; do",
            'path="results/finals/dqn_10k/$tpl/$seed/$fname"',
            'for fname in ckpt.00100000 results.json; do',
            'wget "https://dl.fbaipublicfiles.com/phyre/$path"',
        )
        if any(fragment not in script for fragment in expected):
            raise ValueError(f"{self.name}: official checkpoint inventory contract changed")
        return tuple(
            (
                template,
                seed,
                f"results/finals/dqn_10k/{template}/{seed}/{_CHECKPOINT}",
            )
            for template in _TEMPLATES
            for seed in _SEEDS
        )

    def _record(self, template: str, seed: str, path: str) -> SourceRecord:
        url = f"{_ARTIFACT_ROOT}{path}"
        _validate_url(url, path, self.name)
        checkpoint_id = f"{template}/{seed}"
        local_id = f"checkpoint:phyre-dqn:{template}:{seed}"
        identifier = Identifier("phyre:dqn-checkpoint", checkpoint_id)
        model = ModelHint(
            local_id=local_id,
            name=f"PHYRE DQN {template} seed {seed}",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=path,
        )
        release = ReleaseHint(
            local_id=f"release:phyre-dqn:{template}:{seed}",
            model_local_id=local_id,
            identifiers=(Identifier("phyre:checkpoint-path", path),),
            metadata={
                "algorithm": "DQN",
                "benchmark": "PHYRE",
                "evaluation_setup": template,
                "seed": int(seed),
                "checkpoint_path": path,
                "weight_url": url,
            },
            locator=path,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"PHYRE DQN checkpoint: {template}, seed {seed}",
            raw={
                "evaluation_setup": template,
                "seed": int(seed),
                "checkpoint_path": path,
                "weight_url": url,
            },
            text=(
                "Official final-split PHYRE DQN checkpoint for "
                f"{template}, seed {seed}."
            ),
            identifiers=(identifier,),
            links=(
                Link(_SCRIPT_URL, "checkpoint_manifest", crawl=False,
                     model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_url(url: str, path: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "dl.fbaipublicfiles.com"
        or parsed.path != f"/phyre/{path}"
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: checkpoint URL is outside the official PHYRE path")


__all__ = ["PhyreDqnCheckpointAdapter"]
