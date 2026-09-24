"""Google Research Football's two documented PPO checkpoint objects."""

from __future__ import annotations

import re
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

_README_URL = "https://raw.githubusercontent.com/google-research/football/master/README.md"
_REPOSITORY_URL = "https://github.com/google-research/football"
_ARTIFACT_ROOT = "https://storage.googleapis.com/gfootball-public-bucket/"
_CHECKPOINTS = {
    "11_vs_11_easy_stochastic": "trained_model_11_vs_11_easy_stochastic",
    "academy_run_to_score_with_keeper": (
        "trained_model_academy_run_to_score_with_keeper_v2"
    ),
}
_LINK = re.compile(r"\[(?P<scenario>[^\]]+)\]\((?P<url>https://[^)]+)\)")


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GoogleFootballCheckpointAdapter:
    """Index the two exact trained PPO objects linked from the official README."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers only the two PPO checkpoint objects linked in Google Research "
        "Football's README. It excludes other scenarios, training runs, and object bytes."
    )

    def __init__(
        self,
        *,
        name: str = "google-football-ppo-checkpoints",
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
                "adapter": "google-football-ppo-checkpoints-v1",
                "readme_url": _README_URL,
                "checkpoint_names": _CHECKPOINTS,
                "max_response_bytes": max_response_bytes,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        response: HttpResponse = self.client.get(
            _README_URL, headers={"Accept": "text/plain"}
        )
        if response.status != 200:
            raise ValueError(f"{self.name}: project README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: project README exceeds response limit")
        text = response.text()
        records = self._records(text)
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "readme_sha256": content_hash(response.body),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _records(self, text: str) -> tuple[SourceRecord, ...]:
        observed: dict[str, str] = {}
        for match in _LINK.finditer(text):
            scenario, url = match.group("scenario", "url")
            if scenario not in _CHECKPOINTS:
                continue
            if scenario in observed:
                raise ValueError(f"{self.name}: duplicate checkpoint for {scenario!r}")
            expected_url = f"{_ARTIFACT_ROOT}{_CHECKPOINTS[scenario]}"
            if url != expected_url:
                raise ValueError(f"{self.name}: unexpected checkpoint URL for {scenario!r}")
            _validate_url(url, self.name)
            observed[scenario] = url
        if set(observed) != set(_CHECKPOINTS):
            raise ValueError(f"{self.name}: documented checkpoint inventory is incomplete")
        return tuple(self._record(key, observed[key]) for key in sorted(observed))

    def _record(self, scenario: str, url: str) -> SourceRecord:
        object_name = _CHECKPOINTS[scenario]
        local_id = f"checkpoint:google-football:{scenario}"
        identifier = Identifier("google-football:ppo-checkpoint", scenario)
        model = ModelHint(
            local_id=local_id,
            name=f"Google Research Football PPO {scenario}",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=object_name,
        )
        release = ReleaseHint(
            local_id=f"release:google-football:{scenario}",
            model_local_id=local_id,
            identifiers=(Identifier("google-football:checkpoint-object", object_name),),
            metadata={
                "algorithm": "PPO",
                "scenario": scenario,
                "checkpoint_object": object_name,
                "weight_url": url,
                "tensorflow_version": "1.15",
            },
            locator=object_name,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(url),
            title=f"Google Research Football PPO checkpoint: {scenario}",
            raw={"scenario": scenario, "checkpoint_object": object_name, "weight_url": url},
            text=(
                "Official trained PPO checkpoint for Google Research Football "
                f"scenario {scenario}."
            ),
            identifiers=(identifier,),
            links=(
                Link(_README_URL, "model_card", crawl=False, model_local_ids=(local_id,)),
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _validate_url(url: str, source: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "storage.googleapis.com"
        or parsed.path.split("/")[:2] != ["", "gfootball-public-bucket"]
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{source}: checkpoint URL is outside the documented public bucket")


__all__ = ["GoogleFootballCheckpointAdapter"]
