from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from modelome.normalize import content_hash
from modelome.pipeline import SyncEngine, SyncOutcome
from modelome.sources.huggingface import HuggingFaceSourceAdapter, _hub_repo_id
from modelome.storage import Database


class HuggingFaceRevisionPilotSourceAdapter(HuggingFaceSourceAdapter):
    """Traverse historical revisions for a fixed, exact set of Hub model repos.

    This adapter deliberately does not enumerate the global model catalog. Use
    :func:`run_huggingface_revision_pilot` to impose a per-invocation request cap.
    """

    def __init__(
        self,
        *,
        repo_ids: Iterable[str],
        name: str = "huggingface-revision-pilot",
        max_revision_tree_pages: int = 20,
        max_revision_weight_file_state_bytes: int = 262_144,
        max_response_bytes: int = 16 * 1024 * 1024,
        token: str | None = None,
        client: Any | None = None,
    ) -> None:
        normalized = tuple(sorted({str(repo_id).strip() for repo_id in repo_ids}))
        if not normalized:
            raise ValueError("repo_ids must contain at least one exact Hugging Face model ID")
        if any(not _hub_repo_id(repo_id) for repo_id in normalized):
            raise ValueError("repo_ids must contain exact owner/repository IDs")
        self.repo_ids = normalized
        super().__init__(
            name=name,
            url="https://huggingface.co/api/models",
            max_response_bytes=max_response_bytes,
            token=token,
            include_revisions=True,
            include_revision_files=True,
            max_revision_tree_pages=max_revision_tree_pages,
            max_revision_weight_file_state_bytes=max_revision_weight_file_state_bytes,
            client=client,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "huggingface-revision-pilot-v1",
                "base_signature": self.checkpoint_signature,
                "repo_ids": list(self.repo_ids),
            }
        )

    def fetch_page(self, state: Mapping[str, Any]):
        if "revision_queue" not in state:
            seeded_state = {
                "revision_queue": [
                    {"model_id": repo_id, "phase": "refs"} for repo_id in self.repo_ids
                ],
                "revision_listing_complete": True,
                "base_state": {},
            }
            return self._fetch_revision_page(seeded_state, self._headers())
        return self._fetch_revision_page(state, self._headers())

    def _headers(self) -> Mapping[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def run_huggingface_revision_pilot(
    database: Database,
    source: HuggingFaceRevisionPilotSourceAdapter,
    *,
    request_budget: int,
) -> SyncOutcome:
    """Run at most ``request_budget`` Hub requests and retain the resume cursor."""

    if not isinstance(request_budget, int) or isinstance(request_budget, bool):
        raise ValueError("request_budget must be a positive integer")
    if request_budget < 1:
        raise ValueError("request_budget must be a positive integer")
    return SyncEngine(database, {source.name: source}).sync_source(
        source,
        max_pages=request_budget,
    )
