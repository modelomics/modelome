from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpClient
from modelome.sources.huggingface import HuggingFaceSourceAdapter

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class EsaFm4csSourceAdapter(HuggingFaceSourceAdapter):
    """Bound the public Hugging Face model/revision inventory to ESA FM4CS.

    FM4CS publishes the THOR model family in its first-party Hub organization.
    The upstream Hub API enumerates exact model repository IDs; the inherited
    revision/file walk resolves immutable revisions and weight-file paths.
    The default catalog enables this scoped inventory alongside the global Hub scan.
    """

    def __init__(
        self,
        *,
        page_size: int = 100,
        max_response_bytes: int = 16 * 1024 * 1024,
        token: str | None = None,
        max_revision_tree_pages: int = 20,
        max_revision_weight_file_state_bytes: int = 262_144,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        super().__init__(
            name="esa-fm4cs",
            url="https://huggingface.co/api/models?author=FM4CS",
            page_size=page_size,
            max_response_bytes=max_response_bytes,
            token=token,
            include_revisions=True,
            include_revision_files=True,
            max_revision_tree_pages=max_revision_tree_pages,
            max_revision_weight_file_state_bytes=max_revision_weight_file_state_bytes,
            client=client,
            clock=clock,
        )
