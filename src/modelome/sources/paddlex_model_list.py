"""PaddleX's independently maintained official model download table."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from modelome.http import HttpClient
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

Clock = Callable[[], datetime]


class PaddleXModelListSourceAdapter(MarkdownModelTableSourceAdapter):
    """Read exact artifact links from the official PaddleX model-list document.

    PaddleX's CPU/GPU model list pairs named models with explicit inference,
    training, and sometimes pretrained-model URLs. The generic Markdown table
    parser preserves those row-scoped links and this adapter scopes it to that
    first-party document.
    """

    def __init__(
        self,
        *,
        name: str = "paddlex-model-list",
        repository: str = "PaddlePaddle/PaddleX",
        branch: str = "release/3.7",
        source_path: str = "docs/support_list/models_list.en.md",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_rows: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "name": name,
            "repository": repository,
            "branch": branch,
            "document_path": source_path,
            "provider_namespace": "paddlex:model",
            "model_column": 0,
            "model_header_pattern": r"^(?:Model Name|Model)$",
            "max_response_bytes": max_response_bytes,
            "max_rows": max_rows,
        }
        if client is not None:
            kwargs["client"] = client
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(**kwargs)


__all__ = ["PaddleXModelListSourceAdapter"]
