"""PaddleOCR's V2 PP-Structure model-download table."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from modelome.http import HttpClient
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter

Clock = Callable[[], datetime]


class PaddleOcrPPStructureModelListSourceAdapter(MarkdownModelTableSourceAdapter):
    """Read exact PP-Structure layout, table, and KIE checkpoint links.

    This is separate from PaddleOCR's PP-OCR list: the PP-Structure page
    declares direct layout-analysis, table-recognition, and key-information
    extraction artifacts that are absent from that OCR-specific inventory.
    The task/configuration column is included in identity context so repeated
    KIE architecture names for SER and RE retain distinct model identities.
    """

    def __init__(
        self,
        *,
        name: str = "paddleocr-ppstructure-model-list",
        repository: str = "PaddlePaddle/PaddleOCR",
        branch: str = "main",
        source_path: str = "docs/version2.x/ppstructure/models_list.en.md",
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
            "provider_namespace": "paddleocr:ppstructure-model",
            "model_column": 0,
            "model_header_pattern": r"^(?:Model Name|Model)$",
            "identity_context_columns": (2,),
            "max_response_bytes": max_response_bytes,
            "max_rows": max_rows,
        }
        if client is not None:
            kwargs["client"] = client
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(**kwargs)


__all__ = ["PaddleOcrPPStructureModelListSourceAdapter"]
