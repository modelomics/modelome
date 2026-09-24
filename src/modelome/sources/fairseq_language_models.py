"""Bounded fairseq pretrained language-model tables with first-party archives."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from modelome.http import HttpClient
from modelome.normalize import content_hash
from modelome.sources.markdown_model_table import (
    MarkdownModelTableSourceAdapter,
    _TableRow,
)

Clock = Callable[[], datetime]
_DOCUMENTS = {
    "examples/roberta/README.md": "roberta",
    "examples/bart/README.md": "bart",
    "examples/xlmr/README.md": "xlmr",
    "examples/mbart/README.md": "mbart",
    "examples/language_model/README.md": "language-model",
}
_ARCHIVE_HOST = "dl.fbaipublicfiles.com"
_CHECKPOINT_PATH = re.compile(r"\.(?:tar\.gz|tar\.bz2|pt|pth|ckpt)(?:$|/)", re.I)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class FairseqPretrainedLanguageModelSourceAdapter(MarkdownModelTableSourceAdapter):
    """Read a fixed fairseq language-model README and direct archive links.

    The path must be one of the five first-party model-zoo documents covered by
    this adapter. It retains only table rows with direct checkpoint archives on
    fairseq's public-file host. It does not crawl the repository, follow links,
    or include Hugging Face model declarations.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers released model rows with direct dl.fbaipublicfiles.com checkpoint "
        "archives in one of five fixed fairseq language-model README tables. It "
        "does not include external/Hugging Face checkpoints, models outside those "
        "documents, or download artifact bytes."
    )

    def __init__(
        self,
        *,
        name: str,
        document_path: str,
        max_response_bytes: int = 8 * 1024 * 1024,
        max_rows: int = 10_000,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        if document_path not in _DOCUMENTS:
            raise ValueError("document_path must be a configured fairseq language-model README")
        self.model_family = _DOCUMENTS[document_path]
        super().__init__(
            name=name,
            repository="facebookresearch/fairseq",
            branch="main",
            document_path=document_path,
            provider_namespace=f"fairseq:{self.model_family}-model",
            model_column=0,
            model_header_pattern=r"^Model$",
            dataset_column=(2 if document_path.endswith("language_model/README.md") else None),
            identity_include_heading=True,
            max_response_bytes=max_response_bytes,
            max_rows=max_rows,
            client=client,
            clock=clock,
        )
        self.checkpoint_signature = content_hash(
            {
                "adapter": "fairseq-pretrained-language-model-table-v1",
                "repository": self.repository,
                "branch": self.branch,
                "document_path": self.document_path,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_rows": self.max_rows,
                "artifact_host": _ARCHIVE_HOST,
                "artifact_path": _CHECKPOINT_PATH.pattern,
            }
        )

    def _rows(self, document: str) -> tuple[tuple[_TableRow, ...], int]:
        rows, skipped = super()._rows(document)
        admitted: list[_TableRow] = []
        for row in rows:
            artifacts = tuple(
                (url, relation)
                for url, relation in row.links
                if _is_first_party_checkpoint(url)
            )
            if artifacts:
                admitted.append(
                    _TableRow(
                        name=row.name,
                        dataset=row.dataset,
                        heading=row.heading,
                        description=row.description,
                        links=artifacts,
                        locator=row.locator,
                    )
                )
            else:
                skipped += 1
        return tuple(admitted), skipped


def _is_first_party_checkpoint(url: str) -> bool:
    parsed = urlsplit(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == _ARCHIVE_HOST
        and parsed.port is None
        and parsed.username is None
        and parsed.password is None
        and bool(_CHECKPOINT_PATH.search(parsed.path))
    )
