"""PaddleNLP's literal BERT transformer checkpoint map."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from modelome.http import HttpClient
from modelome.sources.paddlenlp_ernie_registry import PaddleNlpErnieRegistrySourceAdapter

Clock = Callable[[], datetime]


class PaddleNlpBertRegistrySourceAdapter(PaddleNlpErnieRegistrySourceAdapter):
    """Read BERT and UER-RoBERTa IDs paired with exact PaddleNLP checkpoints."""

    def __init__(
        self,
        *,
        name: str = "paddlenlp-bert-pretrained-registry",
        repository: str = "PaddlePaddle/PaddleNLP",
        branch: str = "develop",
        source_path: str = "paddlenlp/transformers/bert/configuration.py",
        provider_namespace: str = "paddlenlp:transformer-model",
        max_response_bytes: int = 4 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "name": name,
            "repository": repository,
            "branch": branch,
            "source_path": source_path,
            "provider_namespace": provider_namespace,
            "max_response_bytes": max_response_bytes,
        }
        if client is not None:
            kwargs["client"] = client
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(**kwargs)


__all__ = ["PaddleNlpBertRegistrySourceAdapter"]
