"""Reader for Microsoft Graphormer's literal pretrained checkpoint map."""

from __future__ import annotations

from typing import Any

from modelome.normalize import content_hash
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)


class GraphormerCheckpointRegistrySourceAdapter(
    StaticPythonCheckpointRegistrySourceAdapter
):
    """Read ``PRETRAINED_MODEL_URLS`` without importing Graphormer."""

    coverage_limitation = (
        "Covers only literal model-name-to-URL entries in Microsoft's Graphormer "
        "pretrain module at a pinned commit. It does not import code, infer model "
        "variants, or download weights. The upstream source marks its OC20 entry "
        "temporarily unavailable."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "microsoft-graphormer-pretrained-checkpoints")
        kwargs.setdefault("repository", "microsoft/Graphormer")
        kwargs.setdefault("branch", "main")
        kwargs.setdefault("source_path", "graphormer/pretrain/__init__.py")
        kwargs.setdefault("mapping_variable", "PRETRAINED_MODEL_URLS")
        kwargs.setdefault("provider_namespace", "microsoft-graphormer:checkpoint")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "microsoft-graphormer-checkpoint-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "source_path": self.source_path,
                "mapping_variable": self.mapping_variable,
                "provider_namespace": self.provider_namespace,
                "max_response_bytes": self.max_response_bytes,
                "max_entries": self.max_entries,
                "admission": "literal PRETRAINED_MODEL_URLS entries",
            }
        )
