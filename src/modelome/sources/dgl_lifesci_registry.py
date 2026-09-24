"""DGL-LifeSci-specific wrapper for first-party relative checkpoint maps."""

from __future__ import annotations

from typing import Any

from modelome.sources.graph_ml_registry import GraphMLRegistrySourceAdapter


class DglLifeSciCheckpointRegistrySourceAdapter(GraphMLRegistrySourceAdapter):
    """Read a DGL-LifeSci literal checkpoint map without importing its package."""

    coverage_limitation = (
        "Covers one configured literal checkpoint map from DGL-LifeSci's first-party "
        "pretrain source tree at a pinned commit. Values are resolved under DGL's "
        "documented data host. It does not import package code or fetch checkpoint bytes."
    )

    def __init__(self, *, mapping_variable: str = "generative_url", **kwargs: Any) -> None:
        kwargs.setdefault("repository", "awslabs/dgl-lifesci")
        kwargs.setdefault("branch", "master")
        kwargs.setdefault(
            "source_path", "python/dgllife/model/pretrain/generative_models.py"
        )
        kwargs.setdefault("provider_namespace", "dgllife:generative-checkpoint")
        kwargs.setdefault("checkpoint_base_url", "https://data.dgl.ai/")
        super().__init__(mapping_variable=mapping_variable, **kwargs)
