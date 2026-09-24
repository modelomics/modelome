"""ONNX Model Zoo's first-party Hugging Face model inventory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class OnnxModelZooHubSourceAdapter(HuggingFaceSourceAdapter):
    """Enumerate public model repositories owned by the ONNX Model Zoo org.

    The ONNX repository README directs users to this organization for model
    access. The Hub API provides a paginated, owner-scoped inventory, including
    repository siblings so declared ONNX files are linked without downloading
    their contents. Pagination and response-size bounds come from the parent
    adapter.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        max_entries: int = 5000,
        **kwargs: Any,
    ) -> None:
        if isinstance(max_entries, bool) or int(max_entries) < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = int(max_entries)
        kwargs.setdefault("name", "onnx-model-zoo-hub")
        kwargs.setdefault(
            "url", "https://huggingface.co/api/models?author=onnxmodelzoo"
        )
        kwargs.setdefault("page_size", 100)
        super().__init__(client=client, **kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "onnx-model-zoo-hub-v1",
                "base_signature": self.checkpoint_signature,
                "max_entries": self.max_entries,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        page = super().fetch_page(state)
        prior_count = state.get("raw_items_seen", 0)
        try:
            prior_count = int(prior_count)
        except (TypeError, ValueError):
            prior_count = 0
        observed_count = prior_count + len(page.records)
        if page.upstream_count is not None and page.upstream_count > self.max_entries:
            raise ValueError(
                f"{self.name}: inventory exceeds {self.max_entries} model entries"
            )
        if observed_count > self.max_entries:
            raise ValueError(
                f"{self.name}: inventory exceeds {self.max_entries} model entries"
            )
        return page


__all__ = ["OnnxModelZooHubSourceAdapter"]
