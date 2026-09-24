"""First-party MACE-OFF23 checkpoint handles and URLs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit

from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)


class MaceOff23CheckpointRegistrySourceAdapter(StaticPythonCheckpointRegistrySourceAdapter):
    """Read MACE-OFF23's exact small/medium/large URLs from MACE's literal map.

    The MACE implementation is resolved to a Git commit before the dictionary is
    parsed. The source is never imported; model URLs must stay at the direct
    ACEsuit/mace-off path matching each source-native handle.
    """

    coverage_limitation = (
        "Covers only the MACE-OFF23 small, medium, and large handles in "
        "ACEsuit/mace's literal `mace_off_urls` mapping. It does not cover other "
        "MACE foundation families, user checkpoints, or download model bytes."
    )

    def __init__(self, *, name: str = "mace-off23-checkpoints", **kwargs: Any) -> None:
        expected = {
            "repository": "ACEsuit/mace",
            "source_path": "mace/calculators/foundations_models.py",
            "mapping_variable": "mace_off_urls",
            "provider_namespace": "mace:off23-checkpoint",
        }
        for key, value in expected.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"{name}: {key} must be {value!r}")
            kwargs[key] = value
        kwargs.setdefault("branch", "develop")
        kwargs.setdefault("max_response_bytes", 4 * 1024 * 1024)
        kwargs.setdefault("max_entries", 100)
        super().__init__(name=name, **kwargs)
        self.checkpoint_signature = content_hash({
            "adapter": "mace-off23-checkpoints-v1",
            "repository": self.repository,
            "branch": self.branch,
            "source_path": self.source_path,
            "mapping_variable": self.mapping_variable,
            "provider_namespace": self.provider_namespace,
            "max_response_bytes": self.max_response_bytes,
            "max_entries": self.max_entries,
            "admission": "mace_off_urls exact ACEsuit/mace-off raw model path",
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        page = super().fetch_page(state)
        for record in page.records:
            handle = record.raw.get("checkpoint_handle")
            if not isinstance(handle, str) or handle not in {"small", "medium", "large"}:
                raise ValueError(f"{self.name}: unsupported MACE-OFF23 handle")
            expected_url = (
                "https://raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/"
                f"MACE-OFF23_{quote(handle, safe='')}.model"
            )
            parsed = urlsplit(str(record.raw.get("weight_url", "")))
            if (
                parsed.scheme != "https"
                or parsed.hostname != "raw.githubusercontent.com"
                or parsed.geturl() != expected_url
            ):
                raise ValueError(f"{self.name}: {handle} does not use its official checkpoint URL")
        return page


__all__ = ["MaceOff23CheckpointRegistrySourceAdapter"]
