from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    store: Path
    lake: Path
    sources: Path
    frontier_limit: int = 200
    frontier_max_depth: int = 1

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            store=Path(os.environ.get("MODELOME_STORE", "data/store")),
            lake=Path(os.environ.get("MODELOME_LAKE", "data/lake")),
            sources=Path(os.environ.get("MODELOME_SOURCES", "config/sources.toml")),
            frontier_limit=_positive_int(os.environ.get("MODELOME_FRONTIER_LIMIT"), default=200),
            frontier_max_depth=_nonnegative_int(
                os.environ.get("MODELOME_FRONTIER_MAX_DEPTH"), default=1
            ),
        )


def _positive_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 1:
        raise ValueError("value must be positive")
    return parsed


def _nonnegative_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError("value must be non-negative")
    return parsed
