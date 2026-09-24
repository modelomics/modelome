from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from modelome.models import SourcePage


class SourceAdapter(Protocol):
    name: str

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage: ...
