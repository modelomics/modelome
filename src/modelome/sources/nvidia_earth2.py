from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from modelome.http import HttpClient, HttpResponse
from modelome.models import Link, SourceIssue, SourcePage, SourceRecord
from modelome.sources.huggingface import HuggingFaceSourceAdapter

Clock = Callable[[], datetime]
_COLLECTION_URL = "https://huggingface.co/api/collections/nvidia/earth-2"
_COLLECTION_PAGE = "https://huggingface.co/collections/nvidia/earth-2"
_MODEL_API_URL = "https://huggingface.co/api/models"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class NvidiaEarth2SourceAdapter(HuggingFaceSourceAdapter):
    """Project the model entries in NVIDIA's public Earth-2 Hub collection.

    Collection membership is maintained by NVIDIA. The Hub model API supplies
    each exact repo revision and file inventory; the inherited projector emits
    weight links pinned to that revision. The default catalog enables this
    curated collection alongside the global Hub source.
    """

    def __init__(
        self,
        *,
        page_size: int = 20,
        max_response_bytes: int = 16 * 1024 * 1024,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        super().__init__(
            name="nvidia-earth2",
            url=_MODEL_API_URL,
            page_size=page_size,
            max_response_bytes=max_response_bytes,
            client=client,
            clock=clock,
        )
        self.collection_url = _COLLECTION_URL
        self.collection_page = _COLLECTION_PAGE

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        item_ids = state.get("collection_model_ids")
        cursor = state.get("collection_cursor", 0)
        if not isinstance(item_ids, list):
            response: HttpResponse = self.client.get(
                self.collection_url,
                headers={"Accept": "application/json"},
            )
            if response.status != 200:
                raise ValueError(
                    f"{self.name}: collection returned HTTP {response.status}"
                )
            if len(response.body) > self.max_response_bytes:
                raise ValueError(
                    f"{self.name}: collection response exceeds "
                    f"{self.max_response_bytes} bytes"
                )
            payload = response.json()
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: collection response must be an object")
            raw_items = payload.get("items")
            if not isinstance(raw_items, list) or len(raw_items) > 1_000:
                raise ValueError(f"{self.name}: collection items are invalid or too large")
            item_ids = []
            for item in raw_items:
                if not isinstance(item, Mapping):
                    continue
                item_type = item.get("item_type") or item.get("type") or item.get("repoType")
                if item_type != "model":
                    continue
                repo_id = item.get("item_id") or item.get("id")
                if isinstance(repo_id, str) and repo_id and repo_id not in item_ids:
                    item_ids.append(repo_id)
            cursor = 0

        if not isinstance(cursor, int) or isinstance(cursor, bool) or cursor < 0:
            raise ValueError(f"{self.name}: collection cursor is invalid")
        if cursor > len(item_ids):
            raise ValueError(f"{self.name}: collection cursor is beyond the item list")

        selected = item_ids[cursor : cursor + self.page_size]
        records: list[SourceRecord] = []
        issues: list[SourceIssue] = []
        for repo_id in selected:
            if not isinstance(repo_id, str) or not repo_id:
                continue
            encoded_id = quote(repo_id, safe="/")
            response = self.client.get(
                f"{_MODEL_API_URL}/{encoded_id}",
                params={"full": "true", "cardData": "true", "config": "true"},
                headers={"Accept": "application/json"},
            )
            source_record_id = f"{self.name}:{repo_id}"
            if response.status != 200:
                issues.append(
                    SourceIssue(
                        source_record_id=source_record_id,
                        stage="model_metadata",
                        error=f"HTTP {response.status}",
                        summary={"repo_id": repo_id},
                    )
                )
                continue
            if len(response.body) > self.max_response_bytes:
                issues.append(
                    SourceIssue(
                        source_record_id=source_record_id,
                        stage="model_metadata",
                        error="response exceeds configured byte limit",
                        summary={"repo_id": repo_id},
                    )
                )
                continue
            try:
                metadata = response.json()
                if not isinstance(metadata, Mapping) or metadata.get("id") != repo_id:
                    raise ValueError("model API response identity does not match collection item")
                record = self._record(metadata)
                raw = dict(record.raw)
                raw["curated_collection"] = self.collection_page
                records.append(
                    replace(
                        record,
                        raw=raw,
                        links=record.links
                        + (
                            Link(
                                self.collection_page,
                                relation="curated_collection",
                                locator="collection.items",
                                crawl=False,
                            ),
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=source_record_id,
                        stage="model_normalize",
                        error=f"{type(error).__name__}: {error}",
                        summary={"repo_id": repo_id},
                    )
                )

        next_cursor = cursor + len(selected)
        complete = next_cursor == len(item_ids)
        next_state: dict[str, Any] = {
            "collection_model_ids": item_ids,
            "collection_cursor": next_cursor,
            "collection_url": self.collection_url,
            "collection_page": self.collection_page,
        }
        if complete:
            next_state["completed_at"] = self.clock().astimezone(UTC).isoformat()
        return SourcePage(
            records=tuple(records),
            next_state=next_state,
            complete=complete,
            upstream_count=len(item_ids),
            issues=tuple(issues),
        )
