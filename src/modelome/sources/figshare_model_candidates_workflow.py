"""Checkpointable sequential date-window workflow for Figshare candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from modelome.http import HttpClient
from modelome.models import SourcePage
from modelome.normalize import content_hash
from modelome.sources.figshare_model_candidates import (
    FigshareModelCandidatesSourceAdapter,
)

Clock = Callable[[], datetime]


class FigshareModelCandidatesWorkflow:
    """Harvest a bounded Figshare history using one-day OAI windows by default.

    Only the current window and its opaque OAI token are stored in checkpoint
    state. Completed windows advance a single date cursor, so the workflow does
    not need hundreds of source configurations or a large in-memory date queue.
    """

    coverage_limitation = (
        "Traverses [from_date, until_date) in adjacent bounded windows. OAI exposes "
        "the latest article version only. Token expiry stops traversal explicitly; "
        "the current date window must restart, while the workflow date cursor remains "
        "at that window. No file bytes are downloaded."
    )

    def __init__(
        self,
        *,
        from_date: str,
        until_date: str,
        window_days: int = 1,
        name: str = "figshare-model-candidates",
        oai_url: str = "https://api.figshare.com/v2/oai",
        api_url: str = "https://api.figshare.com/v2/articles",
        max_response_bytes: int = 8 * 1024 * 1024,
        max_versions_per_article: int = 50,
        client: HttpClient | Any | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.start = _date(from_date, "from_date")
        self.end = _date(until_date, "until_date")
        if self.start >= self.end:
            raise ValueError("until_date must be later than from_date")
        if (
            isinstance(window_days, bool)
            or not isinstance(window_days, int)
            or not 1 <= window_days <= 31
        ):
            raise ValueError("window_days must be an integer between 1 and 31")
        self.window_days = window_days
        self.name = name
        self.oai_url = oai_url
        self.api_url = api_url
        self.max_response_bytes = max_response_bytes
        self.max_versions_per_article = max_versions_per_article
        self.client = client
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "workflow": "figshare-model-candidates-daily-oai-v2",
                "start": self.start.isoformat(),
                "end": self.end.isoformat(),
                "window_days": window_days,
                "name": name,
                "oai_url": oai_url,
                "api_url": api_url,
                "max_response_bytes": max_response_bytes,
                "max_versions_per_article": max_versions_per_article,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        if state:
            if state.get("workflow_signature") != self.checkpoint_signature:
                raise ValueError(f"{self.name}: workflow checkpoint does not match configuration")
            cursor = _date(state.get("cursor"), "checkpoint cursor")
            if not self.start <= cursor <= self.end:
                raise ValueError(f"{self.name}: checkpoint cursor is outside configured range")
        else:
            cursor = self.start
        if cursor >= self.end:
            if state.get("window_state"):
                raise ValueError(f"{self.name}: completed workflow has an active-window checkpoint")
            return SourcePage(
                records=(),
                next_state=dict(state),
                complete=True,
                upstream_count=None,
                authoritative_snapshot=False,
            )

        window_end = min(cursor + timedelta(days=self.window_days), self.end)
        window_state = state.get("window_state", {}) if state else {}
        if not isinstance(window_state, Mapping):
            raise ValueError(f"{self.name}: invalid current-window checkpoint")
        restarts = _safe_count(state.get("token_window_restarts", 0)) if state else 0
        restarting = _token_near_expiry(window_state, (self.clock or _utcnow)())
        if restarting:
            # Restart the exact same window from its first page. This can replay
            # records, but never moves the date cursor past an uncompleted window.
            window_state = {}
            restarts += 1
        adapter = FigshareModelCandidatesSourceAdapter(
            name=self.name,
            oai_url=self.oai_url,
            api_url=self.api_url,
            from_date=cursor.isoformat(),
            until_date=window_end.isoformat(),
            max_response_bytes=self.max_response_bytes,
            max_versions_per_article=self.max_versions_per_article,
            client=self.client,
            clock=self.clock or _utcnow,
        )
        page = adapter.fetch_page(window_state)
        prior = _counters(state)
        current_pages = page.next_state["pages_seen"]
        current_records = page.next_state["records_seen"]
        previous_pages = _safe_count(window_state.get("pages_seen", 0))
        previous_records = _safe_count(window_state.get("records_seen", 0))
        totals = {
            "pages_seen": prior["pages_seen"] + current_pages - previous_pages,
            "records_seen": prior["records_seen"] + current_records - previous_records,
            "candidates_seen": prior["candidates_seen"] + len(page.records),
        }

        if page.complete:
            next_cursor = window_end
            complete = next_cursor >= self.end
            next_window_state: Mapping[str, Any] = {}
        else:
            next_cursor = cursor
            complete = False
            next_window_state = page.next_state
        next_state = {
            "workflow_signature": self.checkpoint_signature,
            "cursor": next_cursor.isoformat(),
            "window_state": dict(next_window_state),
            "token_window_restarts": restarts,
            **totals,
        }
        return SourcePage(
            records=page.records,
            next_state=next_state,
            complete=complete,
            upstream_count=None,
            authoritative_snapshot=False,
        )


def _date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must use YYYY-MM-DD") from error


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid inner window counters")
    return value


def _token_near_expiry(state: Mapping[str, Any], now: datetime) -> bool:
    """Restart the same window before its upstream token safety margin ends."""
    expiration = state.get("token_expires_at")
    if not state.get("resumption_token") or not isinstance(expiration, str):
        return False
    try:
        expires_at = datetime.fromisoformat(expiration.replace("Z", "+00:00"))
    except ValueError:
        return False  # Let the page adapter report the malformed checkpoint.
    if expires_at.tzinfo is None:
        return False
    now_utc = now.astimezone(UTC)
    return expires_at.astimezone(UTC) - now_utc <= timedelta(minutes=5)


def _counters(state: Mapping[str, Any]) -> dict[str, int]:
    if not state:
        return {"pages_seen": 0, "records_seen": 0, "candidates_seen": 0}
    return {
        name: _safe_count(state.get(name, 0))
        for name in ("pages_seen", "records_seen", "candidates_seen")
    }


def _utcnow() -> datetime:
    return datetime.now(UTC)
