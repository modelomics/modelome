from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from modelome.lake import ParquetLandingZone, ReleaseReceipt, ShardReceipt
from modelome.models import SourceRecord
from modelome.storage import Database

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class BulkShardPlan:
    """Source-neutral release identity selected from one control record."""

    source: str
    dataset: str
    release: str
    shard: str
    control_sha256: str | None = None
    upstream_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in ("source", "dataset", "release", "shard"):
            object.__setattr__(self, field, _required_text(getattr(self, field), field))
        for field in ("control_sha256", "upstream_sha256"):
            value = getattr(self, field)
            if value is None:
                continue
            object.__setattr__(self, field, _sha256(value, field))


@dataclass(frozen=True, slots=True)
class BulkError:
    source_record_id: str | None
    stage: str
    error: str


@dataclass(frozen=True, slots=True)
class BulkOutcome:
    control_source: str
    selected: int
    examined: int
    represented: int
    new: int
    cached: int
    rows: int
    releases: tuple[ReleaseReceipt, ...]
    errors: tuple[BulkError, ...]
    control_complete: bool
    complete: bool
    budget_exhausted: bool


@dataclass(frozen=True, slots=True)
class _PlannedControl:
    record: SourceRecord
    plan: BulkShardPlan
    order: Any


@dataclass(frozen=True, slots=True)
class _ReceiptView:
    receipt: Any
    source: str
    dataset: str
    release: str
    shard: str
    control_sha256: str
    upstream_sha256: str
    row_count: int
    already_committed: bool

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.source, self.dataset, self.release)


class BulkControlOrchestrator:
    """Load active bulk controls under a strict transfer and publication budget."""

    def __init__(self, database: Database, lake: ParquetLandingZone) -> None:
        self.database = database
        self.lake = lake

    def run(
        self,
        *,
        control_source: str,
        loader: Callable[[SourceRecord], Any],
        selector: Callable[[SourceRecord], BulkShardPlan | None],
        order_key: Callable[[SourceRecord], Any] | None = None,
        max_new_shards: int = 1,
    ) -> BulkOutcome:
        """Load selected controls and seal only complete release groups.

        A loader call is skipped once the new-transfer budget is exhausted because
        the orchestrator cannot know whether that call is cached without invoking
        loader-specific APIs. A loader exception consumes one budget slot
        conservatively: the failure may have happened after a transfer began.
        """

        control_source = _required_text(control_source, "control_source")
        if not callable(loader):
            raise TypeError("loader must be callable")
        if not callable(selector):
            raise TypeError("selector must be callable")
        if order_key is not None and not callable(order_key):
            raise TypeError("order_key must be callable")
        if (
            isinstance(max_new_shards, bool)
            or not isinstance(max_new_shards, int)
            or max_new_shards < 0
        ):
            raise ValueError("max_new_shards must be a nonnegative integer")

        control_complete = _control_plane_complete(self.database, control_source)
        records = self.database.list_control_records(control_source)
        planned, planning_errors = _plan_controls(records, selector, order_key)
        if planning_errors:
            return _outcome(
                control_source=control_source,
                planned=planned,
                examined=0,
                receipts={},
                new=0,
                cached=0,
                rows=0,
                releases=(),
                errors=planning_errors,
                control_complete=control_complete,
                max_new_shards=max_new_shards,
            )

        conflicts = _plan_conflicts(planned)
        if conflicts:
            return _outcome(
                control_source=control_source,
                planned=planned,
                examined=0,
                receipts={},
                new=0,
                cached=0,
                rows=0,
                releases=(),
                errors=conflicts,
                control_complete=control_complete,
                max_new_shards=max_new_shards,
            )

        receipts: dict[tuple[str, str, str, str], _ReceiptView] = {}
        errors: list[BulkError] = []
        examined = 0
        new = 0
        cached = 0
        rows = 0
        budget_spent = 0

        for entry in planned:
            if budget_spent >= max_new_shards:
                break
            examined += 1
            record_id = _record_id(entry.record)
            try:
                raw_receipt = loader(entry.record)
            except Exception as error:  # loaders isolate transport/parser failures
                budget_spent += 1
                errors.append(_error(record_id, "load", error))
                break

            try:
                receipt = _receipt_view(raw_receipt)
            except (AttributeError, TypeError, ValueError) as error:
                # The loader may already have transferred data before returning an
                # invalid receipt, so this attempt must conservatively spend budget.
                budget_spent += 1
                errors.append(_error(record_id, "receipt", error))
                break

            if receipt.already_committed:
                cached += 1
            else:
                new += 1
                budget_spent += 1

            try:
                _match_receipt(entry.plan, receipt)
                key = (*receipt.group, receipt.shard)
                previous = receipts.get(key)
                if previous is not None:
                    if (
                        previous.control_sha256 != receipt.control_sha256
                        or previous.upstream_sha256 != receipt.upstream_sha256
                    ):
                        raise ValueError(
                            "one release shard resolved to conflicting control or upstream digests"
                        )
                    raise ValueError("one release shard was represented more than once")
            except ValueError as error:
                errors.append(_error(record_id, "receipt", error))
                break

            receipts[key] = receipt
            rows += receipt.row_count

        releases: list[ReleaseReceipt] = []
        if not errors and control_complete:
            groups = _groups(planned)
            for group in sorted(groups):
                entries = groups[group]
                group_receipts = {
                    entry.plan.shard: receipts.get((*group, entry.plan.shard)) for entry in entries
                }
                if any(receipt is None for receipt in group_receipts.values()):
                    continue
                expected_shards = {
                    shard: _seal_selector(receipt)
                    for shard, receipt in group_receipts.items()
                    if receipt is not None
                }
                try:
                    release = self.lake.seal_release(
                        source=group[0],
                        dataset=group[1],
                        release=group[2],
                        expected_shards=expected_shards,
                    )
                    _validate_release_receipt(release, group, group_receipts)
                except Exception as error:
                    errors.append(_error(None, "seal", error))
                    break
                releases.append(release)

        return _outcome(
            control_source=control_source,
            planned=planned,
            examined=examined,
            receipts=receipts,
            new=new,
            cached=cached,
            rows=rows,
            releases=tuple(releases),
            errors=errors,
            control_complete=control_complete,
            max_new_shards=max_new_shards,
        )


def _plan_controls(
    records: list[SourceRecord],
    selector: Callable[[SourceRecord], BulkShardPlan | None],
    order_key: Callable[[SourceRecord], Any] | None,
) -> tuple[list[_PlannedControl], list[BulkError]]:
    planned: list[_PlannedControl] = []
    errors: list[BulkError] = []
    for record in records:
        record_id = _record_id(record)
        try:
            plan = selector(record)
            if plan is None:
                continue
            if not isinstance(plan, BulkShardPlan):
                raise TypeError("selector must return BulkShardPlan or None")
            order = order_key(record) if order_key is not None else record_id
            planned.append(_PlannedControl(record=record, plan=plan, order=order))
        except Exception as error:
            errors.append(_error(record_id, "select", error))
    if errors:
        return planned, errors
    try:
        planned.sort(key=lambda entry: (entry.order, _record_id(entry.record)))
    except (TypeError, ValueError) as error:
        errors.append(_error(None, "order", error))
    return planned, errors


def _plan_conflicts(planned: list[_PlannedControl]) -> list[BulkError]:
    seen: dict[tuple[str, str, str, str], _PlannedControl] = {}
    errors: list[BulkError] = []
    for entry in planned:
        key = (
            entry.plan.source,
            entry.plan.dataset,
            entry.plan.release,
            entry.plan.shard,
        )
        previous = seen.get(key)
        if previous is None:
            seen[key] = entry
            continue
        conflict = (
            previous.plan.control_sha256 != entry.plan.control_sha256
            or previous.plan.upstream_sha256 != entry.plan.upstream_sha256
        )
        detail = " with conflicting digests" if conflict else ""
        errors.append(
            BulkError(
                source_record_id=_record_id(entry.record),
                stage="select",
                error=f"duplicate selected release shard{detail}",
            )
        )
    return errors


def _control_plane_complete(database: Database, source: str) -> bool:
    matches = [status for status in database.source_status() if status["source"] == source]
    if len(matches) != 1:
        return False
    status = matches[0]
    if status.get("complete") is not True:
        return False
    last_run = status.get("last_run")
    return not isinstance(last_run, Mapping) or last_run.get("status") == "complete"


def _receipt_view(receipt: Any) -> _ReceiptView:
    already_committed = receipt.already_committed
    if not isinstance(already_committed, bool):
        raise TypeError("receipt already_committed must be boolean")
    row_count = receipt.row_count
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError("receipt row_count must be a nonnegative integer")
    return _ReceiptView(
        receipt=receipt,
        source=_required_text(receipt.source, "receipt source"),
        dataset=_required_text(receipt.dataset, "receipt dataset"),
        release=_required_text(receipt.release, "receipt release"),
        shard=_required_text(receipt.shard, "receipt shard"),
        control_sha256=_sha256(receipt.control_sha256, "receipt control_sha256"),
        upstream_sha256=_sha256(receipt.upstream_sha256, "receipt upstream_sha256"),
        row_count=row_count,
        already_committed=already_committed,
    )


def _match_receipt(plan: BulkShardPlan, receipt: _ReceiptView) -> None:
    identity = (receipt.source, receipt.dataset, receipt.release, receipt.shard)
    expected = (plan.source, plan.dataset, plan.release, plan.shard)
    if identity != expected:
        raise ValueError("loader receipt identity does not match its selected control")
    if plan.control_sha256 and receipt.control_sha256 != plan.control_sha256:
        raise ValueError("loader receipt control digest does not match selected control")
    if plan.upstream_sha256 and receipt.upstream_sha256 != plan.upstream_sha256:
        raise ValueError("loader receipt upstream digest does not match selected control")


def _groups(
    planned: list[_PlannedControl],
) -> dict[tuple[str, str, str], list[_PlannedControl]]:
    groups: dict[tuple[str, str, str], list[_PlannedControl]] = {}
    for entry in planned:
        key = (entry.plan.source, entry.plan.dataset, entry.plan.release)
        groups.setdefault(key, []).append(entry)
    return groups


def _seal_selector(receipt: _ReceiptView) -> str | ShardReceipt:
    if isinstance(receipt.receipt, ShardReceipt):
        return receipt.receipt
    return receipt.upstream_sha256


def _validate_release_receipt(
    receipt: ReleaseReceipt,
    group: tuple[str, str, str],
    shards: Mapping[str, _ReceiptView | None],
) -> None:
    identity = (
        _required_text(receipt.source, "release receipt source"),
        _required_text(receipt.dataset, "release receipt dataset"),
        _required_text(receipt.release, "release receipt release"),
    )
    if identity != group:
        raise ValueError("release receipt identity does not match the requested group")
    shard_count = receipt.shard_count
    if shard_count != len(shards):
        raise ValueError("release receipt shard count does not match selected controls")
    row_count = receipt.row_count
    expected_rows = sum(shard.row_count for shard in shards.values() if shard is not None)
    if row_count != expected_rows:
        raise ValueError("release receipt row count does not match selected controls")


def _outcome(
    *,
    control_source: str,
    planned: list[_PlannedControl],
    examined: int,
    receipts: Mapping[tuple[str, str, str, str], _ReceiptView],
    new: int,
    cached: int,
    rows: int,
    releases: tuple[ReleaseReceipt, ...],
    errors: list[BulkError],
    control_complete: bool,
    max_new_shards: int,
) -> BulkOutcome:
    represented = len(receipts)
    complete = control_complete and not errors and represented == len(planned)
    return BulkOutcome(
        control_source=control_source,
        selected=len(planned),
        examined=examined,
        represented=represented,
        new=new,
        cached=cached,
        rows=rows,
        releases=releases,
        errors=tuple(errors),
        control_complete=control_complete,
        complete=complete,
        budget_exhausted=(not errors and represented < len(planned) and new >= max_new_shards),
    )


def _error(record_id: str | None, stage: str, error: Exception) -> BulkError:
    return BulkError(
        source_record_id=record_id,
        stage=stage,
        error=f"{type(error).__name__}: {error}",
    )


def _record_id(record: SourceRecord) -> str:
    return _required_text(record.source_record_id, "source_record_id")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    digest = _required_text(value, field).casefold()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a SHA-256 digest")
    return digest


__all__ = [
    "BulkControlOrchestrator",
    "BulkError",
    "BulkOutcome",
    "BulkShardPlan",
]
