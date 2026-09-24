from __future__ import annotations

import ipaddress
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

from modelome.http import HttpClient, HttpResponse
from modelome.models import ArtifactKind, Identifier, SourceIssue, SourcePage, SourceRecord
from modelome.normalize import content_hash

Clock = Callable[[], datetime]

_DEFAULT_DATASETS = ("papers", "abstracts", "paper-ids")
_LATEST = "latest"
_SNAPSHOT_STAGE = "snapshot"
_DIFF_STAGE = "diff"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SemanticScholarDatasetSourceAdapter:
    """Enumerate complete Semantic Scholar dataset releases and shard manifests.

    This is deliberately a control-plane adapter. It emits one record for the
    pinned release and one record for every full-snapshot or incremental-diff
    shard. It never downloads a shard. Semantic Scholar returns temporary,
    pre-signed object URLs; records retain a stable query-free object selector
    and the API manifest location so a downloader can acquire a fresh URL when
    it is ready to transfer the file.
    """

    def __init__(
        self,
        *,
        name: str = "semantic-scholar-datasets",
        url: str = "https://api.semanticscholar.org/datasets/v1",
        datasets: Sequence[str] = _DEFAULT_DATASETS,
        release_id: str = _LATEST,
        page_size: int = 100,
        prefer_diffs: bool = True,
        api_key: str | None = None,
        client: HttpClient | Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.name = _required_text(name, "source name")
        self.url = _api_base_url(url, self.name)
        self.datasets = _dataset_names(datasets, self.name)
        self.release_id = _release_selector(release_id, self.name)
        self.page_size = int(page_size)
        if self.page_size < 1:
            raise ValueError(f"{self.name}: page size must be positive")
        self.prefer_diffs = bool(prefer_diffs)
        self._api_key = _required_text(api_key, "Semantic Scholar API key")
        self.client = client or HttpClient()
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "semantic-scholar-datasets-v1",
                "url": self.url,
                "datasets": list(self.datasets),
                "release_id": self.release_id,
                "page_size": self.page_size,
                "prefer_diffs": self.prefer_diffs,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        stage = _text(state.get("stage"))
        if not stage:
            return self._begin_scan(state)
        if stage == _SNAPSHOT_STAGE:
            return self._snapshot_page(state)
        if stage == _DIFF_STAGE:
            return self._diff_page(state)
        raise ValueError(f"{self.name}: unknown checkpoint stage {stage!r}")

    def list_releases(self) -> tuple[str, ...]:
        """Return every valid release ID in chronological order."""

        releases, issues = self._release_catalog()
        if issues:
            raise ValueError(f"{self.name}: release catalog contained malformed entries")
        return releases

    def resolve_download_url(self, record: SourceRecord) -> str:
        """Reacquire one temporary shard URL from its authenticated manifest.

        Control records intentionally omit pre-signed query values. This method
        validates the complete pinned manifest again and returns the matching URL
        only to the immediate caller; it never mutates or persists the record.
        """

        if not isinstance(record, SourceRecord):
            raise TypeError("shard control record must be a SourceRecord")
        raw = record.raw
        if raw.get("record_type") != "dataset_shard":
            raise ValueError(f"{self.name}: record is not a dataset shard control record")
        dataset = _required_text(raw.get("dataset"), "control dataset")
        if dataset not in self.datasets:
            raise ValueError(f"{self.name}: control record dataset is not configured")
        target_release = _required_release_id(
            raw.get("target_release"), "control target_release", self.name
        )
        stable_url = _required_text(raw.get("stable_object_url"), "stable object URL")
        if record.canonical_url != stable_url:
            raise ValueError(f"{self.name}: control record canonical URL does not match shard")
        operation = _required_text(raw.get("operation"), "control operation")
        expected_signature = _required_text(
            raw.get("manifest_signature"), "control manifest signature"
        )
        manifest_url = _required_text(raw.get("manifest_url"), "control manifest URL")

        matches: list[_DownloadTarget]
        if operation == "snapshot":
            if manifest_url != self._dataset_url(target_release, dataset):
                raise ValueError(f"{self.name}: snapshot manifest URL is not canonical")
            payload = self._get_json(manifest_url)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: dataset manifest must be a JSON object")
            targets, metadata, issues = self._snapshot_manifest(
                payload,
                dataset=dataset,
                release_id=target_release,
                manifest_url=manifest_url,
            )
            signature = _snapshot_signature(target_release, dataset, metadata, targets)
            matches = [target for target in targets if target.stable_url == stable_url]
        elif operation in {"upsert", "delete"}:
            base_release = _required_release_id(
                raw.get("base_release"), "control base_release", self.name
            )
            from_release = _required_release_id(
                raw.get("from_release"), "control from_release", self.name
            )
            to_release = _required_release_id(
                raw.get("to_release"), "control to_release", self.name
            )
            if manifest_url != self._diff_url(base_release, target_release, dataset):
                raise ValueError(f"{self.name}: diff manifest URL is not canonical")
            payload = self._get_json(manifest_url)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{self.name}: diff manifest must be a JSON object")
            targets, issues = self._diff_manifest(
                payload,
                dataset=dataset,
                base_release=base_release,
                target_release=target_release,
                manifest_url=manifest_url,
            )
            signature = _diff_signature(dataset, base_release, target_release, targets)
            matches = [
                target.download
                for target in targets
                if target.download.stable_url == stable_url
                and target.operation == operation
                and target.from_release == from_release
                and target.to_release == to_release
            ]
        else:
            raise ValueError(f"{self.name}: unsupported shard operation")

        if issues:
            raise ValueError(f"{self.name}: refreshed shard manifest is malformed")
        if signature != expected_signature:
            raise ValueError(f"{self.name}: refreshed shard manifest changed")
        if len(matches) != 1:
            raise ValueError(
                f"{self.name}: stable shard selector did not resolve exactly once"
            )
        return matches[0].temporary_url

    def _begin_scan(self, state: Mapping[str, Any]) -> SourcePage:
        releases, issues = self._release_catalog()
        if not releases:
            raise ValueError(f"{self.name}: release catalog is empty")
        target = releases[-1] if self.release_id == _LATEST else self.release_id
        if target not in releases:
            raise ValueError(f"{self.name}: configured release {target!r} is unavailable")

        prior_release = _optional_release_id(state.get("watermark"), "watermark", self.name)
        if prior_release is not None and prior_release not in releases:
            raise ValueError(
                f"{self.name}: checkpoint release {prior_release!r} is unavailable upstream"
            )
        if prior_release is not None and prior_release > target:
            raise ValueError(
                f"{self.name}: refusing to move checkpoint backward from "
                f"{prior_release} to {target}"
            )

        release = self._release_metadata(target)
        summaries, release_issues = self._required_dataset_summaries(release, target)
        issues.extend(release_issues)
        retry_state = _completed_boundary(state)
        now = _isoformat(self.clock())

        if prior_release == target:
            next_state = {"watermark": target, "completed_at": now}
            return SourcePage(
                records=(),
                next_state=next_state,
                complete=not issues,
                upstream_count=0,
                issues=tuple(issues),
                retry_state=retry_state,
            )

        stage = _DIFF_STAGE if prior_release is not None and self.prefer_diffs else _SNAPSHOT_STAGE
        release_signature = content_hash(
            {
                "release_id": target,
                "README": _optional_text(release.get("README"), self.name, "release README"),
                "datasets": summaries,
            }
        )
        next_state: dict[str, Any] = {
            "stage": stage,
            "target_release": target,
            "dataset_index": 0,
            "shard_offset": 0,
            "control_records_seen": 1,
            "release_signature": release_signature,
            "started_at": now,
        }
        if prior_release is not None:
            next_state["base_release"] = prior_release
            next_state["watermark"] = prior_release

        record = self._release_record(
            release=release,
            summaries=summaries,
            releases=releases,
            target=target,
            base_release=prior_release,
            stage=stage,
            signature=release_signature,
        )
        return SourcePage(
            records=(record,),
            next_state=next_state,
            complete=False,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _snapshot_page(self, state: Mapping[str, Any]) -> SourcePage:
        scan = self._scan_boundary(state, _SNAPSHOT_STAGE)
        dataset = self.datasets[scan.dataset_index]
        manifest_url = self._dataset_url(scan.target_release, dataset)
        payload = self._get_json(manifest_url)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: {dataset} manifest must be a JSON object")

        targets, metadata, issues = self._snapshot_manifest(
            payload,
            dataset=dataset,
            release_id=scan.target_release,
            manifest_url=manifest_url,
        )
        signature = _snapshot_signature(scan.target_release, dataset, metadata, targets)
        drift_issue, retry_state = self._manifest_continuity(
            state,
            scan=scan,
            manifest_signature=signature,
            item_count=len(targets),
        )
        if drift_issue is not None:
            issues.append(drift_issue)

        page_targets = targets[scan.shard_offset : scan.shard_offset + self.page_size]
        records = tuple(
            self._shard_record(
                target=target,
                target_release=scan.target_release,
                dataset=dataset,
                manifest_url=manifest_url,
                manifest_signature=signature,
                manifest_index=scan.shard_offset + index,
                operation="snapshot",
                metadata=metadata,
            )
            for index, target in enumerate(page_targets)
        )
        return self._advance(
            state,
            scan=scan,
            manifest_signature=signature,
            manifest_count=len(targets),
            emitted=len(records),
            records=records,
            issues=issues,
            retry_state=retry_state,
        )

    def _diff_page(self, state: Mapping[str, Any]) -> SourcePage:
        scan = self._scan_boundary(state, _DIFF_STAGE)
        if scan.base_release is None:
            raise ValueError(f"{self.name}: incremental checkpoint is missing base_release")
        dataset = self.datasets[scan.dataset_index]
        manifest_url = self._diff_url(scan.base_release, scan.target_release, dataset)
        payload = self._get_json(manifest_url)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: {dataset} diff manifest must be a JSON object")

        targets, issues = self._diff_manifest(
            payload,
            dataset=dataset,
            base_release=scan.base_release,
            target_release=scan.target_release,
            manifest_url=manifest_url,
        )
        signature = _diff_signature(
            dataset,
            scan.base_release,
            scan.target_release,
            targets,
        )
        drift_issue, retry_state = self._manifest_continuity(
            state,
            scan=scan,
            manifest_signature=signature,
            item_count=len(targets),
        )
        if drift_issue is not None:
            issues.append(drift_issue)

        page_targets = targets[scan.shard_offset : scan.shard_offset + self.page_size]
        records = tuple(
            self._shard_record(
                target=target.download,
                target_release=scan.target_release,
                dataset=dataset,
                manifest_url=manifest_url,
                manifest_signature=signature,
                manifest_index=scan.shard_offset + index,
                operation=target.operation,
                from_release=target.from_release,
                to_release=target.to_release,
                diff_index=target.diff_index,
                operation_index=target.operation_index,
                base_release=scan.base_release,
            )
            for index, target in enumerate(page_targets)
        )
        return self._advance(
            state,
            scan=scan,
            manifest_signature=signature,
            manifest_count=len(targets),
            emitted=len(records),
            records=records,
            issues=issues,
            retry_state=retry_state,
        )

    def _advance(
        self,
        state: Mapping[str, Any],
        *,
        scan: _ScanBoundary,
        manifest_signature: str,
        manifest_count: int,
        emitted: int,
        records: tuple[SourceRecord, ...],
        issues: list[SourceIssue],
        retry_state: Mapping[str, Any],
    ) -> SourcePage:
        next_offset = scan.shard_offset + emitted
        records_seen = scan.control_records_seen + emitted
        if next_offset < manifest_count:
            next_state = dict(state)
            next_state.update(
                {
                    "shard_offset": next_offset,
                    "manifest_count": manifest_count,
                    "manifest_signature": manifest_signature,
                    "control_records_seen": records_seen,
                }
            )
            complete = False
            upstream_count = None
        elif scan.dataset_index + 1 < len(self.datasets):
            next_state = dict(state)
            next_state.update(
                {
                    "dataset_index": scan.dataset_index + 1,
                    "shard_offset": 0,
                    "control_records_seen": records_seen,
                }
            )
            next_state.pop("manifest_count", None)
            next_state.pop("manifest_signature", None)
            complete = False
            upstream_count = None
        else:
            next_state = {
                "watermark": scan.target_release,
                "completed_at": _isoformat(self.clock()),
            }
            complete = True
            upstream_count = records_seen

        return SourcePage(
            records=records,
            next_state=next_state,
            complete=complete,
            upstream_count=upstream_count,
            issues=tuple(issues),
            retry_state=retry_state,
        )

    def _scan_boundary(self, state: Mapping[str, Any], stage: str) -> _ScanBoundary:
        target = _required_release_id(state.get("target_release"), "target_release", self.name)
        dataset_index = _state_integer(state, "dataset_index", self.name)
        if dataset_index >= len(self.datasets):
            raise ValueError(f"{self.name}: dataset_index is outside configured datasets")
        shard_offset = _state_integer(state, "shard_offset", self.name)
        control_records_seen = _state_integer(
            state,
            "control_records_seen",
            self.name,
            default=1,
        )
        if _text(state.get("stage")) != stage:
            raise ValueError(f"{self.name}: checkpoint stage changed during scan")
        release_signature = _required_text(
            state.get("release_signature"), "release signature"
        )
        base_release = _optional_release_id(state.get("base_release"), "base_release", self.name)
        if stage == _SNAPSHOT_STAGE and base_release is not None:
            raise ValueError(f"{self.name}: snapshot checkpoint unexpectedly has base_release")
        if base_release is not None and base_release >= target:
            raise ValueError(f"{self.name}: base_release must precede target_release")
        return _ScanBoundary(
            stage=stage,
            target_release=target,
            base_release=base_release,
            dataset_index=dataset_index,
            shard_offset=shard_offset,
            control_records_seen=control_records_seen,
            release_signature=release_signature,
        )

    def _manifest_continuity(
        self,
        state: Mapping[str, Any],
        *,
        scan: _ScanBoundary,
        manifest_signature: str,
        item_count: int,
    ) -> tuple[SourceIssue | None, Mapping[str, Any]]:
        retry_state = dict(state)
        expected_signature = _text(state.get("manifest_signature"))
        expected_count = _optional_state_integer(state.get("manifest_count"), self.name)
        if scan.shard_offset > item_count:
            issue = self._manifest_issue(
                scan,
                f"checkpoint offset {scan.shard_offset} exceeds manifest count {item_count}",
                item_count=item_count,
            )
            return issue, _reset_manifest_boundary(state)
        if expected_count is not None and expected_count != item_count:
            issue = self._manifest_issue(
                scan,
                f"manifest count changed from {expected_count} to {item_count}",
                item_count=item_count,
            )
            return issue, _reset_manifest_boundary(state)
        if expected_signature and expected_signature != manifest_signature:
            issue = self._manifest_issue(
                scan,
                "pinned manifest contents changed while resuming",
                item_count=item_count,
            )
            return issue, _reset_manifest_boundary(state)
        return None, retry_state

    def _release_catalog(self) -> tuple[tuple[str, ...], list[SourceIssue]]:
        url = f"{self.url}/release/"
        payload = self._get_json(url)
        if not _is_sequence(payload):
            raise ValueError(f"{self.name}: release catalog must be a JSON array")
        releases: list[str] = []
        issues: list[SourceIssue] = []
        seen: set[str] = set()
        for index, value in enumerate(payload):
            try:
                release_id = _required_release_id(value, f"release[{index}]", self.name)
                if release_id in seen:
                    raise ValueError("duplicate release ID")
                seen.add(release_id)
                releases.append(release_id)
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            f"{self.name}:release:{index}:"
                            f"{content_hash(repr(value))[:24]}"
                        ),
                        stage="source_manifest",
                        error=f"{type(error).__name__}: {error}",
                        summary={"manifest": "release catalog", "index": index},
                    )
                )
        return tuple(sorted(releases)), issues

    def _release_metadata(self, target: str) -> Mapping[str, Any]:
        url = self._release_url(target)
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{self.name}: release metadata must be a JSON object")
        release_id = _required_release_id(payload.get("release_id"), "release_id", self.name)
        if release_id != target:
            raise ValueError(
                f"{self.name}: requested release {target}, received metadata for {release_id}"
            )
        _optional_text(payload.get("README"), self.name, "release README")
        if not _is_sequence(payload.get("datasets")):
            raise ValueError(f"{self.name}: release.datasets must be an array")
        return payload

    def _required_dataset_summaries(
        self,
        release: Mapping[str, Any],
        target: str,
    ) -> tuple[list[dict[str, str]], list[SourceIssue]]:
        by_name: dict[str, dict[str, str]] = {}
        issues: list[SourceIssue] = []
        for index, raw in enumerate(release["datasets"]):
            try:
                if not isinstance(raw, Mapping):
                    raise TypeError("dataset summary is not a JSON object")
                name = _required_text(raw.get("name"), "dataset name")
                description = _optional_text(
                    raw.get("description"), self.name, f"dataset {name} description"
                )
                readme = _optional_text(
                    raw.get("README"), self.name, f"dataset {name} README"
                )
                if name in by_name:
                    raise ValueError(f"duplicate dataset summary {name!r}")
                by_name[name] = {
                    "name": name,
                    "description": description,
                    "README": readme,
                }
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=f"{self.name}:{target}:dataset-summary:{index}",
                        stage="source_manifest",
                        error=f"{type(error).__name__}: {error}",
                        summary={"release_id": target, "index": index},
                    )
                )
        missing = [name for name in self.datasets if name not in by_name]
        if missing:
            raise ValueError(
                f"{self.name}: release {target} is missing required dataset(s): "
                f"{', '.join(missing)}"
            )
        return [by_name[name] for name in self.datasets], issues

    def _snapshot_manifest(
        self,
        payload: Mapping[str, Any],
        *,
        dataset: str,
        release_id: str,
        manifest_url: str,
    ) -> tuple[list[_DownloadTarget], dict[str, str], list[SourceIssue]]:
        actual_name = _required_text(payload.get("name"), "dataset manifest name")
        if actual_name != dataset:
            raise ValueError(
                f"{self.name}: requested dataset {dataset!r}, received {actual_name!r}"
            )
        metadata = {
            "name": actual_name,
            "description": _optional_text(
                payload.get("description"), self.name, f"dataset {dataset} description"
            ),
            "README": _optional_text(
                payload.get("README"), self.name, f"dataset {dataset} README"
            ),
        }
        files = payload.get("files")
        if not _is_sequence(files):
            raise ValueError(f"{self.name}: {dataset}.files must be an array")
        if not files:
            raise ValueError(f"{self.name}: {dataset}.files must not be empty")
        targets, issues = self._download_targets(
            files,
            manifest_url=manifest_url,
            issue_prefix=f"{self.name}:{release_id}:{dataset}:snapshot",
        )
        return targets, metadata, issues

    def _diff_manifest(
        self,
        payload: Mapping[str, Any],
        *,
        dataset: str,
        base_release: str,
        target_release: str,
        manifest_url: str,
    ) -> tuple[list[_DiffTarget], list[SourceIssue]]:
        actual_dataset = _required_text(payload.get("dataset"), "diff dataset")
        actual_start = _required_release_id(
            payload.get("start_release"), "diff start_release", self.name
        )
        actual_end = _required_release_id(
            payload.get("end_release"), "diff end_release", self.name
        )
        if (actual_dataset, actual_start, actual_end) != (
            dataset,
            base_release,
            target_release,
        ):
            raise ValueError(
                f"{self.name}: diff manifest identity does not match the pinned request"
            )
        diffs = payload.get("diffs")
        if not _is_sequence(diffs):
            raise ValueError(f"{self.name}: diff manifest.diffs must be an array")
        if not diffs:
            raise ValueError(
                f"{self.name}: diff from {base_release} to {target_release} is empty"
            )

        flattened: list[_DiffTarget] = []
        issues: list[SourceIssue] = []
        expected_from = base_release
        seen_urls: set[str] = set()
        for diff_index, raw in enumerate(diffs):
            if not isinstance(raw, Mapping):
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            f"{self.name}:{base_release}:{target_release}:"
                            f"{dataset}:diff:{diff_index}"
                        ),
                        stage="source_manifest",
                        error="TypeError: diff entry is not a JSON object",
                        summary={"dataset": dataset, "diff_index": diff_index},
                    )
                )
                continue
            try:
                from_release = _required_release_id(
                    raw.get("from_release"), "diff from_release", self.name
                )
                to_release = _required_release_id(
                    raw.get("to_release"), "diff to_release", self.name
                )
                if from_release != expected_from:
                    raise ValueError(
                        f"diff chain expected {expected_from}, received {from_release}"
                    )
                if to_release <= from_release:
                    raise ValueError("diff to_release must follow from_release")
                for field, operation in (
                    ("update_files", "upsert"),
                    ("delete_files", "delete"),
                ):
                    raw_files = raw.get(field)
                    if not _is_sequence(raw_files):
                        raise TypeError(f"{field} must be an array")
                    targets, target_issues = self._download_targets(
                        raw_files,
                        manifest_url=manifest_url,
                        issue_prefix=(
                            f"{self.name}:{from_release}:{to_release}:{dataset}:{field}"
                        ),
                        seen_urls=seen_urls,
                    )
                    issues.extend(target_issues)
                    flattened.extend(
                        _DiffTarget(
                            download=target,
                            operation=operation,
                            from_release=from_release,
                            to_release=to_release,
                            diff_index=diff_index,
                            operation_index=index,
                        )
                        for index, target in enumerate(targets)
                    )
                expected_from = to_release
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=(
                            f"{self.name}:{base_release}:{target_release}:"
                            f"{dataset}:diff:{diff_index}"
                        ),
                        stage="source_manifest",
                        error=f"{type(error).__name__}: {error}",
                        summary={"dataset": dataset, "diff_index": diff_index},
                    )
                )
        if expected_from != target_release:
            issues.append(
                SourceIssue(
                    source_record_id=(
                        f"{self.name}:{base_release}:{target_release}:{dataset}:diff-chain"
                    ),
                    stage="source_manifest",
                    error=(
                        f"ValueError: diff chain ended at {expected_from}, "
                        f"expected {target_release}"
                    ),
                    summary={
                        "dataset": dataset,
                        "start_release": base_release,
                        "end_release": target_release,
                    },
                )
            )
        return flattened, issues

    def _download_targets(
        self,
        values: Sequence[Any],
        *,
        manifest_url: str,
        issue_prefix: str,
        seen_urls: set[str] | None = None,
    ) -> tuple[list[_DownloadTarget], list[SourceIssue]]:
        targets: list[_DownloadTarget] = []
        issues: list[SourceIssue] = []
        seen = seen_urls if seen_urls is not None else set()
        for index, raw in enumerate(values):
            try:
                target = _download_target(raw, api_key=self._api_key)
                if target.stable_url in seen:
                    raise ValueError("duplicate shard object URL")
                seen.add(target.stable_url)
                targets.append(target)
            except (TypeError, ValueError) as error:
                issues.append(
                    SourceIssue(
                        source_record_id=f"{issue_prefix}:{index}",
                        stage="source_manifest",
                        error=f"{type(error).__name__}: {error}",
                        summary={
                            "manifest_url": manifest_url,
                            "index": index,
                            "value_digest": content_hash(repr(raw)),
                        },
                    )
                )
        return targets, issues

    def _release_record(
        self,
        *,
        release: Mapping[str, Any],
        summaries: list[dict[str, str]],
        releases: tuple[str, ...],
        target: str,
        base_release: str | None,
        stage: str,
        signature: str,
    ) -> SourceRecord:
        source_record_id = f"semantic-scholar:release:{target}"
        raw: dict[str, Any] = {
            "record_type": "release_manifest",
            "release_id": target,
            "release_readme": _optional_text(
                release.get("README"), self.name, "release README"
            ),
            "datasets": summaries,
            "available_releases": list(releases),
            "selection": "latest" if self.release_id == _LATEST else "configured",
            "transfer_mode": stage,
            "release_signature": signature,
        }
        if base_release is not None:
            raw["base_release"] = base_release
        return SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=self._release_url(target),
            title=f"Semantic Scholar Academic Graph release {target}",
            raw=raw,
            published_at=target,
            identifiers=(Identifier("semantic-scholar:release", target),),
        )

    def _shard_record(
        self,
        *,
        target: _DownloadTarget,
        target_release: str,
        dataset: str,
        manifest_url: str,
        manifest_signature: str,
        manifest_index: int,
        operation: str,
        metadata: Mapping[str, str] | None = None,
        from_release: str | None = None,
        to_release: str | None = None,
        diff_index: int | None = None,
        operation_index: int | None = None,
        base_release: str | None = None,
    ) -> SourceRecord:
        identity = {
            "dataset": dataset,
            "target_release": target_release,
            "operation": operation,
            "from_release": from_release,
            "to_release": to_release,
            "stable_object_url": target.stable_url,
        }
        shard_key = content_hash(identity)
        source_record_id = f"semantic-scholar:shard:{shard_key}"
        raw: dict[str, Any] = {
            "record_type": "dataset_shard",
            "dataset": dataset,
            "target_release": target_release,
            "operation": operation,
            "manifest_url": manifest_url,
            "manifest_signature": manifest_signature,
            "manifest_index": manifest_index,
            "stable_object_url": target.stable_url,
            "temporary_download_url_persisted": False,
            "download_resolution": "refetch_manifest_and_match_stable_object_url",
        }
        if target.query_keys:
            raw["temporary_url_query_keys"] = list(target.query_keys)
        if metadata is not None:
            raw["dataset_description"] = metadata["description"]
            raw["dataset_readme"] = metadata["README"]
        else:
            raw["license_metadata_record_id"] = (
                f"semantic-scholar:release:{target_release}"
            )
        if from_release is not None:
            raw.update(
                {
                    "base_release": base_release,
                    "from_release": from_release,
                    "to_release": to_release,
                    "diff_index": diff_index,
                    "operation_index": operation_index,
                }
            )
        label = "full" if operation == "snapshot" else operation
        return SourceRecord(
            source_record_id=source_record_id,
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=target.stable_url,
            title=(
                f"Semantic Scholar {dataset} {label} shard "
                f"{manifest_index} for {target_release}"
            ),
            raw=raw,
            published_at=target_release,
            modified_at=to_release or target_release,
            identifiers=(Identifier("semantic-scholar:dataset-shard", shard_key),),
        )

    def _manifest_issue(
        self,
        scan: _ScanBoundary,
        error: str,
        *,
        item_count: int,
    ) -> SourceIssue:
        dataset = self.datasets[scan.dataset_index]
        return SourceIssue(
            source_record_id=(
                f"{self.name}:{scan.target_release}:{dataset}:manifest-continuity"
            ),
            stage="source_manifest",
            error=f"ValueError: {error}",
            summary={
                "stage": scan.stage,
                "target_release": scan.target_release,
                "dataset": dataset,
                "checkpoint_offset": scan.shard_offset,
                "manifest_count": item_count,
            },
        )

    def _get_json(self, url: str) -> Any:
        headers = {"Accept": "application/json", "x-api-key": self._api_key}
        try:
            response: HttpResponse = self.client.get(url, headers=headers)
        except Exception as error:
            if self._api_key in str(error):
                raise RuntimeError(f"{self.name}: Semantic Scholar request failed") from None
            raise
        return response.json()

    def _release_url(self, release_id: str) -> str:
        return f"{self.url}/release/{quote(release_id, safe='')}"

    def _dataset_url(self, release_id: str, dataset: str) -> str:
        return f"{self._release_url(release_id)}/dataset/{quote(dataset, safe='')}"

    def _diff_url(self, start: str, end: str, dataset: str) -> str:
        return (
            f"{self.url}/diffs/{quote(start, safe='')}/to/{quote(end, safe='')}/"
            f"{quote(dataset, safe='')}"
        )


class _ScanBoundary:
    __slots__ = (
        "stage",
        "target_release",
        "base_release",
        "dataset_index",
        "shard_offset",
        "control_records_seen",
        "release_signature",
    )

    def __init__(
        self,
        *,
        stage: str,
        target_release: str,
        base_release: str | None,
        dataset_index: int,
        shard_offset: int,
        control_records_seen: int,
        release_signature: str,
    ) -> None:
        self.stage = stage
        self.target_release = target_release
        self.base_release = base_release
        self.dataset_index = dataset_index
        self.shard_offset = shard_offset
        self.control_records_seen = control_records_seen
        self.release_signature = release_signature


class _DownloadTarget:
    __slots__ = ("stable_url", "query_keys", "temporary_url")

    def __init__(
        self,
        stable_url: str,
        query_keys: tuple[str, ...],
        temporary_url: str,
    ) -> None:
        self.stable_url = stable_url
        self.query_keys = query_keys
        self.temporary_url = temporary_url


class _DiffTarget:
    __slots__ = (
        "download",
        "operation",
        "from_release",
        "to_release",
        "diff_index",
        "operation_index",
    )

    def __init__(
        self,
        *,
        download: _DownloadTarget,
        operation: str,
        from_release: str,
        to_release: str,
        diff_index: int,
        operation_index: int,
    ) -> None:
        self.download = download
        self.operation = operation
        self.from_release = from_release
        self.to_release = to_release
        self.diff_index = diff_index
        self.operation_index = operation_index


def _download_target(value: Any, *, api_key: str) -> _DownloadTarget:
    url = _required_text(value, "shard URL")
    if api_key in url:
        raise ValueError("shard URL contains the Semantic Scholar API key")
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError("shard URL must be a credential-free HTTPS URL")
    try:
        port = parts.port
    except ValueError:
        raise ValueError("shard URL has an invalid port") from None
    if port not in {None, 443}:
        raise ValueError("shard URL must use the default HTTPS port")
    if not parts.path or parts.path == "/":
        raise ValueError("shard URL is missing an object path")
    host = parts.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("shard URL host is not public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        stable_host = host
    else:
        if not address.is_global:
            raise ValueError("shard URL host is not public")
        stable_host = f"[{host}]" if address.version == 6 else host
    stable_url = urlunsplit(("https", stable_host, parts.path, "", ""))
    query_keys = tuple(
        sorted(
            {
                component.partition("=")[0]
                for component in parts.query.split("&")
                if component
            }
        )
    )
    return _DownloadTarget(stable_url, query_keys, url)


def _snapshot_signature(
    release_id: str,
    dataset: str,
    metadata: Mapping[str, str],
    targets: Sequence[_DownloadTarget],
) -> str:
    return content_hash(
        {
            "release_id": release_id,
            "dataset": dataset,
            "metadata": dict(metadata),
            "files": [target.stable_url for target in targets],
        }
    )


def _diff_signature(
    dataset: str,
    base_release: str,
    target_release: str,
    targets: Sequence[_DiffTarget],
) -> str:
    return content_hash(
        {
            "dataset": dataset,
            "base_release": base_release,
            "target_release": target_release,
            "files": [
                {
                    "from_release": target.from_release,
                    "to_release": target.to_release,
                    "operation": target.operation,
                    "url": target.download.stable_url,
                }
                for target in targets
            ],
        }
    )


def _reset_manifest_boundary(state: Mapping[str, Any]) -> dict[str, Any]:
    retry = dict(state)
    retry["shard_offset"] = 0
    retry.pop("manifest_count", None)
    retry.pop("manifest_signature", None)
    return retry


def _completed_boundary(state: Mapping[str, Any]) -> dict[str, Any]:
    retry = dict(state)
    retry.pop("completed_at", None)
    return retry


def _api_base_url(value: str, source: str) -> str:
    url = _required_text(value, "Semantic Scholar datasets API URL").rstrip("/")
    parts = urlsplit(url)
    if (
        parts.scheme.casefold() != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError(f"{source}: datasets API URL must be an HTTPS origin and path")
    host = parts.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"{source}: datasets API URL host must be public")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        if not address.is_global:
            raise ValueError(f"{source}: datasets API URL host must be public")
    return urlunsplit(("https", parts.netloc.casefold(), parts.path.rstrip("/"), "", ""))


def _dataset_names(values: Sequence[str], source: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{source}: datasets must be a sequence of names")
    names = tuple(_required_text(value, "dataset name") for value in values)
    if not names:
        raise ValueError(f"{source}: at least one dataset is required")
    if len(set(names)) != len(names):
        raise ValueError(f"{source}: dataset names must be unique")
    return names


def _release_selector(value: Any, source: str) -> str:
    selector = _required_text(value, "release selector")
    if selector == _LATEST:
        return selector
    return _required_release_id(selector, "release selector", source)


def _required_release_id(value: Any, field: str, source: str) -> str:
    text = _required_text(value, field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{source}: {field} must be an ISO calendar date") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{source}: {field} must use YYYY-MM-DD format")
    return text


def _optional_release_id(value: Any, field: str, source: str) -> str | None:
    if value is None or not _text(value):
        return None
    return _required_release_id(value, field, source)


def _state_integer(
    state: Mapping[str, Any],
    key: str,
    source: str,
    *,
    default: int = 0,
) -> int:
    value = state.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{source}: checkpoint {key} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{source}: checkpoint {key} must be a nonnegative integer"
        ) from None
    if parsed < 0 or str(parsed) != str(value):
        raise ValueError(f"{source}: checkpoint {key} must be a nonnegative integer")
    return parsed


def _optional_state_integer(value: Any, source: str) -> int | None:
    if value is None:
        return None
    return _state_integer({"value": value}, "value", source)


def _optional_text(value: Any, source: str, field: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise TypeError(f"{source}: {field} must be text")
    return value


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value.strip()


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _isoformat(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = ["SemanticScholarDatasetSourceAdapter"]
