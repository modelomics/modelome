from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote

from modelome.http import HttpFailure, HttpResponse
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourceIssue,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url
from modelome.sources.huggingface import (
    Clock,
    HuggingFaceSourceAdapter,
    _is_weight_file,
    _safe_repo_filename,
    _sequence,
    _text,
    _utcnow,
)


class HuggingFaceSpacesCheckpointSourceAdapter(HuggingFaceSourceAdapter):
    """Enumerate checkpoint-format files in public Hugging Face Spaces.

    A Space is an application repository, not necessarily a model release.
    Records are therefore explicitly candidate bundles. The global listing
    SHA is checked against the detail response before file links are emitted.
    """

    coverage_limitation = (
        "Covers recognized checkpoint-format files in public Spaces returned by "
        "the Hub listing. A Space is an application repository and its files are "
        "candidate model artifacts; private/protected Spaces and other file "
        "formats are excluded. A completed cursor scan has no documented snapshot "
        "isolation guarantee."
    )

    def __init__(
        self,
        *,
        name: str = "huggingface-spaces-checkpoints",
        url: str = "https://huggingface.co/api/spaces",
        page_size: int = 10,
        max_response_bytes: int = 16 * 1024 * 1024,
        max_checkpoint_files: int = 10_000,
        token: str | None = None,
        client: Any | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        self.max_checkpoint_files = int(max_checkpoint_files)
        if self.max_checkpoint_files < 1:
            raise ValueError("max_checkpoint_files must be positive")
        super().__init__(
            name=name,
            url=url,
            artifact_kind=ArtifactKind.WEIGHTS,
            page_size=page_size,
            overlap_days=0,
            max_response_bytes=max_response_bytes,
            token=token,
            include_private=False,
            client=client,
            clock=clock,
        )

    def _listing_params(self, created_at_sweep: bool) -> Mapping[str, Any]:
        del created_at_sweep
        return {"limit": self.page_size, "full": "true"}

    def _record(self, item: Mapping[str, Any]) -> SourceRecord:
        if item.get("private") is not False:
            raise ValueError("Space result is not explicitly marked public")
        space_id = _space_id(item.get("id"))
        sha = _text(item.get("sha"))
        if not space_id or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Space result lacks a valid ID or full commit SHA")
        return SourceRecord(
            source_record_id=f"{space_id}@{sha}:detail-queue",
            kind=ArtifactKind.CATALOG_RECORD,
            canonical_url=canonicalize_url(
                f"https://huggingface.co/spaces/{quote(space_id, safe='/')}"
            ),
            title=f"{space_id} Space file inventory queue",
            raw={
                "id": space_id,
                "sha": sha,
                "space_detail_pending": True,
                "private": False,
                "lastModified": _text(item.get("lastModified"))[:128],
            },
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        queue = _sequence(state.get("space_detail_queue"))
        if not queue:
            base = super().fetch_page(state)
            queue = [
                dict(record.raw)
                for record in base.records
                if record.raw.get("space_detail_pending") is True
            ]
            if not queue:
                return base
            return SourcePage(
                records=(),
                next_state={
                    "spaces_base_state": dict(base.next_state),
                    "space_detail_queue": queue,
                    "spaces_listing_complete": base.complete,
                    "spaces_upstream_count": base.upstream_count,
                },
                complete=False,
                upstream_count=base.upstream_count,
                issues=base.issues,
            )

        item = queue[0]
        if not isinstance(item, Mapping):
            raise ValueError(f"{self.name}: invalid detail checkpoint")
        space_id, expected_sha = _space_id(item.get("id")), _text(item.get("sha"))
        if not space_id or not re.fullmatch(r"[0-9a-f]{40}", expected_sha):
            raise ValueError(f"{self.name}: invalid detail checkpoint identity")
        detail_url = f"https://huggingface.co/api/spaces/{quote(space_id, safe='/')}"
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        issues: tuple[SourceIssue, ...] = ()
        records: tuple[SourceRecord, ...] = ()
        try:
            response: HttpResponse = self.client.get(
                f"{detail_url}?expand=siblings&expand=sha", headers=headers
            )
        except HttpFailure as error:
            status_match = re.search(r"HTTP Error (401|403|404)\b", str(error))
            if status_match is None:
                raise
            issues = (
                self._issue(space_id, f"detail unavailable (HTTP {status_match.group(1)})"),
            )
        else:
            if response.status in {401, 403, 404}:
                issues = (
                    self._issue(space_id, f"detail unavailable (HTTP {response.status})"),
                )
            elif response.status != 200:
                raise ValueError(f"{self.name}: detail returned HTTP {response.status}")
            elif len(response.body) > self.max_response_bytes:
                issues = (self._issue(space_id, "detail exceeds response size limit"),)
            else:
                payload = response.json()
                if not isinstance(payload, Mapping):
                    issues = (self._issue(space_id, "detail is not a JSON object"),)
                elif _space_id(payload.get("id")) != space_id:
                    issues = (self._issue(space_id, "detail identity changed"),)
                elif _text(payload.get("sha")) != expected_sha:
                    issues = (
                        SourceIssue(
                            source_record_id=f"{space_id}@{expected_sha}",
                            stage="source_normalize",
                            error="Space detail revision changed during listing",
                            summary={
                                "listed_sha": expected_sha,
                                "detail_sha": _text(payload.get("sha")) or None,
                                "file_inventory_status": "revision_drift_incomplete",
                            },
                        ),
                    )
                elif not isinstance(payload.get("siblings"), Sequence) or isinstance(
                    payload.get("siblings"), (str, bytes, bytearray)
                ):
                    issues = (self._issue(space_id, "detail lacks file siblings"),)
                else:
                    record, incomplete = self._checkpoint_record(payload)
                    if record:
                        records = (record,)
                    if incomplete:
                        issues = (
                            self._issue(
                                space_id,
                                "checkpoint file inventory incomplete (unsafe name or file limit)",
                            ),
                        )

        remaining = [dict(value) for value in queue[1:] if isinstance(value, Mapping)]
        base_state = dict(state.get("spaces_base_state") or {})
        next_state = (
            {
                "spaces_base_state": base_state,
                "space_detail_queue": remaining,
                "spaces_listing_complete": state.get("spaces_listing_complete") is True,
                "spaces_upstream_count": state.get("spaces_upstream_count"),
            }
            if remaining
            else base_state
        )
        return SourcePage(
            records=records,
            next_state=next_state,
            complete=not remaining and state.get("spaces_listing_complete") is True,
            upstream_count=state.get("spaces_upstream_count"),
            issues=issues,
            advance_on_source_issues=bool(issues),
        )

    def _issue(self, space_id: str, error: str) -> SourceIssue:
        return SourceIssue(
            source_record_id=f"{self.name}:{space_id}",
            stage="source_normalize",
            error=error,
            summary={"space_id": space_id, "file_inventory_status": "incomplete"},
        )

    def _checkpoint_record(self, payload: Mapping[str, Any]) -> tuple[SourceRecord | None, bool]:
        space_id = _space_id(payload.get("id"))
        sha = _text(payload.get("sha"))
        if not space_id or not re.fullmatch(r"[0-9a-f]{40}", sha):
            return None, False
        siblings = _sequence(payload.get("siblings"))
        filenames = {
            _text(sibling.get("rfilename"))
            for sibling in siblings
            if isinstance(sibling, Mapping) and _text(sibling.get("rfilename"))
        }
        checkpoint_names = {
            filename
            for filename in filenames
            if _space_checkpoint_file(filename, tuple(filenames))
        }
        unsafe_names = any(not _safe_repo_filename(filename) for filename in checkpoint_names)
        files = sorted(
            filename
            for filename in checkpoint_names
            if _safe_repo_filename(filename)
        )
        truncated = len(files) > self.max_checkpoint_files
        files = files[: self.max_checkpoint_files]
        inventory_complete = not truncated and not unsafe_names
        if not files:
            return None, truncated or unsafe_names
        encoded = quote(space_id, safe="/")
        local_id = f"{space_id}#space-checkpoint-bundle"
        links = tuple(
            Link(
                canonicalize_url(
                    f"https://huggingface.co/spaces/{encoded}/resolve/{sha}/"
                    f"{quote(filename, safe='/')}"
                ),
                relation="weights",
                locator="$.siblings",
                crawl=False,
            )
            for filename in files
        )
        tags = tuple(_text(tag) for tag in _sequence(payload.get("tags")) if _text(tag))
        return SourceRecord(
            source_record_id=f"{space_id}@{sha}",
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(f"https://huggingface.co/spaces/{encoded}"),
            title=f"{space_id} Space checkpoint bundle",
            raw={
                "id": space_id,
                "sha": sha,
                "repo_type": "space",
                "weight_files": files,
                "weight_files_complete": inventory_complete,
                "tags": list(tags),
            },
            identifiers=(Identifier("huggingface:space", space_id),),
            links=links,
            models=(
                ModelHint(
                    local_id=local_id,
                    name=f"{space_id} Space checkpoint bundle",
                    identifiers=(Identifier("huggingface:space-checkpoint-candidate", space_id),),
                    status=ModelStatus.CANDIDATE,
                    locator="$.siblings",
                ),
            ),
            releases=(
                ReleaseHint(
                    local_id=f"{space_id}#space-release:{sha}",
                    model_local_id=local_id,
                    revision=sha,
                    identifiers=(Identifier("huggingface:space-revision", f"{space_id}@{sha}"),),
                    metadata={
                        "repo_type": "space",
                        "weight_files": files,
                        "weight_files_complete": inventory_complete,
                    },
                    locator="$.sha",
                ),
            ),
        ), truncated or unsafe_names


def _space_id(value: Any) -> str:
    text = _text(value)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text):
        return ""
    return text


def _space_checkpoint_file(filename: str, siblings: Sequence[str]) -> bool:
    """Include documented compound and named serialized model files."""

    basename = filename.rsplit("/", 1)[-1].casefold()
    return (
        basename.endswith(".pth.tar")
        or re.fullmatch(r"shape_predictor_[a-z0-9_-]+\.dat", basename) is not None
        or _is_weight_file(filename, siblings)
    )
