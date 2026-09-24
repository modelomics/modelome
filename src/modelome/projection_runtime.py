from __future__ import annotations

from dataclasses import dataclass

from modelome.lake import ParquetLandingZone
from modelome.model_candidate_projection import (
    CandidateProjectionReceipt,
    SemanticScholarModelCandidateProjector,
)
from modelome.semantic_scholar_materialize import (
    ProjectionPlan,
    ProjectionReceipt,
    SemanticScholarProjectionMaterializer,
)


@dataclass(frozen=True, slots=True)
class ProjectionOutcome:
    """Result of publishing or reopening one corpus-scale source projection."""

    source: str
    release: str | None
    status: str
    plan: ProjectionPlan | None
    receipt: ProjectionReceipt | None
    created: bool = False
    error: str | None = None
    candidate_receipt: CandidateProjectionReceipt | None = None
    candidate_created: bool = False


def run_semantic_scholar_projection(
    lake: ParquetLandingZone,
    *,
    release: str | None = None,
    base_artifact_id: str | None = None,
    target_artifact_id: str | None = None,
    project_candidates: bool = True,
) -> ProjectionOutcome:
    """Publish the newest ready S2 state without relying on process memory.

    Ambiguous projection lineages fail closed. Operators may supply exact base or
    target artifact IDs to resolve an intentional multi-layout lake explicitly.
    """

    if not isinstance(lake, ParquetLandingZone):
        raise TypeError("lake must be a ParquetLandingZone")
    if base_artifact_id is not None and target_artifact_id is not None:
        raise ValueError("specify either a base or target artifact ID, not both")
    materializer = SemanticScholarProjectionMaterializer(lake)
    target = release.strip() if isinstance(release, str) else release
    if target == "":
        raise ValueError("release must not be empty")
    target = target or materializer.latest_ready_release()
    if target is None:
        if base_artifact_id is not None or target_artifact_id is not None:
            raise ValueError("artifact IDs require a ready target release")
        return ProjectionOutcome(
            source=materializer.source,
            release=None,
            status="unavailable",
            plan=None,
            receipt=None,
        )

    plan = materializer.plan(target)
    if target_artifact_id is not None:
        receipt = materializer.open_projection(
            target,
            artifact_id=target_artifact_id,
        )
        return _completed_outcome(
            materializer,
            plan,
            receipt,
            source_created=False,
            project_candidates=project_candidates,
        )

    if plan.mode == "snapshot" and base_artifact_id is not None:
        raise ValueError("a snapshot projection cannot specify a base artifact")

    existing = materializer.list_projections(target)
    if base_artifact_id is None:
        if len(existing) == 1:
            return _completed_outcome(
                materializer,
                plan,
                existing[0],
                source_created=False,
                project_candidates=project_candidates,
            )
        if len(existing) > 1:
            raise ValueError(
                f"release {target} has multiple sealed projections; "
                "target_artifact_id or base_artifact_id is required"
            )

    base = None
    if plan.mode == "diff":
        if plan.base_release is None:
            raise ValueError("diff projection plan is missing its base release")
        base = materializer.open_projection(
            plan.base_release,
            artifact_id=base_artifact_id,
        )
    receipt = materializer.materialize(target, base=base)
    return _completed_outcome(
        materializer,
        plan,
        receipt,
        source_created=not receipt.already_materialized,
        project_candidates=project_candidates,
    )


def _completed_outcome(
    materializer: SemanticScholarProjectionMaterializer,
    plan: ProjectionPlan,
    receipt: ProjectionReceipt,
    *,
    source_created: bool,
    project_candidates: bool,
) -> ProjectionOutcome:
    candidate_receipt = None
    if project_candidates:
        candidate_receipt = SemanticScholarModelCandidateProjector(
            materializer
        ).materialize(receipt)
    return ProjectionOutcome(
        source=materializer.source,
        release=receipt.release,
        status="complete",
        plan=plan,
        receipt=receipt,
        created=source_created,
        candidate_receipt=candidate_receipt,
        candidate_created=(
            candidate_receipt is not None
            and not candidate_receipt.already_materialized
        ),
    )


__all__ = ["ProjectionOutcome", "run_semantic_scholar_projection"]
