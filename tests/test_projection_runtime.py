from pathlib import Path
from types import SimpleNamespace

import pytest

from modelome.lake import ParquetLandingZone
from modelome.model_candidate_projection import CandidateProjectionReceipt
from modelome.projection_runtime import run_semantic_scholar_projection
from modelome.semantic_scholar_materialize import ProjectionPlan, ProjectionReceipt


def receipt(tmp_path: Path, release: str, artifact: str = "a") -> ProjectionReceipt:
    return ProjectionReceipt(
        source="semantic-scholar",
        release=release,
        artifact_id=artifact * 64,
        row_count=3,
        part_count=1,
        bucket_count=2,
        path=tmp_path / release / (artifact * 64),
        already_materialized=True,
    )


def test_projection_runtime_reports_an_empty_lake_without_fabricating_a_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = SimpleNamespace(
        source="semantic-scholar",
        latest_ready_release=lambda: None,
    )
    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarProjectionMaterializer",
        lambda lake: fake,
    )

    outcome = run_semantic_scholar_projection(
        ParquetLandingZone(tmp_path / "lake"), project_candidates=False
    )

    assert outcome.status == "unavailable"
    assert outcome.release is None
    assert outcome.receipt is None


def test_projection_runtime_reopens_the_single_existing_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = receipt(tmp_path, "2026-09-01")
    plan = ProjectionPlan(
        source="semantic-scholar",
        release=existing.release,
        mode="snapshot",
        base_release=None,
        transitions=(),
        input_release_sha256=(("papers", "b" * 64),),
    )
    fake = SimpleNamespace(
        source="semantic-scholar",
        latest_ready_release=lambda: existing.release,
        plan=lambda release: plan,
        list_projections=lambda release: (existing,),
    )
    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarProjectionMaterializer",
        lambda lake: fake,
    )

    outcome = run_semantic_scholar_projection(
        ParquetLandingZone(tmp_path / "lake"), project_candidates=False
    )

    assert outcome.status == "complete"
    assert outcome.receipt is existing
    assert outcome.created is False


def test_projection_runtime_reopens_the_diff_base_after_a_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = receipt(tmp_path, "2026-08-01", "b")
    target = ProjectionReceipt(
        source="semantic-scholar",
        release="2026-09-01",
        artifact_id="c" * 64,
        row_count=3,
        part_count=1,
        bucket_count=2,
        path=tmp_path / "2026-09-01" / ("c" * 64),
        already_materialized=False,
    )
    plan = ProjectionPlan(
        source="semantic-scholar",
        release=target.release,
        mode="diff",
        base_release=base.release,
        transitions=((0, base.release, target.release),),
        input_release_sha256=(("papers", "d" * 64),),
    )
    calls = []

    class FakeMaterializer:
        source = "semantic-scholar"

        def latest_ready_release(self):
            return target.release

        def plan(self, release):
            return plan

        def list_projections(self, release):
            return ()

        def open_projection(self, release, *, artifact_id=None):
            calls.append(("open", release, artifact_id))
            return base

        def materialize(self, release, *, base=None):
            calls.append(("materialize", release, base))
            return target

    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarProjectionMaterializer",
        lambda lake: FakeMaterializer(),
    )

    outcome = run_semantic_scholar_projection(
        ParquetLandingZone(tmp_path / "lake"),
        base_artifact_id=base.artifact_id,
        project_candidates=False,
    )

    assert calls == [
        ("open", base.release, base.artifact_id),
        ("materialize", target.release, base),
    ]
    assert outcome.created is True
    assert outcome.receipt is target


def test_projection_runtime_fails_closed_on_ambiguous_target_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = receipt(tmp_path, "2026-09-01", "a")
    second = receipt(tmp_path, "2026-09-01", "b")
    plan = ProjectionPlan(
        source="semantic-scholar",
        release=first.release,
        mode="snapshot",
        base_release=None,
        transitions=(),
        input_release_sha256=(("papers", "c" * 64),),
    )
    fake = SimpleNamespace(
        source="semantic-scholar",
        latest_ready_release=lambda: first.release,
        plan=lambda release: plan,
        list_projections=lambda release: (first, second),
    )
    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarProjectionMaterializer",
        lambda lake: fake,
    )

    with pytest.raises(ValueError, match="multiple sealed projections"):
        run_semantic_scholar_projection(ParquetLandingZone(tmp_path / "lake"))


def test_projection_runtime_rejects_conflicting_artifact_selectors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="either a base or target"):
        run_semantic_scholar_projection(
            ParquetLandingZone(tmp_path / "lake"),
            base_artifact_id="a" * 64,
            target_artifact_id="b" * 64,
        )


def test_projection_runtime_publishes_candidate_assertions_from_completed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_receipt = receipt(tmp_path, "2026-09-01")
    plan = ProjectionPlan(
        source="semantic-scholar",
        release=source_receipt.release,
        mode="snapshot",
        base_release=None,
        transitions=(),
        input_release_sha256=(("papers", "b" * 64),),
    )
    materializer = SimpleNamespace(
        source="semantic-scholar",
        latest_ready_release=lambda: source_receipt.release,
        plan=lambda release: plan,
        list_projections=lambda release: (source_receipt,),
    )
    candidate = CandidateProjectionReceipt(
        source="semantic-scholar",
        release=source_receipt.release,
        artifact_id="c" * 64,
        source_projection_artifact_id=source_receipt.artifact_id,
        row_count=2,
        part_count=1,
        document_count=3,
        path=tmp_path / "candidates" / ("c" * 64),
        already_materialized=False,
    )

    class FakeCandidateProjector:
        def __init__(self, received_materializer):
            assert received_materializer is materializer

        def materialize(self, received_receipt):
            assert received_receipt is source_receipt
            return candidate

    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarProjectionMaterializer",
        lambda lake: materializer,
    )
    monkeypatch.setattr(
        "modelome.projection_runtime.SemanticScholarModelCandidateProjector",
        FakeCandidateProjector,
    )

    outcome = run_semantic_scholar_projection(ParquetLandingZone(tmp_path / "lake"))

    assert outcome.receipt is source_receipt
    assert outcome.candidate_receipt is candidate
    assert outcome.candidate_created is True
