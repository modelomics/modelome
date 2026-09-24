from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from modelome.artifact_relations import ArtifactRelationMaterializer, ArtifactRelationReceipt


@dataclass(frozen=True, slots=True)
class ArtifactRelationOutcome:
    status: str
    source_commit: str | None
    receipt: ArtifactRelationReceipt | None
    created: bool = False
    error: str | None = None


def run_artifact_relation_projection(store_root: str | Path) -> ArtifactRelationOutcome:
    materializer = ArtifactRelationMaterializer(store_root)
    receipt = materializer.materialize()
    return ArtifactRelationOutcome(
        status="complete",
        source_commit=receipt.source_commit,
        receipt=receipt,
        created=not receipt.already_materialized,
    )


__all__ = ["ArtifactRelationOutcome", "run_artifact_relation_projection"]
