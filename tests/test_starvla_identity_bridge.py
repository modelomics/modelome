from __future__ import annotations

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.models import (
    ArtifactKind,
    Identifier,
    ModelHint,
    ModelStatus,
    SourceRecord,
)
from modelome.sources.starvla_vlact_collection import StarVLAVLActCollectionAdapter


def test_starvla_collection_repo_joins_exact_huggingface_model_identity() -> None:
    repo = "StarVLA/VLAct_Qwen3PI_Robotwin_Finetune"
    starvla_record = StarVLAVLActCollectionAdapter()._record(
        repo,
        {"note": {"text": "policy checkpoint"}},
        "0123456789abcdef0123456789abcdef01234567",
        "checkpoints/policy.pt",
        "a" * 40,
        1024,
        b"collection fixture",
    )
    huggingface_record = SourceRecord(
        source_record_id=f"huggingface:model:{repo}",
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=f"https://huggingface.co/{repo}",
        title=repo,
        raw={},
        models=(
            ModelHint(
                local_id="hub-model",
                name=repo,
                identifiers=(Identifier("huggingface:model", repo),),
                status=ModelStatus.RELEASED,
            ),
        ),
    )

    result = build_entries(
        (
            source_record_to_entry_seed(starvla_record, source="starvla"),
            source_record_to_entry_seed(huggingface_record, source="huggingface"),
        )
    )

    assert len(result.entries) == 1
    assert {member.source for member in result.entries[0].members} == {
        "huggingface",
        "starvla",
    }
    assert any(
        identifier.namespace == "huggingface:model" and identifier.value == repo
        for identifier in result.entries[0].identifiers
    )
