from __future__ import annotations

from datetime import UTC, datetime

from modelome.models import ModelStatus
from modelome.sources.cellpose_registry import CellposeRegistrySourceAdapter


def test_emits_only_documented_cellpose_checkpoint_names_and_explicit_urls() -> None:
    source = CellposeRegistrySourceAdapter(
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 25
    by_id = {record.source_record_id: record for record in page.records}
    assert "checkpoint:cyto3" in by_id
    assert "checkpoint:denoise_nuclei" in by_id
    for family in ("cytotorch", "cyto2torch", "nucleitorch"):
        for ensemble_member in range(4):
            checkpoint = f"{family}_{ensemble_member}"
            assert f"checkpoint:{checkpoint}" in by_id
            assert by_id[f"checkpoint:{checkpoint}"].releases[0].metadata["weight_url"] == (
                f"https://www.cellpose.org/models/{checkpoint}"
            )
    assert by_id["checkpoint:cytotorch_1"].raw["documentation_url"] == (
        "https://cellpose.readthedocs.io/en/stable/models.html"
    )
    assert "checkpoint:cpsam" not in by_id
    cyto3 = by_id["checkpoint:cyto3"]
    assert cyto3.models[0].identifiers[0].value == "cyto3"
    assert cyto3.models[0].status is ModelStatus.RELEASED
    assert cyto3.releases[0].metadata["weight_url"] == (
        "https://www.cellpose.org/models/cyto3"
    )
    denoise = by_id["checkpoint:denoise_nuclei"]
    assert denoise.raw["documentation_url"].endswith("/restore.html")


def test_custom_explicit_model_url_base_is_reflected_without_suffix_guessing() -> None:
    page = CellposeRegistrySourceAdapter(
        model_url_base="https://models.example.org/cellpose"
    ).fetch_page({})
    cyto3 = next(record for record in page.records if record.source_record_id == "checkpoint:cyto3")
    assert cyto3.releases[0].metadata["weight_url"] == (
        "https://models.example.org/cellpose/cyto3"
    )
