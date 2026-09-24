from __future__ import annotations

from datetime import UTC, datetime

import pytest

from modelome.sources.pfrl_pretrained_model_zoo import (
    _ATARI_ENVS,
    PfrlPretrainedModelZooAdapter,
    _inventory,
)


def test_indexes_only_benchmark_pairs_and_exact_artifact_urls() -> None:
    adapter = PfrlPretrainedModelZooAdapter(
        clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )

    page = adapter.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == len(page.records) == 530
    assert len(_ATARI_ENVS) == 59
    assert page.next_state["model_count"] == 530
    by_path = {record.raw["checkpoint_path"]: record for record in page.records}
    assert len(by_path) == len(page.records)
    assert by_path["DQN/Breakout/best.zip"].canonical_url == (
        "https://pfrl-assets.preferred.jp/DQN/Breakout/best.zip"
    )
    assert by_path["IQN/Breakout/final.zip"].raw["environment"] == (
        "BreakoutNoFrameskip-v4"
    )
    assert by_path["PPO/Swimmer-v2/final.zip"].canonical_url.endswith(
        "/PPO/Swimmer-v2/final.zip"
    )
    assert by_path["SAC/Humanoid-v2/best.zip"].canonical_url.endswith(
        "/SAC/Humanoid-v2/best.zip"
    )
    assert "PPO/Swimmer-v2/best.zip" not in by_path
    assert "C51/Breakout/best.zip" not in by_path
    assert all(record.kind.value == "weights" for record in page.records)


def test_emits_one_model_and_release_identity_per_checkpoint() -> None:
    page = PfrlPretrainedModelZooAdapter().fetch_page({})
    record = next(
        record
        for record in page.records
        if record.raw["checkpoint_path"] == "A3C/Breakout/best.zip"
    )

    assert record.identifiers[0].value == "A3C/Breakout/best.zip"
    assert record.models[0].identifiers == record.identifiers
    assert record.releases[0].model_local_id == record.models[0].local_id
    assert record.releases[0].metadata["weight_url"] == record.canonical_url
    assert record.links[-1].url == record.canonical_url
    assert all(not link.crawl for link in record.links)


def test_enforces_inventory_limit() -> None:
    assert len(_inventory()) == 530
    with pytest.raises(ValueError, match="inventory exceeds 529"):
        PfrlPretrainedModelZooAdapter(max_entries=529).fetch_page({})
    with pytest.raises(ValueError, match="positive integer"):
        PfrlPretrainedModelZooAdapter(max_entries=True)
