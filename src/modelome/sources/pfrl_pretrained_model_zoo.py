"""Exact public checkpoint paths from PFRL's pretrained model zoo."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ModelStatus,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import canonicalize_url, content_hash

_REPOSITORY_URL = "https://github.com/pfnet/pfrl"
_PRETRAINED_SOURCE_URL = (
    "https://github.com/pfnet/pfrl/blob/master/pfrl/utils/pretrained_models.py"
)
_ASSET_ROOT = "https://pfrl-assets.preferred.jp"

# Canonical Gym ALE IDs from the complete 59-game ChainerRL/PFRL benchmark table.
# PFRL's downloader removes the terminal NoFrameskip-v4 suffix before composing
# the upstream object path, so `BreakoutNoFrameskip-v4` maps to `/Breakout/`.
_ATARI_ENVS = (
    "AirRaid", "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing",
    "Breakout", "Carnival", "Centipede", "ChopperCommand", "CrazyClimber",
    "DemonAttack", "DoubleDunk", "Enduro", "FishingDerby", "Freeway", "Frostbite",
    "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond", "JourneyEscape",
    "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "Pooyan", "PrivateEye", "Qbert",
    "Riverraid", "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris",
    "SpaceInvaders", "StarGunner", "Tennis", "TimePilot", "Tutankham", "UpNDown",
    "Venture", "VideoPinball", "WizardOfWor", "YarsRevenge", "Zaxxon",
)
_ATARI_ALGORITHMS = ("A3C", "DQN", "IQN", "Rainbow")

# The official reproducibility paper lists these environments per algorithm.
_MUJOCO_ENVS_BY_ALGORITHM = {
    "DDPG": (
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", "Reacher-v2",
        "InvertedPendulum-v2", "InvertedDoublePendulum-v2",
    ),
    "TRPO": (
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", "Swimmer-v2",
        "Humanoid-v2",
    ),
    "PPO": (
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", "Swimmer-v2",
        "Humanoid-v2",
    ),
    "TD3": (
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", "Reacher-v2",
        "InvertedPendulum-v2", "InvertedDoublePendulum-v2",
    ),
    "SAC": (
        "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Ant-v2", "Swimmer-v2",
        "Humanoid-v2",
    ),
}
_ALGORITHMS = _ATARI_ALGORITHMS + tuple(_MUJOCO_ENVS_BY_ALGORITHM)
_MODEL_TYPES = {
    "A3C": ("best", "final"),
    "DQN": ("best", "final"),
    "IQN": ("best", "final"),
    "Rainbow": ("best", "final"),
    "DDPG": ("best", "final"),
    "TRPO": ("best", "final"),
    "PPO": ("final",),
    "TD3": ("best", "final"),
    "SAC": ("best", "final"),
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _inventory() -> tuple[tuple[str, str, str], ...]:
    entries = [
        (algorithm, f"{environment}NoFrameskip-v4", model_type)
        for algorithm in _ATARI_ALGORITHMS
        for environment in _ATARI_ENVS
        for model_type in _MODEL_TYPES[algorithm]
    ]
    entries.extend(
        (algorithm, environment, model_type)
        for algorithm, environments in _MUJOCO_ENVS_BY_ALGORITHM.items()
        for environment in environments
        for model_type in _MODEL_TYPES[algorithm]
    )
    return tuple(entries)


class PfrlPretrainedModelZooAdapter:
    """Enumerate exact PFRL artifact URLs for its published benchmark inventory.

    The official loader defines the URL template and its per-algorithm allowed
    model types. Benchmark environments come from PFRL's corresponding public
    Atari and MuJoCo reproduction tables. This static, bounded adapter creates
    metadata records only; it does not download checkpoint archives.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers PFRL reproduction checkpoints for the 59 Atari games and the "
        "algorithm-specific MuJoCo tasks in its published benchmark tables. "
        "It excludes non-reproduction examples, other environments, training "
        "intermediate checkpoints, and archive contents."
    )

    def __init__(
        self,
        *,
        name: str = "pfrl-pretrained-model-zoo",
        max_entries: int = 600,
        clock: Any = _utcnow,
    ) -> None:
        if not name.strip():
            raise ValueError("source name must not be empty")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.name = name.strip()
        self.max_entries = max_entries
        self.clock = clock
        self.checkpoint_signature = content_hash(
            {
                "adapter": "pfrl-pretrained-model-zoo-v1",
                "asset_root": _ASSET_ROOT,
                "atari_algorithms": list(_ATARI_ALGORITHMS),
                "atari_environments": list(_ATARI_ENVS),
                "mujoco_environments_by_algorithm": _MUJOCO_ENVS_BY_ALGORITHM,
                "model_types": _MODEL_TYPES,
            }
        )

    @property
    def repository_url(self) -> str:
        return _REPOSITORY_URL

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        del state
        inventory = _inventory()
        if len(inventory) > self.max_entries:
            raise ValueError(
                f"{self.name}: checkpoint inventory exceeds {self.max_entries}"
            )
        checked_at = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(*entry) for entry in inventory)
        return SourcePage(
            records=records,
            next_state={
                "checked_at": checked_at,
                "inventory_sha256": content_hash(list(inventory)),
                "model_count": len(records),
            },
            complete=True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, algorithm: str, environment: str, model_type: str) -> SourceRecord:
        # Matches PFRL's downloader: replace("NoFrameskip-v4", "") before joining.
        env_path = environment.removesuffix("NoFrameskip-v4")
        checkpoint_path = f"{algorithm}/{env_path}/{model_type}.zip"
        checkpoint_url = f"{_ASSET_ROOT}/{checkpoint_path}"
        local_id = f"checkpoint:{content_hash(checkpoint_path)[:24]}"
        model_key = f"{algorithm}/{environment}"
        identifier = Identifier("pfrl:pretrained-checkpoint", checkpoint_path)
        model = ModelHint(
            local_id=local_id,
            name=f"{algorithm} {environment} ({model_type})",
            identifiers=(identifier,),
            status=ModelStatus.RELEASED,
            locator=checkpoint_path,
        )
        release = ReleaseHint(
            local_id=f"release:{content_hash(checkpoint_path)[:24]}",
            model_local_id=local_id,
            identifiers=(Identifier("pfrl:pretrained-release", checkpoint_path),),
            metadata={"checkpoint_path": checkpoint_path, "weight_url": checkpoint_url},
            locator=checkpoint_path,
        )
        return SourceRecord(
            source_record_id=local_id,
            kind=ArtifactKind.WEIGHTS,
            canonical_url=canonicalize_url(checkpoint_url),
            title=f"PFRL {algorithm} {model_key} {model_type} checkpoint",
            raw={
                "algorithm": algorithm,
                "environment": environment,
                "checkpoint_type": model_type,
                "checkpoint_path": checkpoint_path,
                "weight_url": checkpoint_url,
            },
            text=f"PFRL pretrained RL checkpoint: {checkpoint_path}",
            identifiers=(identifier,),
            links=(
                Link(_REPOSITORY_URL, "source_implementation", crawl=False,
                     model_local_ids=(local_id,)),
                Link(_PRETRAINED_SOURCE_URL, "checkpoint_url_contract", crawl=False,
                     model_local_ids=(local_id,)),
                Link(checkpoint_url, "weights", crawl=False, model_local_ids=(local_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


__all__ = ["PfrlPretrainedModelZooAdapter"]
