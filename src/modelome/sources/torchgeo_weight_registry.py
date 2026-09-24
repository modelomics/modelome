"""Pinned AST ingestion of TorchGeo's source-declared pretrained weights.

TorchGeo publishes these checkpoint URLs as torchvision ``WeightsEnum``
constants in ``torchgeo/models``. This adapter parses the first-party GitHub
source archive without importing TorchGeo and never downloads checkpoint data.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from modelome.models import Identifier, SourceRecord
from modelome.normalize import content_hash
from modelome.sources.torchvision_weight_registry import (
    TorchvisionWeightRegistrySourceAdapter,
    _WeightEnum,
)


class TorchGeoWeightRegistrySourceAdapter(TorchvisionWeightRegistrySourceAdapter):
    """Enumerate literal weight enum releases declared by TorchGeo."""

    coverage_limitation = (
        "Covers literal torchvision WeightsEnum members in TorchGeo's first-party "
        "torchgeo/models source tree at one observed Git commit. It does not execute "
        "package code, infer papers or architectures, follow artifact URLs, or download "
        "checkpoint bytes. Some declared artifact URLs may point to TorchGeo-managed "
        "external hosting."
    )

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "torchgeo-pretrained-weight-registry")
        kwargs.setdefault("repository", "torchgeo/torchgeo")
        kwargs.setdefault("package_path", "torchgeo/models")
        super().__init__(**kwargs)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "torchgeo-weight-registry-v1",
                "repository": self.repository,
                "branch": self.branch,
                "package_path": self.package_path,
                "max_archive_bytes": self.max_archive_bytes,
                "max_module_bytes": self.max_module_bytes,
                "max_modules": self.max_modules,
                "admission": "literal WeightsEnum entries with literal URLs",
            }
        )

    def _record(
        self,
        weight_enum: _WeightEnum,
        revision: str,
        archive: bytes,
    ) -> SourceRecord:
        record = super()._record(weight_enum, revision, archive)
        model_identifier = Identifier("torchgeo:weight-enum", weight_enum.name)
        member_releases = tuple(
            replace(
                release,
                identifiers=(
                    Identifier(
                        "torchgeo:weight-enum-member",
                        f"{weight_enum.name}.{member.name}",
                    ),
                ),
            )
            for release, member in zip(record.releases, weight_enum.members, strict=True)
        )
        model = replace(record.models[0], identifiers=(model_identifier,))
        return replace(
            record,
            source_record_id=f"torchgeo-weight-enum:{weight_enum.name}",
            identifiers=(model_identifier,),
            text=record.text.replace("TorchVision", "TorchGeo"),
            models=(model,),
            releases=member_releases,
        )
