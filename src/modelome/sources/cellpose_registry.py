"""Source-declared Cellpose checkpoint endpoints documented by MouseLand."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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

Clock = Callable[[], datetime]

# Exact source names listed by official Cellpose docs. The /models/{name}
# endpoint is explicitly documented by Cellpose as the download URL template.
_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("cytotorch_0", "Cellpose cytoplasm model"),
    ("cytotorch_1", "Cellpose cytoplasm ensemble model 2"),
    ("cytotorch_2", "Cellpose cytoplasm ensemble model 3"),
    ("cytotorch_3", "Cellpose cytoplasm ensemble model 4"),
    ("cyto2torch_0", "Cellpose cyto2 model"),
    ("cyto2torch_1", "Cellpose cyto2 ensemble model 2"),
    ("cyto2torch_2", "Cellpose cyto2 ensemble model 3"),
    ("cyto2torch_3", "Cellpose cyto2 ensemble model 4"),
    ("cyto3", "Cellpose cyto3 model"),
    ("nucleitorch_0", "Cellpose nuclei model"),
    ("nucleitorch_1", "Cellpose nuclei ensemble model 2"),
    ("nucleitorch_2", "Cellpose nuclei ensemble model 3"),
    ("nucleitorch_3", "Cellpose nuclei ensemble model 4"),
    ("denoise_cyto3", "Cellpose cyto3 denoising model"),
    ("deblur_cyto3", "Cellpose cyto3 deblurring model"),
    ("upsample_cyto3", "Cellpose cyto3 upsampling model"),
    ("oneclick_cyto3", "Cellpose cyto3 restoration model"),
    ("denoise_cyto2", "Cellpose cyto2 denoising model"),
    ("deblur_cyto2", "Cellpose cyto2 deblurring model"),
    ("upsample_cyto2", "Cellpose cyto2 upsampling model"),
    ("oneclick_cyto2", "Cellpose cyto2 restoration model"),
    ("denoise_nuclei", "Cellpose nuclei denoising model"),
    ("deblur_nuclei", "Cellpose nuclei deblurring model"),
    ("upsample_nuclei", "Cellpose nuclei upsampling model"),
    ("oneclick_nuclei", "Cellpose nuclei restoration model"),
)
_MODEL_GUIDE = "https://cellpose.readthedocs.io/en/v3.1.1.1/models.html"
_LEGACY_ENSEMBLE_GUIDE = "https://cellpose.readthedocs.io/en/stable/models.html"
_RESTORE_GUIDE = "https://cellpose.readthedocs.io/en/latest/restore.html"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class CellposeRegistrySourceAdapter:
    """Emit exact checkpoint names and download URLs declared by Cellpose docs.

    These are the legacy Cellpose website weights. Cellpose 4's built-in SAM
    and DINO checkpoints are deliberately outside this adapter because those
    are hosted on the global Hugging Face plane.
    """

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the Cellpose website checkpoint names explicitly enumerated in "
        "the Cellpose model guides and current restoration guide. It does not "
        "cover user-trained models, third-party BioImage.IO packages, or the "
        "Cellpose 4 built-ins hosted on Hugging Face."
    )

    def __init__(
        self,
        *,
        name: str = "cellpose-website-checkpoints",
        model_url_base: str = "https://www.cellpose.org/models",
        clock: Clock = _utcnow,
    ) -> None:
        if not name.strip() or not model_url_base.startswith("https://"):
            raise ValueError("name and HTTPS model_url_base are required")
        self.name = name
        self.model_url_base = model_url_base.rstrip("/")
        self.clock = clock
        self.checkpoint_signature = content_hash({
            "adapter": "cellpose-document-declared-checkpoints-v1",
            "model_url_base": self.model_url_base,
            "artifacts": _ARTIFACTS,
            "model_guide": _MODEL_GUIDE,
            "legacy_ensemble_guide": _LEGACY_ENSEMBLE_GUIDE,
            "restore_guide": _RESTORE_GUIDE,
        })

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        checked = self.clock().astimezone(UTC).isoformat().replace("+00:00", "Z")
        records = tuple(self._record(handle, title) for handle, title in _ARTIFACTS)
        return SourcePage(
            records,
            {"checked_at": checked, "model_count": len(records)},
            True,
            upstream_count=len(records),
            authoritative_snapshot=True,
        )

    def _record(self, handle: str, title: str) -> SourceRecord:
        weight_url = f"{self.model_url_base}/{handle}"
        restoration_prefixes = ("denoise_", "deblur_", "upsample_", "oneclick_")
        guide = (
            _RESTORE_GUIDE
            if handle.startswith(restoration_prefixes)
            else (
                _LEGACY_ENSEMBLE_GUIDE
                if _is_legacy_ensemble_variant(handle)
                else _MODEL_GUIDE
            )
        )
        namespace = "cellpose:checkpoint"
        model_id = f"model:{handle}"
        model = ModelHint(
            model_id,
            title,
            identifiers=(Identifier(namespace, handle),),
            aliases=(handle,),
            status=ModelStatus.RELEASED,
        )
        release = ReleaseHint(
            f"release:{handle}",
            model_id,
            identifiers=(Identifier(f"{namespace}:release", handle),),
            metadata={"checkpoint_name": handle, "weight_url": weight_url,
                      "documentation_url": guide},
        )
        return SourceRecord(
            source_record_id=f"checkpoint:{handle}",
            kind=ArtifactKind.MODEL_CARD,
            canonical_url=canonicalize_url(weight_url),
            title=title,
            raw={"checkpoint_name": handle, "weight_url": weight_url,
                 "documentation_url": guide},
            text=f"{title}; source-declared checkpoint name {handle}.",
            identifiers=(Identifier(namespace, handle),),
            links=(
                Link(guide, "model_card", crawl=False, model_local_ids=(model_id,)),
                Link(weight_url, "weights", crawl=False, model_local_ids=(model_id,)),
            ),
            models=(model,),
            releases=(release,),
        )


def _is_legacy_ensemble_variant(handle: str) -> bool:
    return handle.startswith(("cytotorch_", "cyto2torch_", "nucleitorch_")) and (
        handle.rsplit("_", 1)[-1] in {"1", "2", "3"}
    )


__all__ = ["CellposeRegistrySourceAdapter"]
