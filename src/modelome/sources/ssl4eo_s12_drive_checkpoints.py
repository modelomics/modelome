"""Index exact first-party Google Drive checkpoint links for SSL4EO-S12."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from modelome.http import HttpClient
from modelome.models import (
    ArtifactKind,
    Identifier,
    Link,
    ModelHint,
    ReleaseHint,
    SourcePage,
    SourceRecord,
)
from modelome.normalize import content_hash

_REPOSITORY = "zhu-xlab/SSL4EO-S12"
_README_URL = f"https://raw.githubusercontent.com/{_REPOSITORY}/main/README.md"
_README_PAGE = f"https://github.com/{_REPOSITORY}"
_DRIVE_PREFIX = "https://drive.google.com/file/d/"

# Exact full-checkpoint Google Drive IDs in the project's first-party README.
# Hub-hosted MAE rows and backbone-only conversion files are intentionally omitted.
_CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "1OrtPfG2wkO05bimstQ_T9Dza8z3zp8i-": (
        "moco-rn50-s2l1c-13band",
        "SSL4EO-S12 MoCo ResNet-50, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1Tx07L6OilkfcgE2HWiSXHRmRepCPdn6V": (
        "moco-vits16-s2l1c-13band",
        "SSL4EO-S12 MoCo ViT-S/16, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1iSHHp_cudPjZlshqWXVZj5TK74P32a2q": (
        "dino-rn50-s2l1c-13band",
        "SSL4EO-S12 DINO ResNet-50, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1CseO5vvMReGlAulm5o4ZgbjUgj8VlAH7": (
        "dino-vits16-s2l1c-13band",
        "SSL4EO-S12 DINO ViT-S/16, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1QTBKl1asxgQCNd6bO2azXZNPfoQ3Sazv": (
        "mae-vits16-s2l1c-13band",
        "SSL4EO-S12 MAE ViT-S/16, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1VbIGBwzZYndv4v1vx9FiD6IP-YwsHEns": (
        "data2vec-vits16-s2l1c-13band",
        "SSL4EO-S12 Data2vec ViT-S/16, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1iWLm7ljQ6tKZiVp47pJUPDe3Un0BUd9o": (
        "moco-rn18-s2l1c-13band",
        "SSL4EO-S12 MoCo ResNet-18, Sentinel-2 L1C 13-band",
        "full ckpt",
    ),
    "1HfgXS5VpQA39k8mFrWMbHvYwuT_j6Mbi": (
        "moco-rn18-s2l1c-rgb-ep99",
        "SSL4EO-S12 MoCo ResNet-18, Sentinel-2 L1C RGB",
        "full ckpt",
    ),
    "1U_m39Owahk15Vg1uL1MYbPAmAyUWBKfI": (
        "moco-rn18-s2l1c-rgb-ep200",
        "SSL4EO-S12 MoCo ResNet-18, Sentinel-2 L1C RGB, 200-epoch checkpoint",
        "full ckpt ep200",
    ),
    "1UEpA9sOcA47W0cmwQhkSeXfQxrL-EcJB": (
        "moco-rn50-s2l1c-rgb",
        "SSL4EO-S12 MoCo ResNet-50, Sentinel-2 L1C RGB",
        "full ckpt",
    ),
    "1gjTTWikf1qORJyFifWD1ksk9HzezqQ0b": (
        "moco-rn50-s1-sar-2band",
        "SSL4EO-S12 MoCo ResNet-50, Sentinel-1 SAR 2-band",
        "full ckpt",
    ),
}
_ID_RE = re.compile(r"https://drive\.google\.com/file/d/([A-Za-z0-9_-]+)/view")


class SSL4EOS12DriveCheckpointSourceAdapter:
    """Record author-listed Drive full checkpoints absent from the Hub rows."""

    disable_derived_extraction = True
    coverage_limitation = (
        "Covers the 11 exact full-checkpoint Google Drive links listed by the "
        "SSL4EO-S12 authors. Hub-hosted MAE checkpoints, backbone-only files, "
        "logs, and unrelated dataset links are excluded. File bytes are not fetched."
    )

    def __init__(
        self,
        *,
        name: str = "ssl4eo-s12-drive-checkpoints",
        url: str = _README_URL,
        max_response_bytes: int = 2 * 1024 * 1024,
        client: Any | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("source name must be non-empty text")
        if url != _README_URL:
            raise ValueError(f"README URL must be {_README_URL}")
        if isinstance(max_response_bytes, bool) or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.name = name.strip()
        self.url = _README_URL
        self.max_response_bytes = max_response_bytes
        self.client = client or HttpClient(max_response_bytes=max_response_bytes)
        self.checkpoint_signature = content_hash(
            {
                "adapter": "ssl4eo-s12-drive-checkpoints-v1",
                "repository": _REPOSITORY,
                "readme_url": self.url,
                "file_ids": sorted(_CHECKPOINTS),
                "max_response_bytes": max_response_bytes,
            }
        )

    def fetch_page(self, state: Mapping[str, Any]) -> SourcePage:
        response = self.client.get(self.url, headers={"Accept": "text/plain"})
        if response.status != 200:
            raise ValueError(f"{self.name}: README returned HTTP {response.status}")
        if len(response.body) > self.max_response_bytes:
            raise ValueError(f"{self.name}: README exceeds configured byte limit")
        readme = response.body.decode("utf-8")
        observed = set(_ID_RE.findall(readme)) & set(_CHECKPOINTS)
        if observed != set(_CHECKPOINTS):
            missing = sorted(set(_CHECKPOINTS) - observed)
            raise ValueError(
                f"{self.name}: first-party README is missing known checkpoint links: {missing}"
            )
        digest = hashlib.sha256(response.body).hexdigest()
        if digest == state.get("completed_readme_sha256"):
            return SourcePage((), dict(state), True, upstream_count=len(_CHECKPOINTS))
        records = tuple(
            _record(self.name, file_id, model_id, title, label, digest)
            for file_id, (model_id, title, label) in _CHECKPOINTS.items()
        )
        return SourcePage(
            records,
            {"completed_readme_sha256": digest, "checkpoint_count": len(records)},
            True,
            upstream_count=len(records),
        )


def _record(
    source: str,
    file_id: str,
    model_id: str,
    title: str,
    label: str,
    readme_sha256: str,
) -> SourceRecord:
    drive_url = f"{_DRIVE_PREFIX}{file_id}/view?usp=sharing"
    model_identifier = Identifier("ssl4eo-s12:model", model_id)
    checkpoint_identifier = Identifier("ssl4eo-s12:drive-file", file_id)
    return SourceRecord(
        source_record_id=f"{source}:{model_id}",
        kind=ArtifactKind.WEIGHTS,
        canonical_url=drive_url,
        title=title,
        raw={
            "repository": _REPOSITORY,
            "google_drive_file_id": file_id,
            "README_download_label": label,
            "README_sha256": readme_sha256,
        },
        identifiers=(checkpoint_identifier, model_identifier),
        links=(
            Link(
                drive_url,
                relation="weights",
                locator=f"first-party README {label} cell",
                crawl=False,
                model_local_ids=(model_id,),
            ),
            Link(
                _README_PAGE,
                relation="documentation",
                locator="README pre-trained models table",
                crawl=False,
                model_local_ids=(model_id,),
            ),
        ),
        models=(
            ModelHint(
                local_id=model_id,
                name=title,
                identifiers=(model_identifier,),
                locator="SSL4EO-S12 pretrained models table",
            ),
        ),
        releases=(
            ReleaseHint(
                local_id=f"{model_id}@drive-{file_id}",
                model_local_id=model_id,
                identifiers=(checkpoint_identifier,),
                metadata={"drive_file_id": file_id, "download_label": label},
                locator="first-party README model row and full-checkpoint link",
            ),
        ),
    )
