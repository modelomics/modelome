from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.zatom_checkpoint_registry import (
    ZatomCheckpointRegistrySourceAdapter,
    _parse_readme,
)

REVISION = "b" * 40
COMMIT_URL = "https://api.github.com/repos/Zatom-AI/zatom/commits/main"
RAW_URL = f"https://raw.githubusercontent.com/Zatom-AI/zatom/{REVISION}/README.md"
READ_ME = """\
# Zatom-1
### Checkpoints
One can download pretrained checkpoints.
wget -P checkpoints/ https://zenodo.org/records/19766997/files/zatom_1_joint_paper_weights.ckpt
wget -P checkpoints/ https://zenodo.org/records/19766997/files/platom_1_qm9_only_pretraining_paper_weights.ckpt
## Training
"""


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def adapter(client: QueueClient) -> ZatomCheckpointRegistrySourceAdapter:
    return ZatomCheckpointRegistrySourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_literal_checkpoints_and_exact_zenodo_urls() -> None:
    readme = READ_ME.encode()
    client = QueueClient(
        response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()),
        response(RAW_URL, readme),
    )

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 2
    assert client.calls == [COMMIT_URL, RAW_URL]
    record = next(
        record
        for record in page.records
        if record.raw["checkpoint_filename"] == "zatom_1_joint_paper_weights.ckpt"
    )
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.raw["checkpoint_url"] == (
        "https://zenodo.org/records/19766997/files/zatom_1_joint_paper_weights.ckpt"
    )
    assert record.models[0].identifiers == (
        Identifier("zatom:checkpoint", "zatom_1_joint_paper_weights"),
    )


def test_skips_readme_when_official_revision_is_unchanged() -> None:
    client = QueueClient(response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()))

    page = adapter(client).fetch_page({"completed_revision": REVISION, "model_count": 24})

    assert page.records == ()
    assert page.upstream_count == 24
    assert client.calls == [COMMIT_URL]


@pytest.mark.parametrize(
    "readme",
    [
        "# No registry\n## Training\n",
        READ_ME.replace("19766997", "123").replace("\n## Training", "\n## Training"),
        READ_ME.replace("zatom_1_joint_paper_weights.ckpt", "../unsafe.ckpt"),
        READ_ME.replace(
            "zatom_1_joint_paper_weights.ckpt",
            "platom_1_qm9_only_pretraining_paper_weights.ckpt",
        ),
    ],
)
def test_rejects_missing_changed_or_ambiguous_checkpoint_list(readme: str) -> None:
    with pytest.raises(ValueError):
        _parse_readme(readme, "test", 100)


def test_live_first_party_readme_metadata_smoke() -> None:
    """Smoke current GitHub metadata only; no Zenodo checkpoint bytes are read."""
    try:
        page = ZatomCheckpointRegistrySourceAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live Zatom registry unavailable: {error}")
    assert page.authoritative_snapshot
    assert page.upstream_count == 24
    assert any(
        record.raw["checkpoint_filename"] == "zatom_1_joint_paper_weights.ckpt"
        for record in page.records
    )
