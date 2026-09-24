from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier
from modelome.sources.chgnet_pretrained_weights import (
    CHGNetPretrainedWeightsSourceAdapter,
    _parse_checkpoint_map,
)

REVISION = "d" * 40
COMMIT_URL = "https://api.github.com/repos/CederGroupHub/chgnet/commits/main"
MODEL_URL = (
    f"https://raw.githubusercontent.com/CederGroupHub/chgnet/{REVISION}/"
    "chgnet/model/model.py"
)
TREE_URL = (
    f"https://api.github.com/repos/CederGroupHub/chgnet/git/trees/{REVISION}?recursive=1"
)
MODEL_SOURCE = '''\
checkpoint_path = {
    "0.3.0": "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
    "0.2.0": "../pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
    "r2scan": (
        "../pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar"
    ),
}.get(model_name)
'''


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(url: str, body: bytes) -> HttpResponse:
    return HttpResponse(200, {}, body, url)


def tree_body() -> bytes:
    paths = [
        "chgnet/pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar",
        "chgnet/pretrained/0.2.0/chgnet_0.2.0_e30f77s348m32.pth.tar",
        "chgnet/pretrained/r2scan/chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar",
    ]
    tree = [
        {"path": path, "type": "blob", "sha": str(index) * 40, "size": index * 1000}
        for index, path in enumerate(paths, start=1)
    ]
    return json.dumps({"truncated": False, "tree": tree}).encode()


def adapter(client: QueueClient) -> CHGNetPretrainedWeightsSourceAdapter:
    return CHGNetPretrainedWeightsSourceAdapter(
        client=client, clock=lambda: datetime(2026, 9, 24, tzinfo=UTC)
    )


def test_indexes_three_loadable_checkpoint_paths_and_git_blob_metadata() -> None:
    source_body = MODEL_SOURCE.encode()
    tree = tree_body()
    client = QueueClient(
        response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()),
        response(MODEL_URL, source_body),
        response(TREE_URL, tree),
    )

    page = adapter(client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 3
    assert client.calls == [COMMIT_URL, MODEL_URL, TREE_URL]
    records = {record.raw["model_name"]: record for record in page.records}
    record = records["r2scan"]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.raw["checkpoint_path"] == (
        "chgnet/pretrained/r2scan/"
        "chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar"
    )
    assert record.raw["checkpoint_url"] == (
        "https://raw.githubusercontent.com/CederGroupHub/chgnet/"
        f"{REVISION}/chgnet/pretrained/r2scan/"
        "chgnet_r2scan_transfer_learning_e15f36s161m23.pth.tar"
    )
    assert record.raw["size_bytes"] == 3000
    assert record.models[0].identifiers == (Identifier("chgnet:checkpoint", "r2scan"),)
    assert record.releases[0].metadata["git_blob_sha"] == "3" * 40


def test_unchanged_commit_skips_static_metadata_fetches() -> None:
    client = QueueClient(response(COMMIT_URL, f'{{"sha":"{REVISION}"}}'.encode()))

    page = adapter(client).fetch_page({"completed_revision": REVISION, "model_count": 3})

    assert page.records == ()
    assert page.upstream_count == 3
    assert client.calls == [COMMIT_URL]


@pytest.mark.parametrize(
    "source",
    [
        MODEL_SOURCE.replace('"r2scan": (', '"other": ('),
        MODEL_SOURCE.replace("../pretrained/r2scan/", "../../outside/"),
        MODEL_SOURCE.replace(".pth.tar", ".zip"),
    ],
)
def test_rejects_changed_model_names_or_unsafe_checkpoint_paths(source: str) -> None:
    with pytest.raises(ValueError):
        _parse_checkpoint_map(source, "test")


def test_live_chgnet_source_and_file_tree_smoke() -> None:
    """Read source and Git tree metadata only; never fetch model bytes."""
    try:
        page = CHGNetPretrainedWeightsSourceAdapter().fetch_page({})
    except Exception as error:  # pragma: no cover - network-dependent smoke
        pytest.skip(f"live CHGNet metadata unavailable: {error}")
    assert page.upstream_count == 3
    assert {record.raw["model_name"] for record in page.records} == {
        "0.2.0",
        "0.3.0",
        "r2scan",
    }
