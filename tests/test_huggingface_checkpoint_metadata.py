from __future__ import annotations

import json
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.huggingface import HuggingFaceSourceAdapter


class _Client:
    def __init__(
        self,
        listing: list[dict[str, Any]],
        details: dict[str, tuple[int, dict[str, Any]]],
    ) -> None:
        self.listing = listing
        self.details = details
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        if params:
            payload = self.listing
            status = 200
        else:
            status, payload = self.details[url]
        return HttpResponse(
            status=status,
            headers={},
            body=json.dumps(payload).encode(),
            url=url,
        )


def test_checkpoint_metadata_enrichment_is_bounded_and_sha_verified() -> None:
    repo_id = "microsoft/resnet-18"
    sha = "65a5785d9156231087c481e0c7dd33a5ff6f7e3e"
    filename = "model.safetensors"
    detail_url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    client = _Client(
        [
            {
                "id": repo_id,
                "sha": sha,
                "siblings": [{"rfilename": filename}],
            }
        ],
        {
            detail_url: (
                200,
                {
                    "id": repo_id,
                    "sha": sha,
                    "siblings": [
                        {
                            "rfilename": filename,
                            "blobId": "9bd0d7c19b950defc60e48cf02eb1cccae8924eb",
                            "size": 46812324,
                            "lfs": {
                                "sha256": (
                                    "1cf00ee468998c23d084361b02cffadabacb5074105564c596"
                                    "b8079f77ecb126"
                                ),
                                "size": 46812324,
                            },
                        },
                        {"rfilename": "README.md", "size": 1024},
                    ],
                },
            )
        },
    )
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_checkpoint_file_metadata=True,
    )

    queued = adapter.fetch_page({})
    assert queued.records == ()
    assert queued.complete is False
    assert client.calls == [
        (
            "https://huggingface.co/api/models",
            {
                "limit": 100,
                "full": "true",
                "cardData": "true",
                "config": "true",
                "sort": "lastModified",
                "direction": -1,
            },
        )
    ]

    page = adapter.fetch_page(queued.next_state)
    assert page.complete is True
    assert page.issues == ()
    record = page.records[0]
    assert record.releases[0].metadata["weight_file_metadata"] == {
        filename: {
            "blob_id": "9bd0d7c19b950defc60e48cf02eb1cccae8924eb",
            "size_bytes": 46812324,
            "lfs_sha256": "1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126",
        }
    }
    assert record.releases[0].metadata["weight_file_metadata_complete"] is True
    assert record.releases[0].metadata["weight_files"] == [filename]
    assert client.calls[1][0] == detail_url


def test_checkpoint_metadata_sha_drift_advances_queue_and_marks_incomplete() -> None:
    sha = "a" * 40
    changed_sha = "b" * 40
    repos = ("lab/changed", "lab/next")
    details = {
        f"https://huggingface.co/api/models/{repos[0]}?blobs=true": (
            200,
            {
                "id": repos[0],
                "sha": changed_sha,
                "siblings": [
                    {"rfilename": "weights.safetensors", "size": 10, "lfs": {"sha256": "c" * 64}}
                ],
            },
        ),
        f"https://huggingface.co/api/models/{repos[1]}?blobs=true": (
            200,
            {
                "id": repos[1],
                "sha": sha,
                "siblings": [
                    {"rfilename": "weights.safetensors", "size": 20, "lfs": {"sha256": "d" * 64}}
                ],
            },
        ),
    }
    client = _Client(
        [
            {"id": repo_id, "sha": sha, "siblings": [{"rfilename": "weights.safetensors"}]}
            for repo_id in repos
        ],
        details,
    )
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_checkpoint_file_metadata=True,
    )

    queued = adapter.fetch_page({})
    drift = adapter.fetch_page(queued.next_state)
    next_page = adapter.fetch_page(drift.next_state)

    assert drift.records[0].source_record_id == repos[0]
    assert drift.records[0].releases[0].metadata["weight_file_metadata"] == {}
    assert drift.records[0].releases[0].metadata["weight_file_metadata_complete"] is False
    assert drift.advance_on_source_issues is True
    assert drift.issues[0].summary == {
        "model_id": repos[0],
        "file_metadata_status": "incomplete",
        "listed_sha": sha,
        "detail_sha": changed_sha,
    }
    assert next_page.records[0].source_record_id == repos[1]
    assert next_page.records[0].releases[0].metadata["weight_file_metadata_complete"] is True


def test_checkpoint_metadata_state_cap_is_reported_as_incomplete() -> None:
    repo_id = "lab/metadata-cap"
    sha = "e" * 40
    detail_url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    client = _Client(
        [
            {
                "id": repo_id,
                "sha": sha,
                "siblings": [{"rfilename": "model.safetensors"}],
            }
        ],
        {
            detail_url: (
                200,
                {
                    "id": repo_id,
                    "sha": sha,
                    "siblings": [
                        {
                            "rfilename": "model.safetensors",
                            "size": 100,
                            "blobId": "a" * 40,
                            "lfs": {"sha256": "b" * 64},
                        }
                    ],
                },
            )
        },
    )
    adapter = HuggingFaceSourceAdapter(
        client=client,
        include_checkpoint_file_metadata=True,
        max_revision_weight_file_state_bytes=1,
    )

    queued = adapter.fetch_page({})
    page = adapter.fetch_page(queued.next_state)

    assert page.records[0].releases[0].metadata["weight_file_metadata"] == {}
    assert page.records[0].releases[0].metadata["weight_file_metadata_complete"] is False
    assert page.issues[0].summary["file_metadata_status"] == "incomplete"
