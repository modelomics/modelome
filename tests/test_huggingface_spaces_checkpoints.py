from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from modelome.http import HttpFailure, HttpResponse
from modelome.models import ModelStatus
from modelome.sources.huggingface_spaces_checkpoints import (
    HuggingFaceSpacesCheckpointSourceAdapter,
)


class _Client:
    def __init__(self, routes: dict[str, tuple[int, object, dict[str, str]]]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        self.calls.append((url, dict(params or {})))
        if params:
            url = f"{url}?listing=1"
        status, payload, response_headers = self.routes[url]
        return HttpResponse(status, response_headers, json.dumps(payload).encode(), url)


def _space(repo: str, sha: str) -> dict[str, object]:
    return {
        "id": repo,
        "sha": sha,
        "private": False,
        "lastModified": "2026-09-01T00:00:00Z",
    }


def test_global_public_listing_and_sha_matched_detail_emit_pinned_file_links() -> None:
    repo, sha = "liblinear/new-checkpoints-5000", "a" * 40
    listing = "https://huggingface.co/api/spaces?listing=1"
    detail = f"https://huggingface.co/api/spaces/{repo}?expand=siblings&expand=sha"
    client = _Client(
        {
            listing: (
                200,
                [_space(repo, sha), {**_space("org/private", sha), "private": True}],
                {},
            ),
            detail: (
                200,
                {
                    **_space(repo, sha),
                    "siblings": [
                        {"rfilename": "model.safetensors"},
                        {"rfilename": "pytorch_lora_weights.safetensors"},
                        {"rfilename": "optimizer.bin"},
                        {"rfilename": "README.md"},
                        {"rfilename": "unsafe/../bad.pt"},
                    ],
                },
                {},
            ),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client)

    queued = adapter.fetch_page({})
    assert queued.complete is False
    assert queued.next_state["space_detail_queue"] == [
        {"id": repo, "sha": sha, "space_detail_pending": True, "private": False,
         "lastModified": "2026-09-01T00:00:00Z"}
    ]
    page = adapter.fetch_page(queued.next_state)
    record = page.records[0]
    assert page.complete is True
    assert record.raw["weight_files"] == [
        "model.safetensors",
        "optimizer.bin",
        "pytorch_lora_weights.safetensors",
    ]
    assert record.models[0].status is ModelStatus.CANDIDATE
    assert record.releases[0].revision == sha
    assert {link.url for link in record.links} == {
        f"https://huggingface.co/spaces/{repo}/resolve/{sha}/model.safetensors",
        f"https://huggingface.co/spaces/{repo}/resolve/{sha}/optimizer.bin",
        f"https://huggingface.co/spaces/{repo}/resolve/{sha}/pytorch_lora_weights.safetensors",
    }
    assert all(link.crawl is False for link in record.links)
    assert client.calls[0][1] == {"limit": 10, "full": "true"}
    assert client.calls[1][0] == detail


def test_listing_cursor_and_detail_queue_both_resume() -> None:
    repo_a, repo_b = "org/model-a", "org/model-b"
    sha_b = "b" * 40
    first = "https://huggingface.co/api/spaces?listing=1"
    next_url = "https://huggingface.co/api/spaces?cursor=next-page"
    detail_a = f"https://huggingface.co/api/spaces/{repo_a}?expand=siblings&expand=sha"
    detail_b = f"https://huggingface.co/api/spaces/{repo_b}?expand=siblings&expand=sha"
    client = _Client(
        {
            first: (200, [_space(repo_a, sha_b)], {"Link": f'<{next_url}>; rel="next"'}),
            next_url: (200, [_space(repo_b, sha_b)], {}),
            detail_a: (200, {**_space(repo_a, sha_b), "siblings": [{"rfilename": "a.pt"}]}, {}),
            detail_b: (200, {**_space(repo_b, sha_b), "siblings": [{"rfilename": "b.ckpt"}]}, {}),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client, page_size=1)

    first_page = adapter.fetch_page({})
    assert first_page.next_state["spaces_base_state"]["next_url"] == next_url
    detail_a_page = adapter.fetch_page(first_page.next_state)
    assert detail_a_page.records[0].raw["id"] == repo_a
    listing_tail = adapter.fetch_page(detail_a_page.next_state)
    assert listing_tail.next_state["space_detail_queue"][0]["id"] == repo_b
    resumed = HuggingFaceSpacesCheckpointSourceAdapter(client=client, page_size=1)
    final = resumed.fetch_page(listing_tail.next_state)
    assert final.complete is True
    assert final.records[0].raw["id"] == repo_b


def test_sha_mismatch_and_file_cap_surface_completeness_issues() -> None:
    repo, sha = "org/checkpoints", "c" * 40
    listing = "https://huggingface.co/api/spaces?listing=1"
    detail = f"https://huggingface.co/api/spaces/{repo}?expand=siblings&expand=sha"
    client = _Client(
        {
            listing: (200, [_space(repo, sha)], {}),
            detail: (
                200,
                {
                    **_space(repo, "d" * 40),
                    "siblings": [{"rfilename": "one.pt"}],
                },
                {},
            ),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client)
    mismatch = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert mismatch.issues[0].summary["file_inventory_status"] == "revision_drift_incomplete"

    client.routes[detail] = (
        200,
        {
            **_space(repo, sha),
            "siblings": [{"rfilename": "one.pt"}, {"rfilename": "two.safetensors"}],
        },
        {},
    )
    limited = HuggingFaceSpacesCheckpointSourceAdapter(client=client, max_checkpoint_files=1)
    page = limited.fetch_page(limited.fetch_page({}).next_state)
    assert page.records[0].raw["weight_files_complete"] is False
    assert page.issues[0].summary["file_inventory_status"] == "incomplete"


def test_listing_requires_explicit_public_visibility() -> None:
    sha = "e" * 40
    client = _Client(
        {
            "https://huggingface.co/api/spaces?listing=1": (
                200,
                [{"id": "org/unspecified", "sha": sha}],
                {},
            )
        }
    )
    page = HuggingFaceSpacesCheckpointSourceAdapter(client=client).fetch_page({})
    assert page.records == ()
    assert page.issues[0].error.endswith("not explicitly marked public")


def test_http_failure_on_unavailable_detail_advances_with_issue() -> None:
    repo, sha = "org/unavailable", "f" * 40
    next_repo, next_sha = "org/available", "2" * 40
    listing = "https://huggingface.co/api/spaces?listing=1"
    next_detail = (
        f"https://huggingface.co/api/spaces/{next_repo}?expand=siblings&expand=sha"
    )

    class _UnavailableClient(_Client):
        def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
            if "api/spaces/org/unavailable?" in url:
                raise HttpFailure(f"GET {url} failed: HTTP Error 404: Not Found")
            return super().get(url, params=params, headers=headers)

    client = _UnavailableClient(
        {
            listing: (200, [_space(repo, sha), _space(next_repo, next_sha)], {}),
            next_detail: (
                200,
                {**_space(next_repo, next_sha), "siblings": [{"rfilename": "model.pt"}]},
                {},
            ),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client)
    queued = adapter.fetch_page({})
    page = adapter.fetch_page(queued.next_state)
    assert page.records == ()
    assert page.complete is False
    assert page.advance_on_source_issues is True
    assert "HTTP 404" in page.issues[0].error
    assert page.next_state["space_detail_queue"][0]["id"] == next_repo
    following = adapter.fetch_page(page.next_state)
    assert following.complete is True
    assert following.records[0].raw["id"] == next_repo


def test_unsafe_checkpoint_filename_marks_inventory_incomplete() -> None:
    repo, sha = "org/unsafe", "1" * 40
    listing = "https://huggingface.co/api/spaces?listing=1"
    detail = f"https://huggingface.co/api/spaces/{repo}?expand=siblings&expand=sha"
    client = _Client(
        {
            listing: (200, [_space(repo, sha)], {}),
            detail: (
                200,
                {
                    **_space(repo, sha),
                    "siblings": [
                        {"rfilename": "valid.pt"},
                        {"rfilename": "unsafe/../bad.pt"},
                    ],
                },
                {},
            ),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client)
    page = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert page.records[0].raw["weight_files"] == ["valid.pt"]
    assert page.records[0].raw["weight_files_complete"] is False
    assert page.issues[0].summary["file_inventory_status"] == "incomplete"


def test_sadtalker_pth_tar_checkpoint_files_are_recognized() -> None:
    # Hugging Face's public Space trees list `facevid2vid_00189-model.pth.tar`
    # and `mapping_00229-model.pth.tar` under KkLabs/SadTalker/checkpoints.
    repo, sha = "KkLabs/SadTalker", "3" * 40
    listing = "https://huggingface.co/api/spaces?listing=1"
    detail = f"https://huggingface.co/api/spaces/{repo}?expand=siblings&expand=sha"
    client = _Client(
        {
            listing: (200, [_space(repo, sha)], {}),
            detail: (
                200,
                {
                    **_space(repo, sha),
                    "siblings": [
                        {"rfilename": "checkpoints/facevid2vid_00189-model.pth.tar"},
                        {"rfilename": "checkpoints/mapping_00229-model.pth.tar"},
                        {"rfilename": "README.md"},
                    ],
                },
                {},
            ),
        }
    )
    adapter = HuggingFaceSpacesCheckpointSourceAdapter(client=client)
    page = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert page.records[0].raw["weight_files"] == [
        "checkpoints/facevid2vid_00189-model.pth.tar",
        "checkpoints/mapping_00229-model.pth.tar",
    ]


def test_disabled_proposal_has_constructor_fields() -> None:
    proposal = tomllib.loads(
        Path("config/proposals/huggingface_spaces_checkpoints.toml").read_text()
    )["source"][0]
    assert proposal["adapter"] == "huggingface_spaces_checkpoints"
    assert proposal["enabled"] is False
    assert proposal["url"] == "https://huggingface.co/api/spaces"
    assert proposal["page_size"] == 10
    assert proposal["max_checkpoint_files"] == 10000
