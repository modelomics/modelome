from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, ModelStatus
from modelome.sources.gitlab_release_assets import GitLabPublicReleaseAssetsSourceAdapter


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, headers: dict[str, str] | None = None, **_: Any) -> HttpResponse:
        self.calls.append(url)
        return self.responses.pop(0)


def response(body: Any, *, headers: dict[str, str] | None = None, url: str = "") -> HttpResponse:
    return HttpResponse(
        status=200,
        headers=headers or {},
        body=json.dumps(body).encode(),
        url=url or "https://gitlab.com/api/v4/projects?visibility=public",
    )


def project(project_id: int = 42, path: str = "lab/qwen-model") -> dict[str, Any]:
    return {
        "id": project_id,
        "path_with_namespace": path,
        "web_url": f"https://gitlab.com/{path}",
        "visibility": "public",
    }


def release(*links: dict[str, Any]) -> dict[str, Any]:
    return {
        "tag_name": "v1.0",
        "name": "Qwen 2.5 model release",
        "description": "Pretrained model weights for Qwen2.5.",
        "released_at": "2026-09-20T12:00:00Z",
        "_links": {"self": "https://gitlab.com/lab/qwen-model/-/releases/v1.0"},
        "assets": {"links": list(links)},
    }


def test_discovers_explicit_checkpoint_links_with_low_confidence_and_no_auth() -> None:
    client = QueuedClient(
        response([project()]),
        response(
            [
                release(
                    {
                        "id": 8,
                        "name": "qwen2.5-7b.safetensors",
                        "url": "https://gitlab.com/lab/qwen-model/-/releases/v1.0/downloads/qwen2.5-7b.safetensors",
                        "link_type": "package",
                    },
                    {
                        "id": 9,
                        "name": "source archive",
                        "url": "https://gitlab.com/lab/qwen-model/-/archive/v1.0/source.zip",
                    },
                )
            ],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    projects_page = adapter.fetch_page({})
    release_page = adapter.fetch_page(projects_page.next_state)

    assert len(client.calls) == 2
    assert "pagination=keyset" in client.calls[0]
    assert all("PRIVATE-TOKEN" not in call for call in client.calls)
    assert len(release_page.records) == 1
    record = release_page.records[0]
    assert record.kind is ArtifactKind.WEIGHTS
    assert record.canonical_url.endswith("qwen2.5-7b.safetensors")
    assert record.models[0].name == "qwen2.5 7b"
    assert record.models[0].status is ModelStatus.CANDIDATE
    assert record.models[0].confidence == 0.2
    assert release_page.complete is True


def test_release_pagination_is_checkpointed_one_request_at_a_time() -> None:
    next_url = "https://gitlab.com/api/v4/projects/42/releases?pagination=keyset&id_after=1"
    client = QueuedClient(
        response([project()]),
        response(
            [release()],
            headers={"Link": f'<{next_url}>; rel="next"'},
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
        response([], url=next_url),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    project_page = adapter.fetch_page({})
    first_release_page = adapter.fetch_page(project_page.next_state)
    second_release_page = adapter.fetch_page(first_release_page.next_state)

    assert len(client.calls) == 3
    assert first_release_page.complete is False
    assert first_release_page.next_state["release_next_url"] == next_url
    assert second_release_page.complete is True


def test_generic_or_false_positive_release_links_are_not_candidates() -> None:
    client = QueuedClient(
        response([project()]),
        response(
            [
                release(
                    {
                        "name": "model.safetensors",
                        "url": "https://gitlab.com/lab/qwen-model/model.safetensors",
                    },
                    {
                        "name": "tokenizer.json",
                        "url": "https://gitlab.com/lab/qwen-model/tokenizer.json",
                    },
                    {
                        "name": "other.safetensors",
                        "url": "https://files.example.org/other.safetensors",
                    },
                )
            ],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    project_page = adapter.fetch_page({})
    release_page = adapter.fetch_page(project_page.next_state)

    assert release_page.records == ()


def test_model_identity_can_come_from_descriptive_release_link_label() -> None:
    client = QueuedClient(
        response([project()]),
        response(
            [
                release(
                    {
                        "name": "Qwen2.5 7B model weights",
                        "url": "https://gitlab.com/lab/qwen-model/downloads/weights.safetensors",
                    }
                )
            ],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    project_page = adapter.fetch_page({})
    release_page = adapter.fetch_page(project_page.next_state)

    assert release_page.records[0].models[0].name == "Qwen2.5 7B"


def test_rejects_unbounded_page_limits_and_untrusted_resume_urls() -> None:
    with pytest.raises(ValueError, match="page_size"):
        GitLabPublicReleaseAssetsSourceAdapter(page_size=101)

    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=QueuedClient())
    with pytest.raises(ValueError, match="outside the public API"):
        adapter.fetch_page(
            {"projects_started": True, "projects_next_url": "https://evil.test/api/v4/projects"}
        )


def test_release_asset_limit_fails_instead_of_silently_omitting_links() -> None:
    client = QueuedClient(
        response([project()]),
        response(
            [release({"name": "first.pt", "url": "https://gitlab.com/lab/qwen-model/first.pt"},
                     {"name": "second.pt", "url": "https://gitlab.com/lab/qwen-model/second.pt"})],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client, max_assets_per_release=1)

    project_page = adapter.fetch_page({})
    with pytest.raises(ValueError, match="max_assets_per_release"):
        adapter.fetch_page(project_page.next_state)
