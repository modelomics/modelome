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


def response(
    body: Any,
    *,
    headers: dict[str, str] | None = None,
    url: str = "",
    status: int = 200,
) -> HttpResponse:
    return HttpResponse(
        status=status,
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
    assert "order_by=created_at&sort=asc" in client.calls[1]
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


def test_full_release_page_without_link_header_probes_next_offset_page() -> None:
    project_page_url = (
        "https://gitlab.com/api/v4/projects?visibility=public&order_by=id"
        "&sort=asc&pagination=keyset&per_page=1"
    )
    release_page_url = "https://gitlab.com/api/v4/projects/42/releases?per_page=1&page=1"
    client = QueuedClient(
        response([project()], url=project_page_url),
        response([release()], url=release_page_url),
        response([], url=release_page_url.replace("page=1", "page=2")),
        response([], url=project_page_url + "&id_after=42"),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(
        client=client,
        page_size=1,
        max_releases_per_page=1,
    )

    project_page = adapter.fetch_page({})
    first_release_page = adapter.fetch_page(project_page.next_state)
    second_release_page = adapter.fetch_page(first_release_page.next_state)
    final_project_page = adapter.fetch_page(second_release_page.next_state)

    assert first_release_page.next_state["release_next_url"].endswith("page=2")
    assert second_release_page.complete is False
    assert final_project_page.complete is True
    assert len(client.calls) == 4


def test_full_keyset_project_page_without_link_uses_documented_id_after_fallback() -> None:
    project_page_url = (
        "https://gitlab.com/api/v4/projects?visibility=public&order_by=id"
        "&sort=asc&pagination=keyset&per_page=1"
    )
    cursor_url = project_page_url + "&id_after=42"
    client = QueuedClient(
        response([project()], url=project_page_url),
        response([], url="https://gitlab.com/api/v4/projects/42/releases?per_page=100&page=1"),
        response([], url=cursor_url),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client, page_size=1)

    project_page = adapter.fetch_page({})
    release_page = adapter.fetch_page(project_page.next_state)
    final_project_page = adapter.fetch_page(release_page.next_state)

    assert project_page.next_state["projects_next_url"] == cursor_url
    assert final_project_page.complete is True


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


def test_disappeared_project_is_skipped_and_enumeration_continues() -> None:
    next_url = "https://gitlab.com/api/v4/projects?pagination=keyset&id_after=42"
    client = QueuedClient(
        response(
            [project()],
            headers={"Link": f'<{next_url}>; rel="next"'},
        ),
        response([], status=404, url="https://gitlab.com/api/v4/projects/42/releases?per_page=100"),
        response([], url=next_url),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    project_page = adapter.fetch_page({})
    missing_project = adapter.fetch_page(project_page.next_state)
    final_project_page = adapter.fetch_page(missing_project.next_state)

    assert missing_project.records == ()
    assert missing_project.complete is False
    assert missing_project.next_state["projects_next_url"] == next_url
    assert final_project_page.complete is True
    assert len(client.calls) == 3


def test_renamed_project_keeps_the_release_url_from_the_api_response() -> None:
    renamed_release = release(
        {
            "name": "qwen2.5.safetensors",
            "url": "https://gitlab.com/lab/new-name/-/releases/v1.0/downloads/qwen2.5.safetensors",
        }
    )
    renamed_release["_links"]["self"] = "https://gitlab.com/lab/new-name/-/releases/v1.0"
    client = QueuedClient(
        response([project(path="lab/old-name")]),
        response(
            [renamed_release],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client)

    project_page = adapter.fetch_page({})
    release_page = adapter.fetch_page(project_page.next_state)

    assert release_page.records[0].links[0].url == (
        "https://gitlab.com/lab/new-name/-/releases/v1.0"
    )
    assert any(
        item.namespace == "gitlab:project" and item.value == "lab/new-name"
        for item in release_page.records[0].identifiers
    )


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
            [
                release(
                    {"name": "first.pt", "url": "https://gitlab.com/lab/qwen-model/first.pt"},
                    {"name": "second.pt", "url": "https://gitlab.com/lab/qwen-model/second.pt"},
                )
            ],
            url="https://gitlab.com/api/v4/projects/42/releases?per_page=100",
        ),
    )
    adapter = GitLabPublicReleaseAssetsSourceAdapter(client=client, max_assets_per_release=1)

    project_page = adapter.fetch_page({})
    with pytest.raises(ValueError, match="max_assets_per_release"):
        adapter.fetch_page(project_page.next_state)
