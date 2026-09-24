from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Mapping
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelStatus
from modelome.sources.tensorflow_hub_archive import TensorFlowHubArchiveSourceAdapter

_REVISION = "b" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(payload: bytes | Mapping[str, Any]) -> HttpResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return HttpResponse(200, {}, body, "https://fixtures.test/tfhub")


def _archive(files: Mapping[str, str]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as package:
        for path, document in files.items():
            package.writestr(f"tfhub.dev-{_REVISION}/{path}", document)
    return output.getvalue()


def test_archive_enumerates_exact_versioned_handles_and_availability_is_not_asserted() -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    "assets/docs/deepmind/models/biggan-256/1.md": """\
# Module deepmind/biggan-256/1 BigGAN v1 historical entry.

## Changelog

#### Version 1

* Initial release.
""",
                    "assets/docs/deepmind/models/biggan-256/2.md": """\
# Module deepmind/biggan-256/2 BigGAN image generator.

<!-- task: image-generation -->
<!-- network-architecture: biggan -->
<!-- asset-path: https://storage.googleapis.com/tfhub-modules/deepmind/biggan-256/2.tar.gz -->

## Changelog

#### Version 2

* Fixed a race condition.
""",
                    "assets/docs/deepmind/collections/biggan/1.md": "# Collection BigGAN\n",
                    "assets/docs/deepmind/models/no-version/readme.md": "# Not a release\n",
                }
            ),
        ),
    )
    adapter = TensorFlowHubArchiveSourceAdapter(client=client)

    page = adapter.fetch_page({})

    assert page.complete is True
    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert page.next_state["completed_revision"] == _REVISION
    assert page.next_state["document_count"] == 2
    assert len(client.calls) == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")
    v1, v2 = page.records
    assert v1.kind is ArtifactKind.MODEL_CARD
    assert v1.canonical_url == "https://tfhub.dev/deepmind/biggan-256/1"
    assert v1.identifiers == (
        Identifier("tensorflow-hub:model", "deepmind/biggan-256"),
        Identifier("tensorflow-hub:handle", "deepmind/biggan-256/1"),
    )
    assert v1.models[0].status is ModelStatus.DOCUMENTED
    assert v1.releases[0].version == "1"
    assert v1.releases[0].identifiers == (
        Identifier("tensorflow-hub:handle", "deepmind/biggan-256/1"),
    )
    assert v1.releases[0].metadata["availability"] == "unverified"
    assert v2.releases[0].version == "2"
    assert v2.releases[0].metadata["metadata"] == {
        "task": "image-generation",
        "network-architecture": "biggan",
        "asset_path": "https://storage.googleapis.com/tfhub-modules/deepmind/biggan-256/2.tar.gz",
    }
    assert any(
        link.url == v2.releases[0].metadata["metadata"]["asset_path"]
        and link.relation == "weights"
        and not link.crawl
        for link in v2.links
    )


def test_archive_skips_the_source_archive_when_commit_is_unchanged() -> None:
    first_client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    "assets/docs/google/models/nnlm-en-dim128/2.md": (
                        "# Module google/nnlm-en-dim128/2 Text embedding model.\n"
                    ),
                }
            )
        ),
    )
    adapter = TensorFlowHubArchiveSourceAdapter(client=first_client)
    first = adapter.fetch_page({})

    second_client = _QueuedClient(_response({"sha": _REVISION}))
    adapter.client = second_client
    second = adapter.fetch_page(first.next_state)

    assert second.records == ()
    assert second.complete is True
    assert len(second_client.calls) == 1


def test_archive_fails_closed_when_doc_handle_has_a_conflicting_publisher_or_version() -> None:
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    "assets/docs/google/models/nnlm-en-dim128/2.md": (
                        "# Module google/nnlm-en-dim128/1 Wrong version.\n"
                    ),
                }
            )
        ),
    )

    with pytest.raises(ValueError, match="conflicting module heading"):
        TensorFlowHubArchiveSourceAdapter(client=client).fetch_page({})


def test_archive_derives_exact_handle_from_path_when_card_has_no_module_heading() -> None:
    path = (
        "assets/docs/adityakane2001/models/"
        "regnety200mf_classification/lite/1.md"
    )
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    path: """\
# RegNetY-200MF Classification Lite

This archived model card uses a descriptive title instead of a TF Hub handle.
""",
                }
            )
        ),
    )

    page = TensorFlowHubArchiveSourceAdapter(client=client).fetch_page({})

    record = page.records[0]
    assert record.canonical_url == (
        "https://tfhub.dev/adityakane2001/"
        "regnety200mf_classification/lite/1"
    )
    assert record.identifiers == (
        Identifier(
            "tensorflow-hub:model",
            "adityakane2001/regnety200mf_classification/lite",
        ),
        Identifier(
            "tensorflow-hub:handle",
            "adityakane2001/regnety200mf_classification/lite/1",
        ),
    )
    assert record.models[0].name == "regnety200mf_classification/lite"
    assert record.releases[0].version == "1"
    assert record.releases[0].metadata["identity_source"] == "versioned_document_path"
    assert record.releases[0].metadata["declared_title_handle"] is None


def test_archive_uses_explicit_handle_with_models_segment_but_preserves_path_candidate() -> None:
    path = (
        "assets/docs/bohemian-visual-recognition-alliance/models/"
        "mushroom-identification_v1/1.md"
    )
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    path: """\
# Module bohemian-visual-recognition-alliance/models/mushroom-identification_v1/1

The card's code example uses this exact TensorFlow Hub module handle.
""",
                }
            )
        ),
    )

    page = TensorFlowHubArchiveSourceAdapter(client=client).fetch_page({})

    record = page.records[0]
    exact_handle = (
        "bohemian-visual-recognition-alliance/models/"
        "mushroom-identification_v1/1"
    )
    path_candidate = (
        "bohemian-visual-recognition-alliance/"
        "mushroom-identification_v1/1"
    )
    assert record.canonical_url == f"https://tfhub.dev/{exact_handle}"
    assert record.identifiers == (
        Identifier(
            "tensorflow-hub:model",
            "bohemian-visual-recognition-alliance/models/mushroom-identification_v1",
        ),
        Identifier("tensorflow-hub:handle", exact_handle),
    )
    assert record.releases[0].metadata["identity_source"] == "explicit_module_heading"
    assert record.releases[0].metadata["path_handle_candidate"] == path_candidate
    assert record.releases[0].metadata["declared_title_handle"] == exact_handle


@pytest.mark.parametrize(
    "heading_handle",
    [
        "nvidia/&zwnj;unet/&zwnj;industrial/&zwnj;class_1/1",
        "nvidia/\u200cunet/\u200cindustrial/\u200cclass_1/1",
    ],
)
def test_archive_normalizes_only_zwnj_in_explicit_handle_and_preserves_heading(
    heading_handle: str,
) -> None:
    path = "assets/docs/nvidia/models/unet/industrial/class_1/1.md"
    client = _QueuedClient(
        _response({"sha": _REVISION}),
        _response(
            _archive(
                {
                    path: f"# Module {heading_handle}\n\nHistorical card.\n",
                }
            )
        ),
    )

    record = TensorFlowHubArchiveSourceAdapter(client=client).fetch_page({}).records[0]

    canonical_handle = "nvidia/unet/industrial/class_1/1"
    assert record.canonical_url == f"https://tfhub.dev/{canonical_handle}"
    assert record.identifiers[-1] == Identifier("tensorflow-hub:handle", canonical_handle)
    assert record.releases[0].metadata["declared_title_handle"] == heading_handle
    assert record.releases[0].metadata["normalized_declared_title_handle"] == canonical_handle


def test_archive_enforces_document_count_limit() -> None:
    archive = _archive(
        {
            f"assets/docs/google/models/model-{index}/1.md": (
                f"# Module google/model-{index}/1 Sample.\n"
            )
            for index in range(2)
        }
    )
    client = _QueuedClient(_response({"sha": _REVISION}), _response(archive))

    with pytest.raises(ValueError, match="more than 1 model documents"):
        TensorFlowHubArchiveSourceAdapter(max_documents=1, client=client).fetch_page({})
