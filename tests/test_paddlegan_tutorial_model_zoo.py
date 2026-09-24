from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.paddlegan_tutorial_model_zoo import (
    PaddleGanTutorialModelZooSourceAdapter,
)

_REVISION = "6" * 40
_AOT = b"""# AOT GAN

**Paper:** [Aggregated Contextual Transformations](https://arxiv.org/abs/2104.01031)
**Official Repo:** [upstream implementation](https://github.com/example/aotgan)
AI Studio Project: (https://aistudio.baidu.com/aistudio/projectdetail/123)

Download pretrained generator weights from:
(https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams)
"""
_VSR = (
    b"# Video Super Resolution\n\n"
    b"| Method | Dataset | Download Link | Paper |\n"
    b"| --- | --- | --- | --- |\n"
    b"| EDVR | REDS | "
    b"[checkpoint](https://paddlegan.bj.bcebos.com/models/EDVR.pdparams) | "
    b"[paper](https://arxiv.org/abs/1905.02716) |\n"
    b"| BasicVSR | REDS | "
    b"[checkpoint](https://paddlegan.bj.bcebos.com/models/BasicVSR.pdparams) | "
    b"[paper](https://arxiv.org/abs/2012.02181) |\n"
)


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, str] | None]] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.calls.append((url, headers))
        return self.responses.pop(0)


def _response(body: bytes, *, url: str = "https://fixtures.test/paddlegan") -> HttpResponse:
    return HttpResponse(200, {"etag": '"fixture"'}, body, url)


def _commit() -> HttpResponse:
    return _response((f'{{"sha": "{_REVISION}"}}').encode())


def _archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        for path, body in files.items():
            package.writestr(f"PaddleGAN-{_REVISION}/{path}", body)
    return buffer.getvalue()


def test_paddlegan_tutorials_scope_prose_and_table_resources_to_model_artifacts() -> None:
    archive = _archive(
        {
            "docs/en_US/tutorials/aotgan.md": _AOT,
            "docs/en_US/tutorials/video_super_resolution.md": _VSR,
            "docs/zh_CN/tutorials/aotgan.md": _AOT,
        }
    )
    client = _QueuedClient(_commit(), _response(archive))
    adapter = PaddleGanTutorialModelZooSourceAdapter(
        repository="example/PaddleGAN",
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot
    assert page.upstream_count == 3
    assert page.next_state["document_count"] == 2
    assert client.calls[1][0].endswith(f"/archive/{_REVISION}.zip")
    aot = next(record for record in page.records if record.title == "AOT GAN")
    assert aot.identifiers == (
        Identifier(
            "paddlegan:checkpoint-model",
            "https://paddlegan.bj.bcebos.com/models/AotGan_g.pdparams",
        ),
    )
    assert aot.models[0].status.value == "released"
    assert aot.releases[0].metadata["checkpoint_url"] == aot.identifiers[0].value
    assert {
        (link.url, link.relation, link.crawl, link.model_local_ids)
        for link in aot.links
    } >= {
        (
            "https://arxiv.org/abs/2104.01031",
            "paper_reference",
            False,
            (aot.models[0].local_id,),
        ),
        (
            "https://github.com/example/aotgan",
            "source_implementation",
            False,
            (aot.models[0].local_id,),
        ),
        (
            "https://aistudio.baidu.com/aistudio/projectdetail/123",
            "related_resource",
            False,
            (aot.models[0].local_id,),
        ),
    }
    edvr = next(record for record in page.records if record.title == "EDVR")
    basic = next(record for record in page.records if record.title == "BasicVSR")
    assert ("https://arxiv.org/abs/1905.02716", "paper_reference") in {
        (link.url, link.relation) for link in edvr.links
    }
    assert ("https://arxiv.org/abs/2012.02181", "paper_reference") in {
        (link.url, link.relation) for link in basic.links
    }
    assert all(link.model_local_ids == (edvr.models[0].local_id,) for link in edvr.links)
    assert all(link.model_local_ids == (basic.models[0].local_id,) for link in basic.links)


def test_paddlegan_tutorials_skip_unchanged_commit() -> None:
    client = _QueuedClient(_commit())
    adapter = PaddleGanTutorialModelZooSourceAdapter(client=client)

    page = adapter.fetch_page({"completed_revision": _REVISION, "model_count": 8})

    assert page.records == ()
    assert page.complete
    assert page.upstream_count == 8
    assert len(client.calls) == 1


def test_paddlegan_tutorials_reject_documents_without_first_party_model_artifacts() -> None:
    archive = _archive(
        {
            "docs/en_US/tutorials/example.md": b"""# Example

Download dataset: https://paddlegan.bj.bcebos.com/datasets/example.tar
""",
        }
    )
    client = _QueuedClient(_commit(), _response(archive))

    with pytest.raises(ValueError, match="no direct first-party model artifacts"):
        PaddleGanTutorialModelZooSourceAdapter(client=client).fetch_page({})
