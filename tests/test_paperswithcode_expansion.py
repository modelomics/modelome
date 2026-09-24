from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from modelome.http import HttpResponse
from modelome.models import Identifier, ModelStatus
from modelome.sources.paperswithcode import PapersWithCodeValidatedMethodsSourceAdapter

_REVISION = "fd7c1cd6bb715116ec3c20e10651616da99ff1aa"
_NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


class QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(body: bytes, url: str) -> HttpResponse:
    return HttpResponse(status=200, headers={}, body=body, url=url)


def _parquet(rows: list[dict[str, Any]]) -> bytes:
    output = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(rows), output)
    return output.getvalue().to_pybytes()


def test_paper_only_method_rows_are_retained_at_lower_confidence() -> None:
    metadata = (
        b'{"id":"pwc-archive/methods","sha":"'
        + _REVISION.encode()
        + b'","siblings":[{"rfilename":"data/train-00000-of-00001.parquet"}]}'
    )
    rows = [
        {
            "url": "https://paperswithcode.com/method/paper-only-method",
            "name": "Paper Only Method",
            "full_name": "Paper Only Method",
            "description": "A method with a linked paper but no source-paper metadata.",
            "paper": {
                "title": "Paper Only Method for Image Recognition",
                "url": "https://paperswithcode.com/paper/paper-only-method",
            },
            "source_url": None,
            "source_title": None,
            "num_papers": 1,
            "collections": [],
        }
    ]
    client = QueuedClient(
        _response(metadata, "https://huggingface.co/api/datasets/pwc-archive/methods"),
        _response(_parquet(rows), "https://cas-bridge.xethub.hf.co/methods.parquet"),
    )
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        name="paper-only-method-candidates",
        client=client,
        clock=lambda: _NOW,
        admission="paper_candidate",
    )

    page = source.fetch_page({})

    assert len(page.records) == 1
    record = page.records[0]
    model = record.models[0]
    assert model.status is ModelStatus.CANDIDATE
    assert model.confidence == 0.2
    assert record.raw["admission"] == "paper_candidate"
    assert record.raw["source_url"] is None
    assert [(link.relation, link.url) for link in record.links] == [
        ("associated_paper", "https://paperswithcode.com/paper/paper-only-method")
    ]


def test_arxiv_identifiers_are_preserved_across_candidate_admissions() -> None:
    from modelome.sources.paperswithcode import _method_candidates

    row = {
        "url": "https://paperswithcode.com/method/arxiv-backed-method",
        "name": "Arxiv Backed Method",
        "full_name": "Arxiv Backed Method",
        "paper": {
            "title": "Arxiv Backed Method Paper",
            "url": "https://paperswithcode.com/paper/arxiv-backed-method",
        },
        "source_url": "https://arxiv.org/abs/2401.12345",
        "source_title": "Arxiv Backed Method Paper",
        "num_papers": 1,
        "collections": [],
    }

    paper_only, _ = _method_candidates(
        [row],
        require_title_overlap=False,
        require_arxiv_source=False,
        allow_missing_source=True,
    )
    arxiv_linked, _ = _method_candidates(
        [row],
        require_title_overlap=False,
        require_arxiv_source=True,
    )

    assert len(paper_only) == len(arxiv_linked) == 1
    assert paper_only[0]["arxiv_id"] == arxiv_linked[0]["arxiv_id"] == "2401.12345"


def test_method_candidates_deduplicate_exact_rows_and_separate_linked_papers() -> None:
    from modelome.sources.paperswithcode import _method_candidates

    base = {
        "url": "https://paperswithcode.com/method/example-method",
        "name": "Example Method",
        "full_name": "Example Method",
        "description": "A method used by multiple papers.",
        "paper": {
            "title": "First paper",
            "url": "https://paperswithcode.com/paper/first-paper",
        },
        "source_url": "https://arxiv.org/abs/2401.12345",
        "source_title": "First paper",
    }
    second_paper = {
        **base,
        "paper": {
            "title": "Second paper",
            "url": "https://paperswithcode.com/paper/second-paper",
        },
    }
    candidates, _ = _method_candidates(
        [base, base, second_paper],
        require_title_overlap=False,
        require_arxiv_source=False,
    )
    assert len(candidates) == 2
    source = PapersWithCodeValidatedMethodsSourceAdapter(
        admission="paper_linked_candidate"
    )

    records = [source._record(candidate, "a" * 40, None) for candidate in candidates]

    duplicate_candidate, _ = _method_candidates(
        [base], require_title_overlap=False, require_arxiv_source=False
    )
    duplicate_record = source._record(duplicate_candidate[0], "a" * 40, None)
    assert records[0].source_record_id == duplicate_record.source_record_id
    assert records[0].source_record_id != records[1].source_record_id
    assert [record.raw["paper_url"] for record in records] == [
        "https://paperswithcode.com/paper/first-paper",
        "https://paperswithcode.com/paper/second-paper",
    ]


def test_evaluation_candidates_keep_distinct_dataset_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    row = {
        "model_name": "Shared Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Shared Model paper",
    }
    rows = [
        {
            "task": "Classification",
            "subtasks": [],
            "datasets": [
                {"name": "Dataset One", "sota": {"rows": [row]}},
                {"name": "Dataset Two", "sota": {"rows": [row]}},
            ],
        }
    ]

    records, rejected, raw_model_rows = _evaluation_records(
        rows,
        revision="a" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    assert raw_model_rows == 2
    assert len(records) == 2
    assert {record.raw["datasets"][0] for record in records} == {
        "Dataset One",
        "Dataset Two",
    }
    assert len({record.source_record_id for record in records}) == 2
    assert all(record.models[0].confidence == 0.35 for record in records)
    assert all(record.models[0].status is ModelStatus.CANDIDATE for record in records)


def test_evaluation_candidates_include_recursive_task_and_subdataset_scope() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    model_row = {
        "model_name": "Example Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Example Model paper",
    }

    def subtask(name: str) -> dict[str, Any]:
        return {
            "task": name,
            "subtasks": [],
            "datasets": [
                {
                    "dataset": "ImageNet",
                    "subdatasets": [
                        {
                            "dataset": "ImageNet-1K",
                            "subdatasets": [],
                            "sota": {"rows": [model_row]},
                        }
                    ],
                }
            ],
        }

    rows = [
        {
            "task": "Classification",
            "subtasks": [
                subtask("Fine-grained classification"),
                subtask("Coarse classification"),
            ],
            "datasets": [],
        }
    ]

    records, rejected, raw_model_rows = _evaluation_records(
        rows,
        revision="b" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    assert raw_model_rows == 2
    assert len(records) == 2
    assert {tuple(record.raw["tasks"]) for record in records} == {
        ("Classification", "Fine-grained classification"),
        ("Classification", "Coarse classification"),
    }
    assert all(record.raw["datasets"] == ["ImageNet", "ImageNet-1K"] for record in records)
    assert len({record.source_record_id for record in records}) == 2


def test_evaluation_model_links_project_exact_huggingface_model_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    model_row = {
        "model_name": "Published Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Published Model paper",
        "model_links": [
            {
                "url": "https://huggingface.co/example/published-model/blob/main/model.safetensors",
                "title": "Published Model weights",
            }
        ],
    }
    rows = [
        {
            "task": "Classification",
            "datasets": [{"dataset": "Example", "sota": {"rows": [model_row]}}],
        }
    ]
    records, rejected, _ = _evaluation_records(
        rows,
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(
        model
        for model in records[0].models
        if model.identifiers and model.identifiers[0].namespace == "huggingface:model"
    )
    assert linked_model.identifiers[0].value == "example/published-model"
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_model_links_project_exact_kaggle_model_version() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    row = {
        "model_name": "Kaggle Published Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Kaggle Published Model paper",
        "model_links": [
            {
                "url": "https://www.kaggle.com/api/v1/models/example/published-model/pyTorch/weights/3/download",
                "title": "Published Model version 3",
            }
        ],
    }
    records, rejected, _ = _evaluation_records(
        [{"task": "Classification", "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}]}],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(
        model for model in records[0].models if model.name == "Published Model version 3"
    )
    assert linked_model.identifiers == (
        Identifier("kaggle:model-instance-version", "example/published-model/pyTorch/weights/3"),
    )
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_model_links_do_not_treat_github_or_dataset_urls_as_model_ids() -> None:
    from modelome.sources.paperswithcode import _evaluation_model_identifier_from_url

    assert (
        _evaluation_model_identifier_from_url("https://github.com/example/model/releases/tag/v1")
        is None
    )
    assert (
        _evaluation_model_identifier_from_url("https://www.kaggle.com/datasets/example/models")
        is None
    )
    for unrelated_url in (
        "https://modelscope.cn/datasets/example/data",
        "https://huggingface.co/datasets/example/model-files",
        "https://huggingface.co/spaces/example/model-demo",
        "https://github.com/example/model/releases/download/v1/model.safetensors",
        "https://figshare.com/ndownloader/files/123456",
        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/abc123.th",
        "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1/model.ckpt",
    ):
        assert _evaluation_model_identifier_from_url(unrelated_url) is None
    assert _evaluation_model_identifier_from_url(
        "https://doi.org/10.5281/zenodo.12345"
    ) == Identifier("zenodo:record", "12345")
    assert _evaluation_model_identifier_from_url(
        "https://zenodo.org/records/12345/files/weights.bin"
    ) == Identifier("zenodo:record", "12345")
    assert _evaluation_model_identifier_from_url("https://doi.org/10.1000/example") is None


def test_evaluation_model_links_project_exact_zenodo_record_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    row = {
        "model_name": "Zenodo Published Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Zenodo Published Model paper",
        "model_links": [
            {
                "url": "https://zenodo.org/records/12345/files/model.safetensors",
                "title": "Zenodo weights",
            }
        ],
    }
    records, rejected, _ = _evaluation_records(
        [{"task": "Classification", "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}]}],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(model for model in records[0].models if model.name == "Zenodo weights")
    assert linked_model.identifiers == (Identifier("zenodo:record", "12345"),)
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_model_links_project_exact_modelscope_model_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    row = {
        "model_name": "ModelScope Published Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "ModelScope Published Model paper",
        "model_links": [
            {
                "url": "https://modelscope.cn/api/v1/models/example/published-model/repo?Revision=master&FilePath=model.safetensors",
                "title": "ModelScope weights",
            }
        ],
    }
    records, rejected, _ = _evaluation_records(
        [{"task": "Classification", "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}]}],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(model for model in records[0].models if model.name == "ModelScope weights")
    assert linked_model.identifiers == (
        Identifier("modelscope:model", "example/published-model"),
    )
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_modelscope_model_page_url_projects_same_exact_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_model_identifier_from_url

    assert _evaluation_model_identifier_from_url(
        "https://modelscope.cn/models/example/published-model/resolve/master/model.safetensors"
    ) == Identifier("modelscope:model", "example/published-model")


def test_evaluation_demucs_official_checkpoint_url_projects_exact_model_signature() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    # This exact filename/path form is listed by Demucs' first-party remote loader.
    row = {
        "model_name": "Demucs MDX checkpoint",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Demucs MDX checkpoint paper",
        "model_links": [
            {
                "url": "https://dl.fbaipublicfiles.com/demucs/mdx_final/0d19c1c6-0f06f20e.th",
                "title": "Demucs checkpoint",
            }
        ],
    }
    records, rejected, _ = _evaluation_records(
        [
            {
                "task": "Source separation",
                "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}],
            }
        ],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(model for model in records[0].models if model.name == "Demucs checkpoint")
    assert linked_model.identifiers == (Identifier("demucs:pretrained-signature", "0d19c1c6"),)
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_model_links_project_exact_civitai_model_version() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    # Civitai's first-party model-version response uses this downloadUrl shape.
    row = {
        "model_name": "Civitai Published Model",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Civitai Published Model paper",
        "model_links": [
            {
                "url": "https://civitai.com/api/download/models/2514310",
                "title": "Civitai model version",
            }
        ],
    }
    records, rejected, _ = _evaluation_records(
        [{"task": "Classification", "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}]}],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    linked_model = next(
        model for model in records[0].models if model.name == "Civitai model version"
    )
    assert linked_model.identifiers == (Identifier("civitai:model-version", "2514310"),)
    assert linked_model.status is ModelStatus.CANDIDATE
    assert linked_model.confidence == 0.65
    model_link = next(link for link in records[0].links if link.relation == "model_artifact")
    assert model_link.model_local_ids == (linked_model.local_id,)


def test_evaluation_model_page_links_project_civitai_model_identity() -> None:
    from modelome.sources.paperswithcode import _evaluation_model_identifier_from_url

    assert _evaluation_model_identifier_from_url(
        "https://civitai.com/models/827184/model-name"
    ) == Identifier("civitai:model", "827184")
    assert (
        _evaluation_model_identifier_from_url("https://civitai.com/api/download/images/123")
        is None
    )


def test_evaluation_fal_and_figshare_links_remain_unjoined_artifact_urls() -> None:
    from modelome.sources.paperswithcode import _evaluation_records

    row = {
        "model_name": "Hosted Demo",
        "paper_url": "https://arxiv.org/abs/2401.12345",
        "paper_title": "Hosted Demo paper",
        "model_links": [
            {"url": "https://fal.ai/models/fal-ai/demo/model-25", "title": "Hosted endpoint"},
            {
                "url": "https://ndownloader.figshare.com/files/33947432",
                "title": "Checkpoint file",
            },
        ],
    }
    records, rejected, _ = _evaluation_records(
        [{"task": "Classification", "datasets": [{"dataset": "Example", "sota": {"rows": [row]}}]}],
        revision="c" * 40,
        data_path="data/train.parquet",
        dataset_id="pwc-archive/evaluation-tables",
        license="CC-BY-SA-4.0",
        max_model_rows=10,
    )

    assert rejected == {}
    artifact_links = [link for link in records[0].links if link.relation == "model_artifact"]
    assert {link.url for link in artifact_links} == {
        "https://fal.ai/models/fal-ai/demo/model-25",
        "https://ndownloader.figshare.com/files/33947432",
    }
    assert all(link.model_local_ids == () for link in artifact_links)
    assert len(records[0].models) == 1
