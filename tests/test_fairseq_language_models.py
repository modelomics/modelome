from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.fairseq_language_models import (
    FairseqPretrainedLanguageModelSourceAdapter,
)

_REVISION = "b" * 40


class _QueuedClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def _response(body: str | Mapping[str, Any]) -> HttpResponse:
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    return HttpResponse(200, {}, raw, "https://fixtures.test/fairseq")


def test_fairseq_language_model_source_keeps_direct_first_party_archives_only() -> None:
    document = "\n".join(
        (
            "# Neural Language Modeling",
            "## Pre-trained models",
            "Model | Description | Dataset | Download",
            "--- | --- | --- | ---",
            "`transformer_lm.wiki103.adaptive` | adaptive model | WikiText-103 | "
            "[download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wiki103.tar.bz2)",
            "`transformer_lm.wmt19.en` | English LM | WMT News Crawl | "
            "[download (.tar.gz)](https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en.tar.gz)",
            "`hf-only` | hosted model | N/A | "
            "[download](https://huggingface.co/facebook/hf-only)",
            "`metadata-only` | no checkpoint | N/A | "
            "[dictionary](https://dl.fbaipublicfiles.com/fairseq/models/dict.txt)",
        )
    )
    adapter = FairseqPretrainedLanguageModelSourceAdapter(
        name="fairseq-language-models",
        document_path="examples/language_model/README.md",
        client=_QueuedClient(_response({"sha": _REVISION}), _response(document)),
    )

    page = adapter.fetch_page({})

    assert page.authoritative_snapshot is True
    assert page.upstream_count == 2
    assert [record.title for record in page.records] == [
        "transformer_lm.wiki103.adaptive",
        "transformer_lm.wmt19.en",
    ]
    assert page.records[0].identifiers == (
        Identifier(
            "fairseq:language-model-model",
            "Pre-trained models / transformer_lm.wiki103.adaptive / WikiText-103",
        ),
    )
    assert [link.url for link in page.records[0].links if link.relation == "weights"] == [
        "https://dl.fbaipublicfiles.com/fairseq/models/wiki103.tar.bz2"
    ]


@pytest.mark.parametrize(
    ("document_path", "namespace"),
    (
        ("examples/roberta/README.md", "fairseq:roberta-model"),
        ("examples/bart/README.md", "fairseq:bart-model"),
        ("examples/xlmr/README.md", "fairseq:xlmr-model"),
        ("examples/mbart/README.md", "fairseq:mbart-model"),
        ("examples/language_model/README.md", "fairseq:language-model-model"),
    ),
)
def test_fairseq_language_model_documents_have_separate_namespaces(
    document_path: str, namespace: str
) -> None:
    adapter = FairseqPretrainedLanguageModelSourceAdapter(
        name="fairseq-model-zoo",
        document_path=document_path,
        client=_QueuedClient(),
    )

    assert adapter.repository == "facebookresearch/fairseq"
    assert adapter.branch == "main"
    assert adapter.provider_namespace == namespace


def test_fairseq_source_rejects_paths_outside_the_fixed_model_zoo_docs() -> None:
    with pytest.raises(ValueError, match="configured fairseq language-model README"):
        FairseqPretrainedLanguageModelSourceAdapter(
            name="fairseq-arbitrary-doc",
            document_path="examples/translation/README.md",
            client=_QueuedClient(),
        )


def test_fairseq_language_model_proposal_is_disabled_and_pinned_to_known_docs() -> None:
    import tomllib

    proposal = tomllib.loads(Path("config/proposals/fairseq_language_models.toml").read_text())

    sources = proposal["source"]
    assert len(sources) == 5
    assert all(source["enabled"] is False for source in sources)
    assert {source["document_path"] for source in sources} == {
        "examples/roberta/README.md",
        "examples/bart/README.md",
        "examples/xlmr/README.md",
        "examples/mbart/README.md",
        "examples/language_model/README.md",
    }
