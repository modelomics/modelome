from __future__ import annotations

import json

from modelome.entries import build_entries, source_record_to_entry_seed
from modelome.http import HttpResponse
from modelome.sources.huggingface import HuggingFaceSourceAdapter
from modelome.sources.openrouter import OpenRouterModelsSourceAdapter


class _JsonClient:
    def __init__(self, payload: object, url: str) -> None:
        self.payload = payload
        self.url = url

    def get(self, url: str, *, params=None, headers=None) -> HttpResponse:
        return HttpResponse(
            status=200,
            headers={},
            body=json.dumps(self.payload).encode(),
            url=self.url,
        )


def test_openrouter_and_huggingface_adapter_records_join_by_declared_origin_id() -> None:
    hf_repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    openrouter_page = OpenRouterModelsSourceAdapter(
        client=_JsonClient(
            {
                "data": [
                    {
                        "id": "meta-llama/llama-3.1-8b-instruct",
                        "name": "Llama 3.1 8B Instruct",
                        "hugging_face_id": hf_repo,
                        "created": 1722470400,
                        "architecture": {"modality": "text->text"},
                        "links": {},
                    },
                    {
                        "id": "meta-llama/llama-3.1-8b-instruct:free",
                        "name": "Llama 3.1 8B Instruct Free",
                        "hugging_face_id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                        "created": 1722470400,
                        "architecture": {"modality": "text->text"},
                        "links": {},
                    },
                ],
                "links": {"next": None},
                "total_count": 2,
            },
            "https://openrouter.ai/api/v1/models",
        )
    ).fetch_page({})
    hf_page = HuggingFaceSourceAdapter(
        client=_JsonClient(
            [{"id": hf_repo, "sha": "recorded-commit", "siblings": []}],
            f"https://huggingface.co/api/models/{hf_repo}",
        )
    ).fetch_page({})

    seeds = [
        source_record_to_entry_seed(record, source="openrouter-models")
        for record in openrouter_page.records
    ]
    seeds.extend(
        source_record_to_entry_seed(record, source="huggingface")
        for record in hf_page.records
    )
    entries = build_entries(seeds).entries

    joined = next(
        entry
        for entry in entries
        if any(member.source == "huggingface" for member in entry.members)
    )
    assert {member.source for member in joined.members} == {
        "openrouter-models",
        "huggingface",
    }
    assert len(entries) == 2
    unrelated = next(entry for entry in entries if entry.id != joined.id)
    assert {member.source for member in unrelated.members} == {"openrouter-models"}
    assert any(":free" in member.local_id for member in unrelated.members)
