from __future__ import annotations

from modelome.entries import build_entries


def _cloudflare_seed(
    *, source: str, source_record_id: str, namespace: str, model_id: str
) -> dict[str, object]:
    return {
        "source": source,
        "source_record_id": source_record_id,
        "canonical_url": f"https://example.test/{source_record_id}",
        "title": model_id,
        "kind": "model_card",
        "models": [
            {
                "local_id": "model",
                "name": model_id,
                "identifiers": [{"namespace": namespace, "value": model_id}],
            }
        ],
    }


def test_cloudflare_live_catalog_and_lifecycle_ids_share_exact_model_entry() -> None:
    live_id = "@cf/meta/llama-3.1-8b-instruct"
    retired_id = "@cf/moonshotai/kimi-k2.5"
    result = build_entries(
        [
            _cloudflare_seed(
                source="cloudflare-workers-ai",
                source_record_id="live-llama",
                namespace="cloudflare-workers-ai:model",
                model_id=live_id,
            ),
            _cloudflare_seed(
                source="cloudflare-workers-ai-deprecations-2026-05",
                source_record_id="notice-llama",
                namespace="cloudflare:workers-ai",
                model_id=live_id,
            ),
            _cloudflare_seed(
                source="cloudflare-workers-ai-deprecations-2026-05",
                source_record_id="notice-kimi",
                namespace="cloudflare:workers-ai",
                model_id=retired_id,
            ),
        ]
    )

    by_id = {
        (identifier.namespace, identifier.value): entry
        for entry in result.entries
        for identifier in entry.identifiers
        if identifier.namespace == "cloudflare:workers-ai"
    }
    assert len(result.entries) == 2
    assert len(by_id[("cloudflare:workers-ai", live_id)].members) == 2
    assert len(by_id[("cloudflare:workers-ai", retired_id)].members) == 1
