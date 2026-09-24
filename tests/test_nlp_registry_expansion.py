from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.stanza_resources import StanzaResourcesSourceAdapter

REVISION = "d" * 40


class QueuedClient:
    def __init__(self, *payloads: Any) -> None:
        self.payloads = list(payloads)

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        payload = self.payloads.pop(0)
        return HttpResponse(200, {}, json.dumps(payload).encode(), url)


def test_stanza_catalog_includes_named_pipeline_variants_and_component_links() -> None:
    manifest = {
        "en": {
            "tokenize": {"combined": {"md5": "1" * 32}},
            "pos": {"ewt": {"md5": "2" * 32}},
            "default_processors": {"tokenize": "combined", "pos": "ewt"},
            "packages": {
                "default": {"tokenize": "combined", "pos": "ewt"},
                "tokenizer_only": {"tokenize": "combined"},
            },
        },
        "english": {"alias": "en"},
    }
    tree = {
        "truncated": False,
        "tree": [{"path": "resources_1.14.0.json", "type": "blob"}],
    }
    source = StanzaResourcesSourceAdapter(
        client=QueuedClient({"sha": REVISION}, tree, manifest),
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    record = page.records[0]
    variants = {model.name: model for model in record.models}
    assert variants["en/package/default"].identifiers == (
        Identifier("stanza:pipeline-package", "en/package/default"),
    )
    assert variants["en/package/tokenizer_only"].identifiers == (
        Identifier("stanza:pipeline-package", "en/package/tokenizer_only"),
    )
    assert page.upstream_count == 4
    variant_releases = {
        release.model_local_id: release
        for release in record.releases
        if release.model_local_id.startswith("stanza:en/package/")
    }
    assert variant_releases["stanza:en/package/default#model"].identifiers == (
        Identifier(
            "stanza:pipeline-package-version",
            "en/package/default@1.14.0",
        ),
    )
    assert {
        (relation.subject_local_id, relation.target.identifiers[0].value)
        for relation in record.model_relations
        if relation.predicate == "contains_component"
    } == {
        ("stanza:en/package/default#model", "en/tokenize/combined"),
        ("stanza:en/package/default#model", "en/pos/ewt"),
        ("stanza:en/package/tokenizer_only#model", "en/tokenize/combined"),
    }
