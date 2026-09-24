from __future__ import annotations

import json
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.sources.chai1_components import Chai1ComponentRegistryAdapter

REV = "a" * 40
REPOSITORY = "chaidiscovery/chai-lab"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def json_response(value: Any, url: str) -> HttpResponse:
    return HttpResponse(200, {}, json.dumps(value).encode(), url)


def text_response(value: str, url: str) -> HttpResponse:
    return HttpResponse(200, {}, value.encode(), url)


def test_chai1_components_are_mapped_to_first_party_urls() -> None:
    api = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    inference_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REV}/chai_lab/chai1.py"
    paths_url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REV}/chai_lab/utils/paths.py"
    inference = '''
def run():
    with _component_moved_to("feature_embedding.pt", device): pass
    with _component_moved_to("bond_loss_input_proj.pt", device): pass
    with _component_moved_to("token_embedder.pt", device): pass
    with _component_moved_to("trunk.pt", device): pass
    with _component_moved_to("diffusion_module.pt", device=device): pass
    with _component_moved_to("confidence_head.pt", device=device): pass
'''
    paths = '''
COMPONENT_URL = (
    "https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}"
)
'''
    client = QueueClient(
        json_response({"sha": REV}, api),
        text_response(inference, inference_url),
        text_response(paths, paths_url),
    )

    page = Chai1ComponentRegistryAdapter(client=client).fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 6
    record = page.records[0]
    assert record.models[0].name == "Chai-1"
    components = {release.metadata["component_filename"]: release for release in record.releases}
    assert set(components) == {
        "feature_embedding.pt",
        "bond_loss_input_proj.pt",
        "token_embedder.pt",
        "trunk.pt",
        "diffusion_module.pt",
        "confidence_head.pt",
    }
    assert components["trunk.pt"].metadata["artifact_url"] == (
        "https://chaiassets.com/chai1-inference-depencencies/models_v2/trunk.pt"
    )
    assert len(record.links) == 8
    assert all(not link.crawl for link in record.links)
    assert client.calls == [api, inference_url, paths_url]


def test_chai1_component_registry_skips_unchanged_revision() -> None:
    api = f"https://api.github.com/repos/{REPOSITORY}/commits/main"
    client = QueueClient(json_response({"sha": REV}, api))

    page = Chai1ComponentRegistryAdapter(client=client).fetch_page(
        {"completed_revision": REV, "component_count": 6}
    )

    assert page.records == ()
    assert page.upstream_count == 6
    assert len(client.calls) == 1


def test_chai1_rejects_dynamic_component_names_and_other_asset_templates() -> None:
    from modelome.sources.chai1_components import _parse_components, _parse_url_template

    with pytest.raises(ValueError, match="filenames must be literals"):
        _parse_components("_component_moved_to(COMPONENT, device)", 10)
    with pytest.raises(ValueError, match="changed unexpectedly"):
        _parse_url_template('COMPONENT_URL = "https://example.org/{comp_key}"')
