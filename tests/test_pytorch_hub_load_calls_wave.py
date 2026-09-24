from __future__ import annotations

from collections import deque

from modelome.http import HttpResponse
from modelome.normalize import content_hash
from modelome.sources.pytorch_hub_load_calls import PyTorchHubLoadCallSourceAdapter


class FakeHttp:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = deque(responses)
        self.urls: list[str] = []

    def get(self, url: str, *, headers=None) -> HttpResponse:
        self.urls.append(url)
        response = self.responses.popleft()
        return HttpResponse(response.status, response.headers, response.body, url)


def _html_response(url: str, body: str) -> HttpResponse:
    return HttpResponse(200, {"content-type": "text/html"}, body.encode(), url)


def test_card_code_captures_exact_pinned_hub_load_and_pretrained_release() -> None:
    index = _html_response(
        "https://pytorch.org/hub/",
        """
        <a href="/hub/pytorch_vision_inception_v3/">Inception v3</a>
        <a href="https://evil.example/hub/not-a-card">External</a>
        <a href="/hub/about/help">About</a>
        """,
    )
    card = _html_response(
        "https://pytorch.org/hub/pytorch_vision_inception_v3",
        """
        <title>Inception V3 — PyTorch Hub</title>
        <pre><code>model = torch.hub.load(
            'pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        </code></pre>
        <code>torch.hub.load(repo, 'dynamic')</code>
        """,
    )
    client = FakeHttp([index, card])
    adapter = PyTorchHubLoadCallSourceAdapter(client=client)

    first = adapter.fetch_page({})
    assert first.complete is False
    assert first.next_state["entry_count"] == 1

    page = adapter.fetch_page(first.next_state)
    assert page.complete is True
    assert len(page.records) == 1
    record = page.records[0]
    assert record.identifiers[0].namespace == "pytorch:hub-model-page"
    assert record.identifiers[0].value == "pytorch_vision_inception_v3"
    assert len(record.models) == 1
    assert record.models[0].local_id == (
        "pytorch:hub-model-page:" + content_hash("pytorch_vision_inception_v3")[:24] + "#model"
    )
    assert len(record.releases) == 1
    release = record.releases[0]
    assert release.version == "v0.10.0"
    assert release.metadata == {
        "repository": "pytorch/vision",
        "repository_ref": "v0.10.0",
        "entrypoint": "inception_v3",
        "arguments": {"pretrained": "True"},
        "loader": "torch.hub.load",
    }
    assert any(
        link.url == "https://github.com/pytorch/vision/tree/v0.10.0"
        and link.relation == "official_implementation"
        for link in record.links
    )
    assert client.urls == [
        "https://pytorch.org/hub/",
        "https://pytorch.org/hub/pytorch_vision_inception_v3",
    ]


def test_dynamic_repo_or_non_pretrained_call_does_not_create_release() -> None:
    calls = """
    torch.hub.load(repo_name, 'dynamic', pretrained=True)
    torch.hub.load('pytorch/vision:main', 'resnet50', pretrained=False)
    torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    """
    fake = FakeHttp(
        [
            _html_response(
                "https://pytorch.org/hub/",
                '<a href="/hub/pytorch_vision_resnet">ResNet</a>',
            ),
            _html_response(
                "https://pytorch.org/hub/pytorch_vision_resnet",
                f"<pre><code>{calls}</code></pre>",
            ),
        ]
    )
    adapter = PyTorchHubLoadCallSourceAdapter(client=fake)
    page = adapter.fetch_page(adapter.fetch_page({}).next_state)
    assert len(page.records[0].releases) == 0
    assert len(page.records[0].raw["torch_hub_load_calls"]) == 2


def test_first_party_resnet_card_keeps_each_commented_pretrained_variant_distinct() -> None:
    # These loader examples are copied from the official ResNet Hub card's code sample.
    # https://pytorch.org/hub/pytorch_vision_resnet/
    card_code = """
    import torch
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    # or any of these variants
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    model.eval()
    """
    client = FakeHttp(
        [
            _html_response(
                "https://pytorch.org/hub/",
                '<a href="/hub/pytorch_vision_resnet/">ResNet</a>',
            ),
            _html_response(
                "https://pytorch.org/hub/pytorch_vision_resnet/",
                f"<pre><code>{card_code}</code></pre>",
            ),
        ]
    )
    adapter = PyTorchHubLoadCallSourceAdapter(client=client)
    record = adapter.fetch_page(adapter.fetch_page({}).next_state).records[0]

    assert [release.metadata["entrypoint"] for release in record.releases] == [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
    ]
    assert all(release.version == "v0.10.0" for release in record.releases)
    assert len({release.local_id for release in record.releases}) == 5
    assert len(
        {
            (identifier.namespace, identifier.value)
            for release in record.releases
            for identifier in release.identifiers
        }
    ) == 5
