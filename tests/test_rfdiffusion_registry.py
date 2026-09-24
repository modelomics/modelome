from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from modelome.http import HttpResponse
from modelome.models import Identifier
from modelome.sources.rfdiffusion_registry import RFDiffusionCheckpointSourceAdapter

COMMIT = "b" * 40
BASE = "http://files.ipd.uw.edu/pub/RFdiffusion"


class QueueClient:
    def __init__(self, *responses: HttpResponse) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def get(self, url: str, *, params: Any = None, headers: Any = None) -> HttpResponse:
        self.calls.append(url)
        if not self.responses:
            raise AssertionError(f"unexpected GET {url}")
        return self.responses.pop(0)


def response(value: Any, url: str) -> HttpResponse:
    body = json.dumps(value).encode() if isinstance(value, dict) else value.encode()
    return HttpResponse(200, {}, body, url)


def test_rfdiffusion_official_readme_enumerates_exact_checkpoint_urls() -> None:
    api = "https://api.github.com/repos/RosettaCommons/RFdiffusion/commits/main"
    readme = (
        f"wget {BASE}/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt\n"
        f"wget {BASE}/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt\n"
        f"wget {BASE}/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt\n"
        f"wget {BASE}/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt\n"
        f"wget {BASE}/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt\n"
        f"wget {BASE}/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt\n"
        f"wget {BASE}/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt\n"
        f"wget {BASE}/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt\n"
        f"wget {BASE}/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt\n"
        "wget https://example.test/not-first-party/Ignore.pt\n"
        f"wget {BASE}/not-a-hash/not_a_checkpoint.pt\n"
        f"See {BASE}/00000000000000000000000000000000/linked_only.pt for details.\n"
    )
    raw = (
        f"https://raw.githubusercontent.com/RosettaCommons/RFdiffusion/{COMMIT}/README.md"
    )
    client = QueueClient(
        response({"sha": COMMIT}, api),
        response(readme, raw),
    )
    source = RFDiffusionCheckpointSourceAdapter(
        client=client,
        clock=lambda: datetime(2026, 9, 23, tzinfo=UTC),
    )

    page = source.fetch_page({})

    assert page.complete and page.authoritative_snapshot
    assert page.upstream_count == 9
    record = page.records[0]
    checkpoints = {model.name.removeprefix("RFdiffusion "): model for model in record.models}
    assert set(checkpoints) == {
        "Base_ckpt",
        "Complex_base_ckpt",
        "Complex_Fold_base_ckpt",
        "InpaintSeq_ckpt",
        "InpaintSeq_Fold_ckpt",
        "ActiveSite_ckpt",
        "Base_epoch8_ckpt",
        "Complex_beta_ckpt",
        "RF_structure_prediction_weights",
    }
    base_model = checkpoints["Base_ckpt"]
    assert base_model.identifiers == (Identifier("rfdiffusion:checkpoint", "Base_ckpt"),)
    base_release = next(
        release for release in record.releases if release.model_local_id == base_model.local_id
    )
    exact_url = f"{BASE}/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt"
    assert base_release.identifiers == (Identifier("rfdiffusion:checkpoint:url", exact_url),)
    assert base_release.metadata["checkpoint_url"] == exact_url
    assert all(not link.crawl for link in record.links)
    assert exact_url in {link.url for link in record.links}
    assert client.calls == [api, raw]


def test_rfdiffusion_rejects_untrusted_hosts_and_duplicate_checkpoint_identity() -> None:
    from modelome.sources.rfdiffusion_registry import _parse_weight_commands

    with pytest.raises(ValueError, match="no admitted checkpoint"):
        _parse_weight_commands(
            "wget https://example.test/pub/RFdiffusion/" + "a" * 32 + "/model.pt",
            10,
        )
    duplicated = (
        f"wget {BASE}/" + "a" * 32 + "/model.pt\n"
        f"wget {BASE}/" + "b" * 32 + "/model.pt"
    )
    with pytest.raises(ValueError, match="repeats a checkpoint"):
        _parse_weight_commands(duplicated, 10)


def test_rfdiffusion_adapter_rejects_other_repositories() -> None:
    with pytest.raises(ValueError, match="repository must"):
        RFDiffusionCheckpointSourceAdapter(repository="somefork/RFdiffusion", client=QueueClient())
