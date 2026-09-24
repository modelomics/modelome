from __future__ import annotations

import json
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from modelome.http import HttpResponse
from modelome.sources.nltk_data_models import NltkDataModelIndexSourceAdapter

_REVISION = "b" * 40


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        self.calls.append(url)
        if "/commits/" in url:
            body = json.dumps({"sha": _REVISION}).encode()
        elif url.endswith("/index.xml"):
            root = ET.Element("nltk_data")
            packages = ET.SubElement(root, "packages")
            for package_id, name, subdir, size, checksum in (
                (
                    "averaged_perceptron_tagger_eng",
                    "Averaged Perceptron Tagger (JSON)",
                    "taggers",
                    "1539115",
                    "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b",
                ),
                (
                    "maxent_ne_chunker",
                    "ACE Named Entity Chunker (Maximum entropy)",
                    "chunkers",
                    "13404747",
                    "b7cdb936c551c06ef2cdc6227238c5ccc9c8c5259a11f99f4a937419d52af61b",
                ),
                (
                    "bllip_wsj_no_aux",
                    "BLLIP Parser: WSJ Model",
                    "models",
                    "24516205",
                    "e00339b708f23c24b5cf67ff3db5711dd4d80b21083f52787cf167bf77ac2126",
                ),
                (
                    "moses_sample",
                    "Moses Sample Models",
                    "models",
                    "10961490",
                    "0639dfa1d1939295d29c3d57478b1eb7767405dc916effe2cf6a90071943f7e8",
                ),
                (
                    "word2vec_sample",
                    "Word2Vec Sample",
                    "models",
                    "49396025",
                    "d29ff84a6ceca407f8578648568c55894dac34641ceb1fa02f920264fe326b43",
                ),
                (
                    "punkt",
                    "Punkt Tokenizer Models",
                    "tokenizers",
                    "13905355",
                    "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec",
                ),
                (
                    "punkt_tab",
                    "Punkt Tokenizer Models",
                    "tokenizers",
                    "4319076",
                    "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
                ),
                (
                    "universal_tagset",
                    "Mappings to the Universal Part-of-Speech Tagset",
                    "taggers",
                    "19095",
                    "d490e1ae8f5625dcdfdda04be15c22a2aade8c2561a36a61edcdf0c7d6aa8352",
                ),
                (
                    "brown",
                    "Brown Corpus",
                    "corpora",
                    "3314357",
                    "9b275f9b3b95d7bd66ccfb7cd259f445a13bbe5d1f4107aba09fd3e8364bafa6",
                ),
                (
                    "wmt15_eval",
                    "Evaluation data from WMT15",
                    "models",
                    "383096",
                    "56ea67e320f75be1abdee60b9d57aef1bd50324edd176e11c3c40f451043c80e",
                ),
                (
                    "rslp",
                    "RSLP Stemmer",
                    "stemmers",
                    "3805",
                    "f482f9666a2a76cdd4acab16b01a44b002550ebaac29906dbd5a1bbc281e4f8b",
                ),
            ):
                ET.SubElement(
                    packages,
                    "package",
                    {
                        "id": package_id,
                        "name": name,
                        "subdir": subdir,
                        "size": size,
                        "sha256_checksum": checksum,
                        "url": (
                            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
                            f"packages/{subdir}/{package_id}.zip"
                        ),
                    },
                )
            body = ET.tostring(root)
        else:
            raise AssertionError(f"unexpected URL {url}")
        return HttpResponse(200, {"content-type": "application/xml"}, body, url)


def test_nltk_data_index_extracts_exact_model_package_assets() -> None:
    config = tomllib.loads(
        (Path(__file__).parents[1] / "config/proposals/nltk_data_models.toml").read_text()
    )["source"][0]
    assert config["enabled"] is False
    client = _FakeClient()
    adapter = NltkDataModelIndexSourceAdapter(
        name=config["name"],
        repository=config["repository"],
        branch=config["branch"],
        provider_namespace=config["provider_namespace"],
        client=client,
    )

    page = adapter.fetch_page({})

    assert page.upstream_count == 7
    assert {record.raw["id"] for record in page.records} == {
        "averaged_perceptron_tagger_eng",
        "bllip_wsj_no_aux",
        "maxent_ne_chunker",
        "moses_sample",
        "punkt",
        "punkt_tab",
        "word2vec_sample",
    }
    model = next(record for record in page.records if record.raw["id"] == "punkt_tab")
    assert model.canonical_url.endswith("/packages/tokenizers/punkt_tab.zip")
    assert model.models[0].identifiers[0].value == "punkt_tab"
    assert model.releases[0].metadata["sha256"] == (
        "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
    )
    assert any(link.relation == "weights" for link in model.links)
    assert not any(url.endswith(".zip") for url in client.calls)
