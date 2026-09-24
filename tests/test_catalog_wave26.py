from __future__ import annotations

import pytest

from modelome.sources.catalog import create_source
from modelome.sources.chgnet_pretrained_weights import CHGNetPretrainedWeightsSourceAdapter
from modelome.sources.ftw_release_checkpoints import FTWReleaseCheckpointSourceAdapter
from modelome.sources.geom2vec_checkpoint_files import Geom2VecCheckpointFilesSourceAdapter
from modelome.sources.medigan_registry import MediganRegistrySourceAdapter
from modelome.sources.nvlabs_edm_checkpoints import NVlabsEDMCheckpointSourceAdapter
from modelome.sources.paddlenlp_ernie_registry import PaddleNlpErnieRegistrySourceAdapter
from modelome.sources.pmlr import PmlrSourceAdapter
from modelome.sources.pmt_pretrained_checkpoints import PMTPretrainedCheckpointSourceAdapter
from modelome.sources.sherpa_asr_model_release import SherpaAsrModelReleaseSourceAdapter
from modelome.sources.timm_legacy_efficientnet import TimmLegacyEfficientNetSourceAdapter


@pytest.mark.parametrize(
    ("config", "adapter_type", "attributes"),
    [
        (
            {
                "name": "nvlabs-edm-checkpoints",
                "adapter": "nvlabs_edm_checkpoints",
                "repository": "NVlabs/edm",
                "branch": "main",
                "document_path": "README.md",
                "max_response_bytes": 4194304,
                "max_checkpoints": 100,
            },
            NVlabsEDMCheckpointSourceAdapter,
            {"max_checkpoints": 100},
        ),
        (
            {
                "name": "dinner-group-geom2vec-checkpoints",
                "adapter": "geom2vec_checkpoint_files",
                "repository": "dinner-group/geom2vec",
                "branch": "main",
                "provider_namespace": "geom2vec:checkpoint",
                "max_response_bytes": 4194304,
                "max_entries": 100000,
            },
            Geom2VecCheckpointFilesSourceAdapter,
            {"provider_namespace": "geom2vec:checkpoint", "max_entries": 100000},
        ),
        (
            {
                "name": "ftw-release-checkpoints",
                "adapter": "ftw_release_checkpoints",
                "max_response_bytes": 2097152,
            },
            FTWReleaseCheckpointSourceAdapter,
            {"max_response_bytes": 2097152},
        ),
        (
            {
                "name": "chgnet-pretrained-weights",
                "adapter": "chgnet_pretrained_weights",
                "max_response_bytes": 12582912,
            },
            CHGNetPretrainedWeightsSourceAdapter,
            {"max_response_bytes": 12582912},
        ),
        (
            {
                "name": "medigan-first-party-model-index",
                "adapter": "medigan_registry",
                "repository": "RichardObi/medigan",
                "branch": "main",
                "max_response_bytes": 4194304,
                "max_entries": 500,
            },
            MediganRegistrySourceAdapter,
            {"repository": "RichardObi/medigan", "max_entries": 500},
        ),
        (
            {
                "name": "paddlenlp-ernie-pretrained-registry",
                "adapter": "paddlenlp_ernie_registry",
                "repository": "PaddlePaddle/PaddleNLP",
                "branch": "develop",
                "source_path": "paddlenlp/transformers/ernie/configuration.py",
                "max_response_bytes": 4194304,
            },
            PaddleNlpErnieRegistrySourceAdapter,
            {"repository": "PaddlePaddle/PaddleNLP", "branch": "develop"},
        ),
        (
            {
                "name": "sherpa-asr-model-release",
                "adapter": "sherpa_asr_model_release",
                "max_response_bytes": 16777216,
                "max_assets": 1200,
            },
            SherpaAsrModelReleaseSourceAdapter,
            {"max_assets": 1200},
        ),
        (
            {
                "name": "timm-legacy-efficientnet-v0613",
                "adapter": "timm_legacy_efficientnet",
            },
            TimmLegacyEfficientNetSourceAdapter,
            {},
        ),
        (
            {
                "name": "pmt-pretrained-checkpoints",
                "adapter": "pmt_pretrained_checkpoints",
                "max_response_bytes": 4194304,
                "max_entries": 32,
            },
            PMTPretrainedCheckpointSourceAdapter,
            {"max_entries": 32},
        ),
        (
            {
                "name": "pmlr",
                "adapter": "pmlr",
                "index_url": "https://proceedings.mlr.press/",
                "max_response_bytes": 16777216,
                "max_volumes": 2000,
                "max_papers_per_volume": 5000,
            },
            PmlrSourceAdapter,
            {"max_volumes": 2000, "max_papers_per_volume": 5000},
        ),
    ],
)
def test_wave26_checkpoint_adapters_load_through_factory(
    config: dict[str, object], adapter_type: type, attributes: dict[str, object]
) -> None:
    source = create_source(config)

    assert isinstance(source, adapter_type)
    assert source.name == config["name"]
    for attribute, expected in attributes.items():
        assert getattr(source, attribute) == expected
