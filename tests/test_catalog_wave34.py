from __future__ import annotations

from datetime import UTC, datetime

from modelome.sources.catalog import load_sources
from modelome.sources.ddlp_video_checkpoints import DDLPVideoCheckpointSourceAdapter
from modelome.sources.falcon_vla_checkpoint_zoo import FalconVLACheckpointZooAdapter
from modelome.sources.grand_challenge_algorithms import GrandChallengeAlgorithmsSourceAdapter
from modelome.sources.m3gnet_legacy_checkpoint import M3GNetLegacyCheckpointSourceAdapter
from modelome.sources.moftransformer_figshare import MOFTransformerFigshareAdapter
from modelome.sources.paddlenlp_t5_registry import PaddleNlpT5RegistrySourceAdapter
from modelome.sources.phyre_dqn_checkpoints import PhyreDqnCheckpointAdapter
from modelome.sources.seco_checkpoint_registry import SeCoCheckpointRegistrySourceAdapter
from modelome.sources.timm_legacy_mobilevit import TimmLegacyMobileViTSourceAdapter
from modelome.sources.uvr_model_files import UvrModelFilesAdapter


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_load_sources_routes_wave34_adapters_and_preserves_fetch_shape(tmp_path) -> None:
    source_config = tmp_path / "wave34.toml"
    source_config.write_text(
        '''
[[source]]
name = "grand-challenge-public-algorithms"
adapter = "grand_challenge_algorithms"
page_size = 25
max_entries = 500
max_response_bytes = 8388608

[[source]]
name = "moftransformer-figshare-checkpoints"
adapter = "moftransformer_figshare"
max_response_bytes = 2097152

[[source]]
name = "seco-checkpoint-registry"
adapter = "seco_checkpoint_registry"
max_response_bytes = 1048576

[[source]]
name = "ddlp-video-checkpoints"
adapter = "ddlp_video_checkpoints"
repository = "taldatech/ddlp"
branch = "main"
document_path = "README.md"
max_response_bytes = 4194304
max_checkpoints = 8

[[source]]
name = "phyre-dqn-checkpoints"
adapter = "phyre_dqn_checkpoints"
max_response_bytes = 65536

[[source]]
name = "paddlenlp-t5-pretrained-registry"
adapter = "paddlenlp_t5_registry"
repository = "PaddlePaddle/PaddleNLP"
branch = "develop"
source_path = "paddlenlp/transformers/t5/configuration.py"
provider_namespace = "paddlenlp:transformer-model"
max_response_bytes = 4194304

[[source]]
name = "timm-legacy-mobilevit-v0613"
adapter = "timm_legacy_mobilevit"

[[source]]
name = "m3gnet-legacy-mp-2021-2-8-efs"
adapter = "m3gnet_legacy_checkpoint"
repository = "materialyzeai/m3gnet"
branch = "main"
source_path = "m3gnet/models/_m3gnet.py"
provider_namespace = "m3gnet:checkpoint"
max_response_bytes = 4194304
max_entries = 1

[[source]]
name = "uvr-public-vr-mdx-model-files"
adapter = "uvr_model_files"
min_models = 20
max_models = 200
max_response_bytes = 524288

[[source]]
name = "falcon-vla-checkpoint-zoo"
adapter = "falcon_vla_checkpoint_zoo"
max_response_bytes = 4194304
max_entries = 32
''',
        encoding="utf-8",
    )

    class FakeClient:
        pass

    shared_client = FakeClient()
    sources = load_sources(
        source_config,
        client=shared_client,
        clock=_fixed_clock,
        environ={},
    )

    expected = {
        "grand-challenge-public-algorithms": GrandChallengeAlgorithmsSourceAdapter,
        "moftransformer-figshare-checkpoints": MOFTransformerFigshareAdapter,
        "seco-checkpoint-registry": SeCoCheckpointRegistrySourceAdapter,
        "ddlp-video-checkpoints": DDLPVideoCheckpointSourceAdapter,
        "phyre-dqn-checkpoints": PhyreDqnCheckpointAdapter,
        "paddlenlp-t5-pretrained-registry": PaddleNlpT5RegistrySourceAdapter,
        "timm-legacy-mobilevit-v0613": TimmLegacyMobileViTSourceAdapter,
        "m3gnet-legacy-mp-2021-2-8-efs": M3GNetLegacyCheckpointSourceAdapter,
        "uvr-public-vr-mdx-model-files": UvrModelFilesAdapter,
        "falcon-vla-checkpoint-zoo": FalconVLACheckpointZooAdapter,
    }
    assert set(sources) == set(expected)
    for name, adapter_type in expected.items():
        source = sources[name]
        assert isinstance(source, adapter_type)
        assert callable(source.fetch_page)
        assert source.client is shared_client
