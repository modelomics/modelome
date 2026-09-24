from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from modelome.config import default_source_catalog
from modelome.http import HttpClient
from modelome.semantic_scholar_citations import SemanticScholarCitationGraphAdapter
from modelome.sources.acl_anthology import AclAnthologySourceAdapter
from modelome.sources.admet_ai_checkpoint_registry import ADMETAICheckpointRegistrySourceAdapter
from modelome.sources.aggregator_registry import OpenMLFlowRegistrySourceAdapter
from modelome.sources.allennlp_model_archives import AllenNLPModelArchiveSourceAdapter
from modelome.sources.alphachip_rl_checkpoint import AlphaChipRlCheckpointAdapter
from modelome.sources.alphafold_registry import AlphaFoldParameterArchiveSourceAdapter
from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.arxiv_snapshot import ArxivCompleteSnapshotSourceAdapter
from modelome.sources.astronn_gaia_release import AstroNNGaiaReleaseSourceAdapter
from modelome.sources.atari_pb_checkpoints import AtariPbCheckpointAdapter
from modelome.sources.audio_extra import CoquiTtsRegistrySourceAdapter
from modelome.sources.aws_bedrock_region_matrix import AwsBedrockRegionMatrixAdapter
from modelome.sources.aws_sagemaker_jumpstart_versions import (
    AwsSageMakerJumpStartVersionsSourceAdapter,
)
from modelome.sources.base import SourceAdapter
from modelome.sources.bfl_api_models import BFLAPIModelsSourceAdapter
from modelome.sources.bioimageio import BioImageIoSourceAdapter
from modelome.sources.biomedical_registry import StarDistPretrainedRegistrySourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.biorxiv_jats_supplementary import (
    BioRxivJatsSupplementSourceAdapter,
)
from modelome.sources.bpemb_registry import BPEmbPretrainedVectorRegistrySourceAdapter
from modelome.sources.cellpose_registry import CellposeRegistrySourceAdapter
from modelome.sources.cgschnet_pretrained_bundle import CGSchNetPretrainedBundleSourceAdapter
from modelome.sources.chai1_components import Chai1ComponentRegistryAdapter
from modelome.sources.chem_ml_extra import (
    ChempropCheMeleonCheckpointSourceAdapter,
    UniMofCheckpointSourceAdapter,
)
from modelome.sources.civitai import CivitaiModelsSourceAdapter
from modelome.sources.cloud_extra import OciGenerativeAIModelCatalog
from modelome.sources.cloudflare_workers_ai_deprecations import CloudflareWorkersAIDeprecations
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter
from modelome.sources.conceptnet_numberbatch import ConceptNetNumberbatchSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.csv_source import CsvSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.deepchem_checkpoint import DeepChemMol2VecCheckpointSourceAdapter
from modelome.sources.detectron2_model_zoo import Detectron2ModelZooSourceAdapter
from modelome.sources.dgl_core_tutorial_checkpoint import DGLCoreTutorialCheckpointSourceAdapter
from modelome.sources.dgl_lifesci_registry import DglLifeSciCheckpointRegistrySourceAdapter
from modelome.sources.dipy_registry import DipyPretrainedRegistrySourceAdapter
from modelome.sources.dopamine_checkpoint_bundles import DopamineCheckpointBundleAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.esa_fm4cs import EsaFm4csSourceAdapter
from modelome.sources.espnet_model_zoo import EspnetModelZooSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.fairchem_omat24_checkpoints import FairChemOMat24CheckpointSourceAdapter
from modelome.sources.fairchem_uma_checkpoints import FairChemUMACheckpointSourceAdapter
from modelome.sources.fairseq_language_models import FairseqPretrainedLanguageModelSourceAdapter
from modelome.sources.fengwu_checkpoint_registry import FengWuCheckpointRegistrySourceAdapter
from modelome.sources.fourcastnet_checkpoint_registry import (
    FourCastNetCheckpointRegistrySourceAdapter,
)
from modelome.sources.fs_mol_checkpoints import FSMolCheckpointSourceAdapter
from modelome.sources.galaxea_vla_checkpoints import GalaxeaVLACheckpointSourceAdapter
from modelome.sources.generative_extra import (
    CompVisLatentDiffusionDownloadsSourceAdapter,
    CompVisLatentDiffusionReadmeDownloadsSourceAdapter,
    CompVisStableDiffusionFirstStagesSourceAdapter,
)
from modelome.sources.gensim_registry import GensimDownloaderModelRegistrySourceAdapter
from modelome.sources.geospatial_registry import GeospatialRegistrySourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.github_historical_release_assets import (
    GitHubHistoricalReleaseAssetsSourceAdapter,
)
from modelome.sources.github_repositories import GitHubPublicRepositoriesSourceAdapter
from modelome.sources.gitlab_release_assets import GitLabPublicReleaseAssetsSourceAdapter
from modelome.sources.google_bert_checkpoints import GoogleResearchBertCheckpointSourceAdapter
from modelome.sources.google_football_checkpoints import GoogleFootballCheckpointAdapter
from modelome.sources.google_graphcast_checkpoint_inventory import (
    GoogleGraphCastCheckpointInventorySourceAdapter,
)
from modelome.sources.gpt4all_model_catalog import Gpt4AllModelCatalogSourceAdapter
from modelome.sources.graph_ml_registry import GraphMLRegistrySourceAdapter
from modelome.sources.graphgps_release_asset import GraphGPSReleaseAssetSourceAdapter
from modelome.sources.graphormer_checkpoint_registry import (
    GraphormerCheckpointRegistrySourceAdapter,
)
from modelome.sources.groundingdino_checkpoints import GroundingDINOCheckpointSourceAdapter
from modelome.sources.grover_registry import GroverCheckpointRegistrySourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter
from modelome.sources.huggingface import HuggingFaceSourceAdapter
from modelome.sources.jax_extra_registry import JaxExtraRegistrySourceAdapter
from modelome.sources.jax_registry import JaxRegistrySourceAdapter
from modelome.sources.json_catalog import JsonCatalogSourceAdapter
from modelome.sources.kaggle import KaggleModelsSourceAdapter
from modelome.sources.kaldi_model_index import KaldiModelIndexSourceAdapter
from modelome.sources.keras_hub_preset_registry import KerasHubPresetRegistrySourceAdapter
from modelome.sources.lerobot_molmoact2_relation import (
    LeRobotMolmoAct2RelationSourceAdapter,
)
from modelome.sources.lerobot_pi05_libero_relation import LeRobotPi05LiberoRelationSourceAdapter
from modelome.sources.line_checkpoint_card_catalog import (
    LineCheckpointCardCatalogSourceAdapter,
)
from modelome.sources.mace_foundation_registry import (
    MaceFoundationCheckpointRegistrySourceAdapter,
)
from modelome.sources.mace_omol_checkpoint import MaceOmolCheckpointSourceAdapter
from modelome.sources.mace_registry import MaceOff23CheckpointRegistrySourceAdapter
from modelome.sources.markdown_checkpoint_list import MarkdownCheckpointListSourceAdapter
from modelome.sources.markdown_model_card_list import MarkdownModelCardListSourceAdapter
from modelome.sources.markdown_model_table import MarkdownModelTableSourceAdapter
from modelome.sources.mediapipe_model_catalog import MediaPipeModelCatalogSourceAdapter
from modelome.sources.meta_sam3_checkpoints import MetaSAM3CheckpointSourceAdapter
from modelome.sources.microsoft_aurora_checkpoints import MicrosoftAuroraCheckpointSourceAdapter
from modelome.sources.mindspore_registry import MindSporeModelZooSourceAdapter
from modelome.sources.modelscope import ModelScopeModelsSourceAdapter
from modelome.sources.molecular_registry import OpenFoldCheckpointRegistrySourceAdapter
from modelome.sources.moler_checkpoint import MoLeRCheckpointSourceAdapter
from modelome.sources.molmoact2_checkpoints import MolmoAct2CheckpointSourceAdapter
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter
from modelome.sources.nemo_checkpoints import NemoCheckpointCatalogSourceAdapter
from modelome.sources.neuralgcm_checkpoint_registry import (
    NeuralGCMCheckpointRegistrySourceAdapter,
)
from modelome.sources.ngc import NgcModelsSourceAdapter
from modelome.sources.ngc_cli_versions import NgcCliModelVersionsSourceAdapter
from modelome.sources.nltk_data_models import NltkDataModelIndexSourceAdapter
from modelome.sources.nnunet_registry import NnUNetV1PretrainedRegistryAdapter
from modelome.sources.nnunet_zenodo_bundles import NnUNetZenodoBundleRegistryAdapter
from modelome.sources.nvidia_cosmos3_checkpoints import NvidiaCosmos3CheckpointSourceAdapter
from modelome.sources.nvidia_earth2 import NvidiaEarth2SourceAdapter
from modelome.sources.nvidia_groot_n17_checkpoints import NvidiaGR00TN17CheckpointSourceAdapter
from modelome.sources.ocp_model_registry import OCPModelRegistrySourceAdapter
from modelome.sources.octo_checkpoints import OctoCheckpointSourceAdapter
from modelome.sources.ollama_library_tags import OllamaLibraryTagCatalogAdapter
from modelome.sources.onnx_model_zoo import OnnxModelZooSourceAdapter
from modelome.sources.open_x_rt1x_checkpoint import OpenXRT1XCheckpointSourceAdapter
from modelome.sources.openai_gpt2_checkpoints import OpenAIGPT2CheckpointSourceAdapter
from modelome.sources.openai_models import OpenAIModelsSourceAdapter
from modelome.sources.openaire import OpenAireGraphSourceAdapter
from modelome.sources.openalex import OpenAlexSourceAdapter
from modelome.sources.openclip_pretrained_registry import OpenCLIPPretrainedRegistrySourceAdapter
from modelome.sources.opencsg import OpenCsgModelsSourceAdapter
from modelome.sources.opencv_dnn_model_index import OpenCVDnnModelIndexSourceAdapter
from modelome.sources.openfold3_registry import OpenFold3ParameterRegistryAdapter
from modelome.sources.openmmlab import OpenMMLabModelIndexSourceAdapter
from modelome.sources.openpi_checkpoint_manifest import OpenPiCheckpointManifestAdapter
from modelome.sources.openreview import OpenReviewSourceAdapter
from modelome.sources.openrouter import OpenRouterModelsSourceAdapter
from modelome.sources.openvino_model_zoo import OpenVinoModelZooSourceAdapter
from modelome.sources.openvla_checkpoints import OpenVLACheckpointSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.sources.paddle_detection_model_zoo import PaddleDetectionModelZooSourceAdapter
from modelome.sources.paddle_model_center import PaddleModelCenterSourceAdapter
from modelome.sources.paddleclas_model_registry import PaddleClasModelRegistrySourceAdapter
from modelome.sources.paddlegan_tutorial_model_zoo import (
    PaddleGanTutorialModelZooSourceAdapter,
)
from modelome.sources.paddlehelix_gem_checkpoint import PaddleHelixGemCheckpointSourceAdapter
from modelome.sources.paddlenlp_taskflow_knowledge_mining import (
    PaddleNlpTaskflowKnowledgeMiningSourceAdapter,
)
from modelome.sources.paddlenlp_taskflow_sentiment import PaddleNlpTaskflowSentimentSourceAdapter
from modelome.sources.paddlenlp_taskflow_text_correction import (
    PaddleNlpTaskflowTextCorrectionSourceAdapter,
)
from modelome.sources.paddlenlp_taskflow_text_similarity import (
    PaddleNlpTaskflowTextSimilaritySourceAdapter,
)
from modelome.sources.paddlenlp_taskflow_uie import PaddleNlpTaskflowUieSourceAdapter
from modelome.sources.paddleocr_current_model_list import (
    PaddleOcrCurrentModelListSourceAdapter,
)
from modelome.sources.paddlerec_catalog import PaddleRecCatalogSourceAdapter
from modelome.sources.paddlespeech_ssl_manifest import PaddleSpeechSslManifestSourceAdapter
from modelome.sources.paddlex_model_list import PaddleXModelListSourceAdapter
from modelome.sources.pangu_weather_checkpoint_registry import (
    PanguWeatherCheckpointRegistrySourceAdapter,
)
from modelome.sources.paperswithcode import (
    PapersWithCodeEvaluationMethodsSourceAdapter,
    PapersWithCodeLinksSourceAdapter,
    PapersWithCodeValidatedMethodsSourceAdapter,
)
from modelome.sources.pelican_vla_checkpoint_registry import (
    PelicanVLACheckpointRegistrySourceAdapter,
)
from modelome.sources.plos import PlosSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.sources.proteinmpnn import ProteinMpnSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.sources.pyg_dimenet_checkpoints import PyGDimeNetCheckpointSourceAdapter
from modelome.sources.pyg_gpse_registry import PyGGPSECheckpointRegistrySourceAdapter
from modelome.sources.pyg_schnet_qm9_registry import PyGSchNetQM9RegistrySourceAdapter
from modelome.sources.pytorch_hub_load_calls import PyTorchHubLoadCallSourceAdapter
from modelome.sources.qualcomm_ai_hub_models import QualcommAIHubModelsSourceAdapter
from modelome.sources.replicate import ReplicateModelsSourceAdapter
from modelome.sources.rfdiffusion_registry import RFDiffusionCheckpointSourceAdapter
from modelome.sources.rl_checkpoint_indexes import RlClarityCheckpointIndexAdapter
from modelome.sources.rl_checkpoints_extra import DiffusionPolicyCheckpointIndexAdapter
from modelome.sources.robotics_extra import ArgusCheckpointInventorySourceAdapter
from modelome.sources.robotics_registry_v3 import RoboticsTransformerCheckpointSourceAdapter
from modelome.sources.rosettafold_checkpoints import RoseTTAFoldCheckpointAdapter
from modelome.sources.satmae_checkpoint_registry import SatMAECheckpointRegistrySourceAdapter
from modelome.sources.sdss_ssl_checkpoints import SdssSslCheckpointsSourceAdapter
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter
from modelome.sources.sherpa_audio_tagging import SherpaAudioTaggingSourceAdapter
from modelome.sources.sherpa_source_separation import SherpaSourceSeparationSourceAdapter
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter
from modelome.sources.spacy_models import SpacyModelsSourceAdapter
from modelome.sources.stanza_resources import StanzaResourcesSourceAdapter
from modelome.sources.static_json_checkpoint_registry import (
    StaticJsonCheckpointRegistrySourceAdapter,
)
from modelome.sources.static_python_checkpoint_registry import (
    StaticPythonCheckpointRegistrySourceAdapter,
)
from modelome.sources.tensorflow_audioset_checkpoints import (
    TensorFlowAudioSetCheckpointSourceAdapter,
)
from modelome.sources.tensorflow_garden import TensorFlowGardenSourceAdapter
from modelome.sources.tensorflow_tpu_efficientnet import TensorFlowTPUEfficientNetSourceAdapter
from modelome.sources.timm_model_registry import TimmModelRegistrySourceAdapter
from modelome.sources.torch_hub_extra import TorchHubListingSourceAdapter
from modelome.sources.torchaudio_pipeline_registry import (
    TorchaudioPipelineRegistrySourceAdapter,
)
from modelome.sources.torchgeo_weight_registry import TorchGeoWeightRegistrySourceAdapter
from modelome.sources.torchvision_weight_registry import (
    TorchvisionWeightRegistrySourceAdapter,
)
from modelome.sources.torchxrayvision_registry import TorchXRayVisionRegistrySourceAdapter
from modelome.sources.ultralytics_release_checkpoints import (
    UltralyticsReleaseCheckpointSourceAdapter,
)
from modelome.sources.unimol_checkpoint import UniMolCheckpointSourceAdapter
from modelome.sources.vq_diffusion import MicrosoftVqDiffusionCheckpointManifestSourceAdapter
from modelome.sources.weathernext2_checkpoint_registry import (
    WeatherNext2CheckpointRegistrySourceAdapter,
)
from modelome.sources.wenet_model_zoo import WenetPretrainedModelSourceAdapter
from modelome.sources.yolox_model_zoo import YOLOXModelZooSourceAdapter
from modelome.sources.zenodo import ZenodoModelRecordsSourceAdapter
from modelome.sources.zenodo_oai_candidates import ZenodoOaiModelCandidatesSourceAdapter

Clock = Callable[[], datetime]
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)}")


def load_source_configs(path: str | Path | None = None) -> tuple[dict[str, Any], ...]:
    """Load and minimally validate source tables without constructing clients."""

    return _load_catalog_configs(path, section="source")


def load_benchmark_configs(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load benchmark tables without exposing them as ingestion sources."""

    return _load_catalog_configs(path, section="benchmark")


def _load_catalog_configs(
    path: str | Path | None,
    *,
    section: str,
) -> tuple[dict[str, Any], ...]:
    config_path = default_source_catalog() if path is None else Path(path).expanduser()
    with config_path.open("rb") as handle:
        document = tomllib.load(handle)
    raw_entries = document.get(section, [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"{config_path}: expected zero or more [[{section}]] tables")

    result = []
    names: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{config_path}: {section} entry {index} is not a table")
        entry = dict(raw_entry)
        name = _required_text(entry, "name", section=section)
        _required_text(entry, "adapter", section=section)
        if name in names:
            raise ValueError(f"{config_path}: duplicate {section} name {name!r}")
        names.add(name)
        entry["_catalog_role"] = section
        result.append(entry)

    other_section = "benchmark" if section == "source" else "source"
    other_entries = document.get(other_section, [])
    if not isinstance(other_entries, list):
        raise ValueError(f"{config_path}: expected zero or more [[{other_section}]] tables")
    other_names = {
        _required_text(entry, "name", section=other_section)
        for entry in other_entries
        if isinstance(entry, Mapping)
    }
    if collisions := sorted(names & other_names):
        raise ValueError(
            f"{config_path}: names cannot be both sources and benchmarks: " + ", ".join(collisions)
        )
    return tuple(result)


def create_source(
    config: Mapping[str, Any],
    *,
    client: HttpClient | Any | None = None,
    clock: Clock | None = None,
    environ: Mapping[str, str] | None = None,
) -> SourceAdapter:
    """Construct one adapter from a source table and environment credentials."""

    if config.get("_catalog_role") == "benchmark":
        raise ValueError("benchmark configuration cannot be used as an ingestion source")
    environment = os.environ if environ is None else environ
    expanded = _expand_environment(dict(config), environment)
    name = _required_text(expanded, "name")
    adapter = _required_text(expanded, "adapter").casefold()
    injected: dict[str, Any] = {"client": client or HttpClient()}
    if clock is not None:
        injected["clock"] = clock

    if adapter == "semantic_scholar_citation_graph":
        return SemanticScholarCitationGraphAdapter(
            name=name,
            paper_id=_required_text(expanded, "paper_id"),
            paper_url=_required_text(expanded, "paper_url"),
            paper_title=_required_text(expanded, "paper_title"),
            direction=_text(expanded.get("direction")) or "references",
            page_size=_integer(expanded.get("page_size"), 1_000),
            max_records=_integer(expanded.get("max_records"), 9_999),
            api_key=_credential(expanded, environment, defaults=("S2_API_KEY",)) or None,
            url=_text(expanded.get("url")) or "https://api.semanticscholar.org/graph/v1",
            client=injected["client"],
        )

    if adapter == "pubmed_bulk":
        return PubMedBulkSourceAdapter(
            name=name,
            baseline_url=_required_text(expanded, "baseline_url"),
            update_url=_required_text(expanded, "update_url"),
            page_size=_integer(expanded.get("page_size"), 25),
            **injected,
        )

    if adapter in {"commoncrawl_wet", "commoncrawl-wet"}:
        return CommonCrawlWetSourceAdapter(
            name=name,
            catalog_url=_required_text(expanded, "catalog_url"),
            data_url=_required_text(expanded, "data_url"),
            page_size=_integer(expanded.get("page_size"), 1_000),
            manifests_per_page=_integer(expanded.get("manifests_per_page"), 16),
            max_catalog_bytes=_integer(expanded.get("max_catalog_bytes"), 8 * 1024 * 1024),
            max_collections=_integer(expanded.get("max_collections"), 10_000),
            max_manifest_compressed_bytes=_integer(
                expanded.get("max_manifest_compressed_bytes"), 64 * 1024 * 1024
            ),
            max_manifest_bytes=_integer(expanded.get("max_manifest_bytes"), 512 * 1024 * 1024),
            max_manifest_shards=_integer(expanded.get("max_manifest_shards"), 2_000_000),
            max_path_bytes=_integer(expanded.get("max_path_bytes"), 16 * 1024),
            **injected,
        )

    if adapter in {"openaire_graph", "openaire-graph"}:
        return OpenAireGraphSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            concept_record_id=(_text(expanded.get("concept_record_id")) or "3516917"),
            page_size=_integer(expanded.get("page_size"), 25),
            max_manifest_bytes=_integer(expanded.get("max_manifest_bytes"), 8 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 10_000),
            max_file_bytes=_integer(expanded.get("max_file_bytes"), 1 << 40),
            **injected,
        )

    if adapter == "openreview":
        return OpenReviewSourceAdapter(
            name=name,
            artifact_source=_text(expanded.get("artifact_source")) or name,
            api_v1_url=_required_text(expanded, "api_v1_url"),
            api_v2_url=_required_text(expanded, "api_v2_url"),
            web_base_url=_required_text(expanded, "web_base_url"),
            page_size=_integer(expanded.get("page_size"), 1_000),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            consistency_lag_seconds=_nonnegative_integer(
                expanded.get("consistency_lag_seconds"), 300
            ),
            include_revision_details=_boolean(
                expanded.get("include_revision_details"), default=True
            ),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 32 * 1024 * 1024),
            max_item_bytes=_integer(expanded.get("max_item_bytes"), 8 * 1024 * 1024),
            max_text_chars=_integer(expanded.get("max_text_chars"), 4 * 1024 * 1024),
            max_links=_integer(expanded.get("max_links"), 2_048),
            **injected,
        )

    if adapter in {"bioimageio", "bioimage_io"}:
        return BioImageIoSourceAdapter(
            name=name,
            artifact_source=_text(expanded.get("artifact_source")) or name,
            index_url=_required_text(expanded, "index_url"),
            artifact_base_url=_required_text(expanded, "artifact_base_url"),
            workspace=_text(expanded.get("workspace")) or "bioimage-io",
            page_size=_integer(expanded.get("page_size"), 20),
            max_index_bytes=_integer(expanded.get("max_index_bytes"), 16 * 1024 * 1024),
            max_artifact_bytes=_integer(expanded.get("max_artifact_bytes"), 16 * 1024 * 1024),
            max_rdf_bytes=_integer(expanded.get("max_rdf_bytes"), 16 * 1024 * 1024),
            max_items=_integer(expanded.get("max_items"), 100_000),
            max_versions_per_model=_integer(expanded.get("max_versions_per_model"), 10_000),
            max_links=_integer(expanded.get("max_links"), 4_096),
            max_text_chars=_integer(expanded.get("max_text_chars"), 8 * 1024 * 1024),
            **injected,
        )

    if adapter in {"onnx_model_zoo", "onnx-model-zoo"}:
        return OnnxModelZooSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            repository_url=_required_text(expanded, "repository_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter in {"openmmlab_model_index", "openmmlab-model-index"}:
        return OpenMMLabModelIndexSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_manifests=_integer(expanded.get("max_manifests"), 1_000),
            max_index_bytes=_integer(expanded.get("max_index_bytes"), 2 * 1024 * 1024),
            max_manifest_bytes=_integer(expanded.get("max_manifest_bytes"), 4 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 100_000),
            **injected,
        )

    if adapter in {"openvino_model_zoo", "openvino-model-zoo"}:
        return OpenVinoModelZooSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "openvinotoolkit/open_model_zoo"),
            branch=_text(expanded.get("branch")) or "master",
            max_tree_bytes=_integer(
                expanded.get("max_tree_bytes"),
                16 * 1024 * 1024,
            ),
            max_manifest_bytes=_integer(
                expanded.get("max_manifest_bytes"),
                1 * 1024 * 1024,
            ),
            max_manifests=_integer(expanded.get("max_manifests"), 10_000),
            **injected,
        )

    if adapter in {"timm_model_registry", "timm-model-registry"}:
        return TimmModelRegistrySourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "huggingface/pytorch-image-models"),
            branch=_text(expanded.get("branch")) or "main",
            max_archive_bytes=_integer(
                expanded.get("max_archive_bytes"),
                64 * 1024 * 1024,
            ),
            max_module_bytes=_integer(
                expanded.get("max_module_bytes"),
                2 * 1024 * 1024,
            ),
            max_modules=_integer(expanded.get("max_modules"), 1_000),
            **injected,
        )

    if adapter in {"torchvision_weight_registry", "torchvision-weight-registry"}:
        return TorchvisionWeightRegistrySourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "pytorch/vision",
            branch=_text(expanded.get("branch")) or "main",
            package_path=(_text(expanded.get("package_path")) or "torchvision/models"),
            max_archive_bytes=_integer(
                expanded.get("max_archive_bytes"),
                64 * 1024 * 1024,
            ),
            max_module_bytes=_integer(
                expanded.get("max_module_bytes"),
                2 * 1024 * 1024,
            ),
            max_modules=_integer(expanded.get("max_modules"), 1_000),
            **injected,
        )

    if adapter == "torchgeo_weight_registry":
        return TorchGeoWeightRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            package_path=_required_text(expanded, "package_path"),
            max_archive_bytes=_integer(expanded.get("max_archive_bytes"), 64 * 1024 * 1024),
            max_module_bytes=_integer(expanded.get("max_module_bytes"), 2 * 1024 * 1024),
            max_modules=_integer(expanded.get("max_modules"), 1_000),
            **injected,
        )

    if adapter in {"torchaudio_pipeline_registry", "torchaudio-pipeline-registry"}:
        return TorchaudioPipelineRegistrySourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "pytorch/audio",
            branch=_text(expanded.get("branch")) or "main",
            pipeline_root=(_text(expanded.get("pipeline_root")) or "src/torchaudio/pipelines"),
            max_archive_bytes=_integer(
                expanded.get("max_archive_bytes"),
                128 * 1024 * 1024,
            ),
            max_module_bytes=_integer(
                expanded.get("max_module_bytes"),
                2 * 1024 * 1024,
            ),
            max_modules=_integer(expanded.get("max_modules"), 128),
            max_pipelines=_integer(expanded.get("max_pipelines"), 10_000),
            **injected,
        )

    if adapter in {"keras_hub_preset_registry", "keras-hub-preset-registry"}:
        return KerasHubPresetRegistrySourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "keras-team/keras-hub",
            branch=_text(expanded.get("branch")) or "master",
            preset_root=(_text(expanded.get("preset_root")) or "keras_hub/src/models"),
            max_archive_bytes=_integer(
                expanded.get("max_archive_bytes"),
                64 * 1024 * 1024,
            ),
            max_file_bytes=_integer(
                expanded.get("max_file_bytes"),
                2 * 1024 * 1024,
            ),
            max_files=_integer(expanded.get("max_files"), 5_000),
            max_presets=_integer(expanded.get("max_presets"), 100_000),
            **injected,
        )

    if adapter in {"paddle_model_center", "paddle-model-center"}:
        return PaddleModelCenterSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/models"),
            branch=_text(expanded.get("branch")) or "release/2.4",
            max_tree_bytes=_integer(
                expanded.get("max_tree_bytes"),
                32 * 1024 * 1024,
            ),
            max_info_bytes=_integer(
                expanded.get("max_info_bytes"),
                1 * 1024 * 1024,
            ),
            max_download_bytes=_integer(
                expanded.get("max_download_bytes"),
                4 * 1024 * 1024,
            ),
            max_families=_integer(expanded.get("max_families"), 1_000),
            max_download_files=_integer(
                expanded.get("max_download_files"),
                2_000,
            ),
            **injected,
        )

    if adapter in {"paddleclas_model_registry", "paddleclas-model-registry"}:
        return PaddleClasModelRegistrySourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/PaddleClas"),
            branch=_text(expanded.get("branch")) or "release/2.6",
            source_path=_text(expanded.get("source_path")) or "paddleclas.py",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter in {
        "paddleocr_current_model_list",
        "paddleocr-current-model-list",
    }:
        return PaddleOcrCurrentModelListSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/PaddleOCR"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=(_text(expanded.get("source_path")) or "docs/version3.x/model_list.md"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            max_anchors=_integer(expanded.get("max_anchors"), 100_000),
            max_tables=_integer(expanded.get("max_tables"), 10_000),
            max_cells=_integer(expanded.get("max_cells"), 100_000),
            max_text_chars=_integer(expanded.get("max_text_chars"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter in {"paddlerec_catalog", "paddlerec-algorithm-catalog"}:
        return PaddleRecCatalogSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/PaddleRec"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_text(expanded.get("source_path")) or "README_EN.md",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter in {"paddlegan_tutorial_model_zoo", "paddlegan-tutorial-model-zoo"}:
        return PaddleGanTutorialModelZooSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/PaddleGAN"),
            branch=_text(expanded.get("branch")) or "develop",
            document_prefix=(_text(expanded.get("document_prefix")) or "docs/en_US/tutorials/"),
            max_archive_bytes=_integer(expanded.get("max_archive_bytes"), 128 * 1024 * 1024),
            max_document_bytes=_integer(expanded.get("max_document_bytes"), 2 * 1024 * 1024),
            max_total_document_bytes=_integer(
                expanded.get("max_total_document_bytes"), 64 * 1024 * 1024
            ),
            max_documents=_integer(expanded.get("max_documents"), 256),
            max_models=_integer(expanded.get("max_models"), 100_000),
            **injected,
        )

    if adapter in {
        "paddledetection_model_zoo",
        "paddledetection-model-zoo",
        "paddle_project_model_zoo",
        "paddle-project-model-zoo",
        "archive_markdown_checkpoint_zoo",
        "archive-markdown-checkpoint-zoo",
    }:
        document_paths = expanded.get("document_paths", ())
        if not isinstance(document_paths, list | tuple):
            raise ValueError(f"{name}: document_paths must be an array")
        return PaddleDetectionModelZooSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "PaddlePaddle/PaddleDetection"),
            branch=_text(expanded.get("branch")) or "release/2.9",
            project_name=_text(expanded.get("project_name")) or "PaddleDetection",
            provider_namespace=(_text(expanded.get("provider_namespace")) or "paddledetection"),
            document_prefix=_text(expanded.get("document_prefix")) or "configs/",
            document_suffix=_text(expanded.get("document_suffix")) or "README.md",
            document_paths=document_paths,
            max_archive_bytes=_integer(expanded.get("max_archive_bytes"), 128 * 1024 * 1024),
            max_document_bytes=_integer(expanded.get("max_document_bytes"), 2 * 1024 * 1024),
            max_total_document_bytes=_integer(
                expanded.get("max_total_document_bytes"), 64 * 1024 * 1024
            ),
            max_documents=_integer(expanded.get("max_documents"), 256),
            max_models=_integer(expanded.get("max_models"), 100_000),
            **injected,
        )

    if adapter in {"espnet_model_zoo", "espnet-model-zoo"}:
        return EspnetModelZooSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "espnet/espnet_model_zoo"),
            branch=_text(expanded.get("branch")) or "master",
            table_path=(_text(expanded.get("table_path")) or "espnet_model_zoo/table.csv"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_rows=_integer(expanded.get("max_rows"), 10_000),
            **injected,
        )

    if adapter in {"detectron2_model_zoo", "detectron2-model-zoo"}:
        return Detectron2ModelZooSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "facebookresearch/detectron2"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=(_text(expanded.get("source_path")) or "detectron2/model_zoo/model_zoo.py"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter in {
        "static_json_checkpoint_registry",
        "static-json-checkpoint-registry",
    }:
        return StaticJsonCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter == "openai_gpt2_checkpoints":
        return OpenAIGPT2CheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "fourcastnet_checkpoint_registry":
        return FourCastNetCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 2),
            **injected,
        )

    if adapter == "pyg_schnet_qm9_registry":
        return PyGSchNetQM9RegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "cgschnet_pretrained_bundle":
        return CGSchNetPretrainedBundleSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "microsoft_aurora_checkpoints":
        return MicrosoftAuroraCheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "lerobot_pi05_libero_relation":
        return LeRobotPi05LiberoRelationSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "rosettafold_checkpoint_bundles":
        return RoseTTAFoldCheckpointAdapter(
            name=name,
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 256 * 1024),
            client=injected["client"],
        )

    if adapter == "nnunet_zenodo_bundle_registry":
        return NnUNetZenodoBundleRegistryAdapter(
            name=name,
            record_id=_required_text(expanded, "record_id"),
            max_record_bytes=_integer(expanded.get("max_record_bytes"), 4 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 500),
            **injected,
        )

    if adapter == "google_graphcast_checkpoint_inventory":
        return GoogleGraphCastCheckpointInventorySourceAdapter(
            name=name,
            page_size=_integer(expanded.get("page_size"), 1_000),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "openpi_checkpoint_manifest":
        return OpenPiCheckpointManifestAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 12),
            **injected,
        )

    if adapter == "meta_sam3_checkpoints":
        return MetaSAM3CheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "fairchem_omat24_checkpoints":
        return FairChemOMat24CheckpointSourceAdapter(
            name=name,
            page_url=_required_text(expanded, "page_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "fairchem_uma_checkpoints":
        return FairChemUMACheckpointSourceAdapter(
            name=name,
            page_url=_required_text(expanded, "page_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "nvidia_cosmos3_checkpoints":
        return NvidiaCosmos3CheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 12),
            **injected,
        )

    if adapter == "ultralytics_release_checkpoints":
        model_docs = expanded.get("model_docs", ())
        if not isinstance(model_docs, list | tuple):
            raise ValueError(f"{name}: model_docs must be an array")
        return UltralyticsReleaseCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            assets_repository=_required_text(expanded, "assets_repository"),
            branch=_text(expanded.get("branch")) or "main",
            model_docs=model_docs,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_releases=_integer(expanded.get("max_releases"), 500),
            max_assets_per_release=_integer(expanded.get("max_assets_per_release"), 1_000),
            client=injected["client"],
        )

    if adapter == "chai1_component_registry":
        return Chai1ComponentRegistryAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 512 * 1024),
            max_components=_integer(expanded.get("max_components"), 64),
            client=injected["client"],
        )

    if adapter == "weathernext2_checkpoint_registry":
        return WeatherNext2CheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 18),
            **injected,
        )

    if adapter == "cloudflare_workers_ai_deprecations":
        return CloudflareWorkersAIDeprecations(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 200),
            client=injected["client"],
        )

    if adapter in {
        "static_python_checkpoint_registry",
        "static-python-checkpoint-registry",
    }:
        return StaticPythonCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            mapping_variable=_required_text(expanded, "mapping_variable"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter in {"geospatial_registry", "geospatial-registry"}:
        return GeospatialRegistrySourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "NASA-IMPACT/Prithvi-EO-2.0",
            branch=_text(expanded.get("branch")) or "main",
            source_path=_text(expanded.get("source_path")) or "README.md",
            provider_namespace=_text(expanded.get("provider_namespace")) or "nasa-prithvi:model",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter in {
        "line_checkpoint_card_catalog",
        "line-checkpoint-card-catalog",
    }:
        return LineCheckpointCardCatalogSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter in {"markdown_checkpoint_list", "markdown-checkpoint-list"}:
        return MarkdownCheckpointListSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            document_path=_required_text(expanded, "document_path"),
            section_heading_pattern=_required_text(expanded, "section_heading_pattern"),
            model_handle_pattern=_required_text(expanded, "model_handle_pattern"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter in {"markdown_model_card_list", "markdown-model-card-list"}:
        return MarkdownModelCardListSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            document_path=_required_text(expanded, "document_path"),
            section_heading_pattern=_required_text(expanded, "section_heading_pattern"),
            model_url_pattern=_required_text(expanded, "model_url_pattern"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter in {"allennlp_model_archives", "allennlp-model-archives"}:
        return AllenNLPModelArchiveSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            modelcards_path=_required_text(expanded, "modelcards_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter in {"markdown_model_table", "markdown-model-table"}:
        excluded_headings = expanded.get("excluded_headings", ())
        if not isinstance(excluded_headings, list | tuple):
            raise ValueError(f"{name}: excluded_headings must be an array")
        raw_dataset_column = expanded.get("dataset_column")
        raw_identity_context_columns = expanded.get("identity_context_columns")
        if raw_identity_context_columns is not None and not isinstance(
            raw_identity_context_columns, list | tuple
        ):
            raise ValueError(f"{name}: identity_context_columns must be an array")
        return MarkdownModelTableSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            document_path=_required_text(expanded, "document_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            model_column=_nonnegative_integer(expanded.get("model_column"), 0),
            model_header_pattern=expanded.get("model_header_pattern"),
            model_name_pattern=expanded.get("model_name_pattern"),
            dataset_column=(
                None if raw_dataset_column is None else _nonnegative_integer(raw_dataset_column, 0)
            ),
            identity_context_columns=(
                None
                if raw_identity_context_columns is None
                else tuple(
                    _nonnegative_integer(column, 0) for column in raw_identity_context_columns
                )
            ),
            identity_include_heading=_boolean(
                expanded.get("identity_include_heading"), default=True
            ),
            checkpoint_row_pattern=expanded.get("checkpoint_row_pattern"),
            excluded_headings=excluded_headings,
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                8 * 1024 * 1024,
            ),
            max_rows=_integer(expanded.get("max_rows"), 100_000),
            **injected,
        )

    if adapter in {"openrouter_models", "openrouter-models"}:
        return OpenRouterModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            model_page_base_url=(
                _text(expanded.get("model_page_base_url")) or "https://openrouter.ai"
            ),
            output_modalities=_text(expanded.get("output_modalities")) or None,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter in {"replicate_models", "replicate-models"}:
        token = _credential(expanded, environment, defaults=("REPLICATE_API_TOKEN",))
        if not token:
            raise ValueError(f"{name}: Replicate API token is required")
        return ReplicateModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            model_page_base_url=(
                _text(expanded.get("model_page_base_url")) or "https://replicate.com"
            ),
            token=token,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter in {"openai_models", "openai-models"}:
        token = _credential(expanded, environment, defaults=("OPENAI_API_KEY",))
        if not token:
            raise ValueError(f"{name}: OpenAI API key is required")
        public_owners = expanded.get("public_owners", ("openai", "system"))
        if not isinstance(public_owners, list | tuple):
            raise ValueError(f"{name}: public_owners must be an array")
        return OpenAIModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            token=token,
            public_owners=tuple(str(owner) for owner in public_owners),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 10_000),
            **injected,
        )

    if adapter in {"monai_model_zoo", "monai-model-zoo"}:
        return MonaiModelZooSourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "Project-MONAI/model-zoo",
            branch=_text(expanded.get("branch")) or "dev",
            registry_path=(_text(expanded.get("registry_path")) or "models/model_info.json"),
            max_registry_bytes=_integer(expanded.get("max_registry_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter in {"spacy_models", "spacy-models"}:
        return SpacyModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 10_000),
            max_releases=_integer(expanded.get("max_releases"), 100_000),
            **injected,
        )

    if adapter in {"stanza_resources", "stanza-resources"}:
        return StanzaResourcesSourceAdapter(
            name=name,
            repository=(_text(expanded.get("repository")) or "stanfordnlp/stanza-resources"),
            branch=_text(expanded.get("branch")) or "main",
            max_tree_bytes=_integer(expanded.get("max_tree_bytes"), 2 * 1024 * 1024),
            max_manifest_bytes=_integer(expanded.get("max_manifest_bytes"), 2 * 1024 * 1024),
            max_manifests=_integer(expanded.get("max_manifests"), 100),
            max_models_per_manifest=_integer(expanded.get("max_models_per_manifest"), 20_000),
            **injected,
        )

    if adapter in {"gharchive", "gh_archive"}:
        clock_options: dict[str, Any] = {}
        if clock is not None:
            clock_options["clock"] = clock
        return GhArchiveSourceAdapter(
            name=name,
            data_url=_required_text(expanded, "data_url"),
            page_size=_integer(expanded.get("page_size"), 24),
            initial_lookback_hours=_integer(expanded.get("initial_lookback_hours"), 48),
            availability_lag_hours=_nonnegative_integer(expanded.get("availability_lag_hours"), 1),
            **clock_options,
        )

    if adapter in {"github_public_repositories", "github-public-repositories"}:
        token = _credential(expanded, environment, defaults=("GITHUB_TOKEN",))
        return GitHubPublicRepositoriesSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            page_size=_integer(expanded.get("page_size"), 100),
            initial_since=_nonnegative_integer(expanded.get("initial_since"), 0),
            token=token,
            **injected,
        )

    if adapter == "github_historical_release_assets":
        token = _credential(expanded, environment, defaults=("GITHUB_TOKEN",))
        return GitHubHistoricalReleaseAssetsSourceAdapter(
            name=name,
            initial_since=_nonnegative_integer(expanded.get("initial_since"), 0),
            max_repository_id=_integer(expanded.get("max_repository_id"), 0),
            max_repositories=_integer(expanded.get("max_repositories"), 10),
            page_size=_integer(expanded.get("page_size"), 100),
            max_releases_per_repository=_integer(
                expanded.get("max_releases_per_repository"), 10
            ),
            max_assets_per_release=_integer(expanded.get("max_assets_per_release"), 100),
            token=token or None,
            client=injected["client"],
        )

    if adapter == "fs_mol_checkpoints":
        return FSMolCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "graphormer_checkpoint_registry":
        return GraphormerCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            mapping_variable=_required_text(expanded, "mapping_variable"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "molmoact2_checkpoints":
        return MolmoAct2CheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 12),
            **injected,
        )

    if adapter == "paddlenlp_taskflow_text_correction":
        return PaddleNlpTaskflowTextCorrectionSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "paddlespeech_ssl_manifest":
        return PaddleSpeechSslManifestSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            **injected,
        )

    if adapter == "atari_pb_checkpoints":
        return AtariPbCheckpointAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 100),
            **injected,
        )

    if adapter == "qualcomm_ai_hub_models":
        return QualcommAIHubModelsSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_readme_bytes=_integer(expanded.get("max_readme_bytes"), 512 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            client=injected["client"],
        )

    if adapter == "bfl_api_models":
        return BFLAPIModelsSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 512 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            client=injected["client"],
        )

    if adapter == "tensorflow_audioset_checkpoints":
        return TensorFlowAudioSetCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            document_path=_required_text(expanded, "document_path"),
            max_bytes=_integer(expanded.get("max_bytes"), 2 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "dgl_core_tutorial_checkpoint":
        return DGLCoreTutorialCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10),
            **injected,
        )

    if adapter == "unimol_release_checkpoints":
        return UniMolCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "tensorflow_tpu_efficientnet":
        return TensorFlowTPUEfficientNetSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 500),
            client=injected["client"],
        )

    if adapter == "astronn_gaia_release":
        return AstroNNGaiaReleaseSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "esa_fm4cs":
        return EsaFm4csSourceAdapter(
            page_size=_integer(expanded.get("page_size"), 100),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            token=_credential(expanded, environment, defaults=("HF_TOKEN",)) or None,
            max_revision_tree_pages=_integer(expanded.get("max_revision_tree_pages"), 20),
            max_revision_weight_file_state_bytes=_integer(
                expanded.get("max_revision_weight_file_state_bytes"), 262_144
            ),
            **injected,
        )

    if adapter == "aws_sagemaker_jumpstart_versions":
        if client is None:
            raise ValueError(f"{name}: an injected signed SageMaker client is required")
        return AwsSageMakerJumpStartVersionsSourceAdapter(
            name=name,
            client=client,
            page_size=_integer(expanded.get("page_size"), 50),
            max_versions_per_model=_integer(expanded.get("max_versions_per_model"), 10_000),
            max_records_per_page=_integer(expanded.get("max_records_per_page"), 20_000),
        )

    if adapter == "open_x_rt1x_checkpoint":
        return OpenXRT1XCheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 5_000),
            max_pages=_integer(expanded.get("max_pages"), 50),
            **injected,
        )

    if adapter == "torchxrayvision_registry":
        return TorchXRayVisionRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100_000),
            **injected,
        )

    if adapter == "google_research_bert_checkpoints":
        return GoogleResearchBertCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "satmae_checkpoint_registry":
        return SatMAECheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 8),
            **injected,
        )

    if adapter == "pyg_dimenet_checkpoints":
        return PyGDimeNetCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "google_football_checkpoints":
        return GoogleFootballCheckpointAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "lerobot_molmoact2_relation":
        return LeRobotMolmoAct2RelationSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "paddlehelix_gem_checkpoint":
        return PaddleHelixGemCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "dev",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 1 * 1024 * 1024),
            **injected,
        )

    if adapter == "sdss_ssl_checkpoints":
        return SdssSslCheckpointsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_links=_integer(expanded.get("max_links"), 2_000),
            **injected,
        )

    if adapter == "openclip_pretrained_registry":
        return OpenCLIPPretrainedRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 2 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 2_000),
            client=injected["client"],
        )

    if adapter == "groundingdino_checkpoints":
        return GroundingDINOCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_readme_bytes=_integer(expanded.get("max_readme_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            client=injected["client"],
        )

    if adapter == "yolox_model_zoo":
        return YOLOXModelZooSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_readme_bytes=_integer(expanded.get("max_readme_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            client=injected["client"],
        )

    if adapter == "nvidia_earth2":
        return NvidiaEarth2SourceAdapter(
            page_size=_integer(expanded.get("page_size"), 20),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter in {"software_heritage_origins", "software-heritage-origins"}:
        return SoftwareHeritageOriginSourceAdapter(
            name=name,
            bucket_url=_required_text(expanded, "bucket_url"),
            graph_prefix=_text(expanded.get("graph_prefix")) or "graph/",
            release_list_url=_required_text(expanded, "release_list_url"),
            page_size=_integer(expanded.get("page_size"), 32),
            list_page_size=_integer(expanded.get("list_page_size"), 1_000),
            max_list_bytes=_integer(expanded.get("max_list_bytes"), 16 * 1024 * 1024),
            max_release_list_bytes=_integer(
                expanded.get("max_release_list_bytes"), 4 * 1024 * 1024
            ),
            max_metadata_bytes=_integer(expanded.get("max_metadata_bytes"), 8 * 1024 * 1024),
            max_list_pages=_integer(expanded.get("max_list_pages"), 1_000),
            max_manifest_objects=_integer(expanded.get("max_manifest_objects"), 100_000),
            settlement_age_hours=_integer(expanded.get("settlement_age_hours"), 7 * 24),
            **injected,
        )

    if adapter in {"paperswithcode_links", "paperswithcode-links"}:
        return PapersWithCodeLinksSourceAdapter(
            name=name,
            dataset_id=_required_text(expanded, "dataset_id"),
            metadata_url=_required_text(expanded, "metadata_url"),
            data_path=_required_text(expanded, "data_path"),
            license=_text(expanded.get("license")) or "CC-BY-SA-4.0",
            page_size=_integer(expanded.get("page_size"), 10_000),
            max_dataset_bytes=_integer(expanded.get("max_dataset_bytes"), 64 * 1024 * 1024),
            **injected,
        )

    if adapter in {"paperswithcode_methods_validated", "paperswithcode-methods-validated"}:
        return PapersWithCodeValidatedMethodsSourceAdapter(
            name=name,
            dataset_id=_required_text(expanded, "dataset_id"),
            metadata_url=_required_text(expanded, "metadata_url"),
            data_path=_required_text(expanded, "data_path"),
            license=_text(expanded.get("license")) or "CC-BY-SA-4.0",
            arxiv_api_url=_text(expanded.get("arxiv_api_url"))
            or "https://export.arxiv.org/api/query",
            arxiv_batch_size=_integer(expanded.get("arxiv_batch_size"), 100),
            arxiv_delay_seconds=float(expanded.get("arxiv_delay_seconds", 3)),
            admission=_text(expanded.get("admission")) or "arxiv_verified",
            max_dataset_bytes=_integer(expanded.get("max_dataset_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter in {"paperswithcode_evaluation_methods", "paperswithcode-evaluation-methods"}:
        return PapersWithCodeEvaluationMethodsSourceAdapter(
            name=name,
            dataset_id=_required_text(expanded, "dataset_id"),
            metadata_url=_required_text(expanded, "metadata_url"),
            license=_text(expanded.get("license")) or "CC-BY-SA-4.0",
            max_dataset_bytes=_integer(expanded.get("max_dataset_bytes"), 80 * 1024 * 1024),
            max_model_rows_per_shard=_integer(expanded.get("max_model_rows_per_shard"), 200_000),
            **injected,
        )

    if adapter in {"arxiv_complete_snapshot", "arxiv-complete-snapshot"}:
        return ArxivCompleteSnapshotSourceAdapter(
            name=name,
            artifact_source=_text(expanded.get("artifact_source")) or "arxiv",
            dataset_id=_required_text(expanded, "dataset_id"),
            metadata_url=_required_text(expanded, "metadata_url"),
            data_path=_required_text(expanded, "data_path"),
            license=_required_text(expanded, "license"),
            page_size=_integer(expanded.get("page_size"), 10_000),
            max_range_bytes=_integer(expanded.get("max_range_bytes"), 32 * 1024 * 1024),
            **injected,
        )

    url = (
        _text(expanded.get("url"))
        if adapter
        in {
            "tensorflow_model_garden",
            "jax_registry",
            "openml_flow_registry",
            "proteinmpnn_checkpoints",
            "stardist_pretrained_registry",
            "jax_extra_registry",
            "coqui_tts_registry",
            "graph_ml_registry",
            "dgl_lifesci_checkpoint_registry",
            "unimof_checkpoint_registry",
            "openfold_checkpoints",
            "openfold3_parameter_registry",
            "argus_checkpoint_inventory",
            "cellpose_registry",
            "pyg_gpse_registry",
            "nnunet_v1_pretrained_registry",
            "compvis_latent_diffusion_downloads",
            "compvis_latent_diffusion_readme_downloads",
            "compvis_stable_diffusion_first_stages",
            "mindspore_modelzoo",
            "google_robotics_transformer_checkpoints",
            "diffusion_policy_checkpoints",
            "torch_hub_listing_extra",
            "chemprop_chemeleon_checkpoint",
            "acl_anthology",
            "alphafold_parameter_archive",
            "graphgps_release_asset",
            "google_mediapipe_model_catalog",
            "gensim_downloader_model_registry",
            "allennlp_model_archives",
            "fairseq_pretrained_language_models",
            "mace_off23_checkpoint_registry",
            "mace_foundation_checkpoint_registry",
            "mace_omol_checkpoint",
            "moler_default_checkpoint",
            "fengwu_checkpoint_registry",
            "nvidia_groot_n17_checkpoints",
            "paddlenlp_taskflow_text_similarity",
            "conceptnet_numberbatch",
            "neuralgcm_checkpoint_registry",
            "pangu_weather_checkpoint_registry",
            "paddlex_model_list",
            "wenet_pretrained_models",
            "rl_clarity_checkpoint_index",
            "rfdiffusion_checkpoint_registry",
            "nltk_data_models",
            "pytorch_hub_load_calls",
            "paddlenlp_taskflow_uie",
            "deepchem_mol2vec_checkpoint",
            "grover_checkpoint_registry",
            "microsoft_vq_diffusion_checkpoint_manifest",
            "openvla_checkpoints",
            "sherpa_source_separation",
            "alphachip_rl_checkpoint",
            "paddlenlp_taskflow_sentiment",
            "gitlab_public_release_assets",
            "admet_ai_checkpoint_registry",
            "dipy_pretrained_registry",
            "paddlenlp_taskflow_knowledge_mining",
            "octo_checkpoints",
            "sherpa_audio_tagging",
            "ngc_cli_model_versions",
            "pelican_vla_checkpoint_registry",
            "ocp_model_registry",
            "dopamine_checkpoint_bundles",
            "kaldi_model_index",
            "galaxea_vla_checkpoints",
            "gpt4all_model_catalog",
            "bpemb_pretrained_vector_registry",
        }
        else _required_text(expanded, "url")
    )
    artifact_kind = _text(expanded.get("artifact_kind")) or None

    if adapter in {"html_catalog", "html-catalog"}:
        rules = expanded.get("rules")
        if not isinstance(rules, list | tuple):
            raise ValueError(f"{name}: HTML catalog requires a source.rules array")
        allowed_origins = expanded.get("allowed_origins")
        if allowed_origins is not None and not isinstance(allowed_origins, list | tuple):
            raise ValueError(f"{name}: source.allowed_origins must be an array")
        shared_link_rules = expanded.get("shared_links", ())
        if not isinstance(shared_link_rules, list | tuple):
            raise ValueError(f"{name}: source.shared_links must be an array")
        detail_resource_rules = expanded.get("detail_resource_rules", ())
        if not isinstance(detail_resource_rules, list | tuple):
            raise ValueError(f"{name}: source.detail_resource_rules must be an array")
        return HtmlCatalogSourceAdapter(
            name=name,
            url=url,
            provider_namespace=_required_text(expanded, "provider_namespace"),
            rules=rules,
            shared_link_rules=shared_link_rules,
            detail_resource_rules=detail_resource_rules,
            detail_batch_size=_integer(expanded.get("detail_batch_size"), 25),
            artifact_kind=artifact_kind or "catalog_record",
            model_status=_text(expanded.get("model_status")) or "documented",
            allowed_origins=allowed_origins,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 20_000),
            max_anchors=_integer(expanded.get("max_anchors"), 100_000),
            max_tables=_integer(expanded.get("max_tables"), 2_000),
            max_cells=_integer(expanded.get("max_cells"), 500_000),
            max_text_chars=_integer(expanded.get("max_text_chars"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter == "semantic_scholar_datasets":
        api_key = _credential(expanded, environment, defaults=("S2_API_KEY",))
        if not api_key:
            raise ValueError(f"{name}: configured credential environment variable is unset")
        datasets = expanded.get("datasets", ("papers", "abstracts", "paper-ids"))
        if not isinstance(datasets, list | tuple):
            raise ValueError(f"{name}: source.datasets must be an array")
        return SemanticScholarDatasetSourceAdapter(
            name=name,
            url=url,
            datasets=datasets,
            release_id=_text(expanded.get("release_id")) or "latest",
            page_size=_integer(expanded.get("page_size"), 100),
            prefer_diffs=_boolean(expanded.get("prefer_diffs"), default=True),
            api_key=api_key,
            **injected,
        )

    if adapter == "zenodo_oai_model_candidates":
        if _required_text(expanded, "metadata_prefix") != "dcat":
            raise ValueError(f"{name}: Zenodo OAI source requires dcat metadata")
        return ZenodoOaiModelCandidatesSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter == "csv":
        mapping = expanded.get("mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{name}: CSV source requires a [source.mapping] table")
        headers = expanded.get("headers")
        if headers is not None and not isinstance(headers, Mapping):
            raise ValueError(f"{name}: source.headers must be a table")
        return CsvSourceAdapter(
            name=name,
            url=url,
            mapping=mapping,
            artifact_kind=artifact_kind or "catalog_record",
            headers={str(key): str(value) for key, value in (headers or {}).items()},
            **injected,
        )

    if adapter == "huggingface":
        auth_env = _text(expanded.get("auth_env"))
        token = _text(environment.get(auth_env)) if auth_env else ""
        if auth_env and not token:
            raise ValueError(f"{name}: configured credential environment variable is unset")
        return HuggingFaceSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "model_card",
            page_size=_integer(expanded.get("page_size"), 100),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                16 * 1024 * 1024,
            ),
            token=token or None,
            include_private=_boolean(expanded.get("include_private"), default=False),
            include_revisions=_boolean(expanded.get("include_revisions"), default=False),
            include_revision_files=_boolean(
                expanded.get("include_revision_files"), default=False
            ),
            max_revision_tree_pages=_integer(
                expanded.get("max_revision_tree_pages"), 20
            ),
            created_at_sweep_interval_days=_nonnegative_integer(
                expanded.get("created_at_sweep_interval_days"), 0
            ),
            **injected,
        )

    if adapter in {"kaggle_models", "kaggle-models"}:
        return KaggleModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            page_size=_integer(expanded.get("page_size"), 100),
            sort_by=_text(expanded.get("sort_by")) or "createTime",
            include_all_versions=_boolean(expanded.get("include_all_versions"), default=True),
            include_version_files=_boolean(expanded.get("include_version_files"), default=False),
            artifact_kind=artifact_kind or "model_card",
            client=client or HttpClient(),
        )

    if adapter in {"civitai_models", "civitai-models"}:
        return CivitaiModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            page_size=_integer(expanded.get("page_size"), 100),
            sort_by=_text(expanded.get("sort_by")) or "Newest",
            include_nsfw=_boolean(expanded.get("include_nsfw"), default=True),
            artifact_kind=artifact_kind or "model_card",
            client=client or HttpClient(),
        )

    if adapter in {"opencsg_models", "opencsg-models"}:
        return OpenCsgModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            page_size=_integer(expanded.get("page_size"), 100),
            sort_by=_text(expanded.get("sort_by")) or "recently_update",
            artifact_kind=artifact_kind or "model_card",
            client=client or HttpClient(),
        )

    if adapter in {"modelscope_models", "modelscope-models"}:
        sorts = expanded.get("sorts")
        if sorts is None:
            sorts = ("default", "downloads", "likes", "last_modified")
        if not isinstance(sorts, list | tuple):
            raise ValueError(f"{name}: sorts must be an array")
        return ModelScopeModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            page_size=_integer(expanded.get("page_size"), 50),
            max_pages_per_sort=_integer(
                expanded.get("max_pages_per_sort"),
                60,
            ),
            sorts=tuple(str(sort) for sort in sorts),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            **injected,
        )

    if adapter in {"ngc_models", "ngc-models"}:
        return NgcModelsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            model_page_base_url=(
                _text(expanded.get("model_page_base_url")) or "https://catalog.ngc.nvidia.com"
            ),
            metadata_base_url=(
                _text(expanded.get("metadata_base_url")) or "https://api.ngc.nvidia.com"
            ),
            page_size=_integer(expanded.get("page_size"), 200),
            max_response_bytes=_integer(
                expanded.get("max_response_bytes"),
                4 * 1024 * 1024,
            ),
            **injected,
        )

    if adapter in {"nemo_checkpoint_catalog", "nemo-checkpoint-catalog"}:
        return NemoCheckpointCatalogSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            max_anchors=_integer(expanded.get("max_anchors"), 100_000),
            max_tables=_integer(expanded.get("max_tables"), 10_000),
            max_cells=_integer(expanded.get("max_cells"), 100_000),
            max_text_chars=_integer(expanded.get("max_text_chars"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter in {"zenodo_model_records", "zenodo-model-records"}:
        return ZenodoModelRecordsSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            query=_text(expanded.get("query")) or "resource_type.type:model",
            sort=_text(expanded.get("sort")) or "oldest",
            page_size=_integer(expanded.get("page_size"), 100),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter == "json_catalog":
        mapping = expanded.get("mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError(f"{name}: JSON catalog requires a [source.mapping] table")
        headers = expanded.get("headers")
        if headers is not None and not isinstance(headers, Mapping):
            raise ValueError(f"{name}: source.headers must be a table")
        auth_env = _text(expanded.get("auth_env"))
        auth_token = _text(environment.get(auth_env)) if auth_env else ""
        if auth_env and not auth_token:
            raise ValueError(f"{name}: configured credential environment variable is unset")
        max_response_bytes = _integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024)
        return JsonCatalogSourceAdapter(
            name=name,
            url=url,
            provider_namespace=_required_text(expanded, "provider_namespace"),
            model_card_url_template=_required_text(expanded, "model_card_url_template"),
            mapping=mapping,
            cursor_param=_text(expanded.get("cursor_param")) or "cursor",
            page_size=_integer(expanded.get("page_size"), 100),
            page_size_param=_text(expanded.get("page_size_param")) or None,
            auth_header_name=_text(expanded.get("auth_header")) or None,
            auth_query_param=_text(expanded.get("auth_query_param")) or None,
            auth_token=auth_token or None,
            auth_scheme=(
                _text(expanded.get("auth_scheme")) if "auth_scheme" in expanded else "Bearer"
            ),
            static_headers=headers,
            max_response_bytes=max_response_bytes,
            provider_id_is_release=_boolean(expanded.get("provider_id_is_release"), default=False),
            model_status=_text(expanded.get("model_status")) or "released",
            model_page_crawl=_boolean(expanded.get("model_page_crawl"), default=True),
            client=client or HttpClient(max_response_bytes=max_response_bytes),
        )

    if adapter == "openalex":
        api_key = _credential(expanded, environment, defaults=("OPENALEX_API_KEY",))
        mailto_env = _text(expanded.get("mailto_env")) or "MODELOME_CONTACT_EMAIL"
        mailto = _text(expanded.get("mailto")) or _text(environment.get(mailto_env))
        return OpenAlexSourceAdapter(
            name=name,
            artifact_source=_text(expanded.get("artifact_source")) or name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            page_size=_integer(expanded.get("page_size"), 100),
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            filter=_text(expanded.get("filter")),
            corpus=_text(expanded.get("corpus")) or "all",
            sync_mode=_text(expanded.get("sync_mode")) or "published",
            api_key=api_key,
            mailto=mailto or None,
            **injected,
        )

    if adapter == "arxiv":
        return ArxivSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            **injected,
        )

    if adapter in {"plos", "plos_solr", "plos-solr"}:
        return PlosSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            page_size=_integer(expanded.get("page_size"), 100),
            initial_start_date=(_text(expanded.get("initial_start_date")) or "2003-08-18"),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 14),
            **injected,
        )

    if adapter == "pmc":
        return PmcSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 64 * 1024 * 1024),
            max_record_bytes=_integer(expanded.get("max_record_bytes"), 32 * 1024 * 1024),
            max_records_per_page=_integer(expanded.get("max_records_per_page"), 100),
            max_elements_per_record=_integer(expanded.get("max_elements_per_record"), 250_000),
            max_text_chars_per_record=_integer(
                expanded.get("max_text_chars_per_record"), 10_000_000
            ),
            max_authors_per_record=_integer(expanded.get("max_authors_per_record"), 10_000),
            max_external_urls_per_record=_integer(
                expanded.get("max_external_urls_per_record"), 100_000
            ),
            **injected,
        )

    if adapter in {"hal", "hal_oai", "hal-oai"}:
        return HalSourceAdapter(
            name=name,
            url=url,
            web_base_url=_text(expanded.get("web_base_url")) or "https://hal.science",
            artifact_kind=artifact_kind or "paper",
            initial_start_date=_text(expanded.get("initial_start_date")) or "2002-09-23",
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            **injected,
        )

    if adapter == "crossref":
        mailto_env = _text(expanded.get("mailto_env")) or "MODELOME_CONTACT_EMAIL"
        mailto = _text(expanded.get("mailto")) or _text(environment.get(mailto_env))
        return CrossrefSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            page_size=_integer(expanded.get("page_size"), 1_000),
            initial_lookback_days=_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_integer(expanded.get("overlap_days"), 2),
            mailto=mailto or None,
            **injected,
        )

    if adapter == "datacite":
        return DataCiteSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or None,
            page_size=_integer(expanded.get("page_size"), 1_000),
            initial_lookback_days=_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_integer(expanded.get("overlap_days"), 2),
            **injected,
        )

    if adapter == "europe_pmc":
        email_env = _text(expanded.get("email_env")) or "MODELOME_CONTACT_EMAIL"
        email = _text(expanded.get("email")) or _text(environment.get(email_env))
        return EuropePmcSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            page_size=_integer(expanded.get("page_size"), 1_000),
            initial_lookback_days=_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_integer(expanded.get("overlap_days"), 2),
            email=email or None,
            **injected,
        )

    if adapter == "biorxiv":
        return BioRxivSourceAdapter(
            name=name,
            url=url,
            server=_required_text(expanded, "server"),
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 1),
            **injected,
        )

    if adapter == "biorxiv_jats_supplementary":
        metadata_source = BioRxivSourceAdapter(
            name=_required_text(expanded, "server"),
            url=url,
            server=_required_text(expanded, "server"),
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(
                expanded.get("initial_lookback_days"), 7
            ),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 1),
            **injected,
        )
        max_jats_bytes = _integer(expanded.get("max_jats_bytes"), 32 * 1024 * 1024)
        return BioRxivJatsSupplementSourceAdapter(
            source=metadata_source,
            client=HttpClient(max_response_bytes=max_jats_bytes),
            max_jats_fetches_per_page=_integer(
                expanded.get("max_jats_fetches_per_page"), 30
            ),
            max_jats_bytes=max_jats_bytes,
            max_jats_elements=_integer(expanded.get("max_jats_elements"), 250_000),
            max_supplement_links_per_record=_integer(
                expanded.get("max_supplement_links_per_record"), 2_000
            ),
            minimum_request_interval_seconds=float(
                expanded.get("minimum_request_interval_seconds", 0.34)
            ),
        )

    if adapter in {"biorxiv_publications", "biorxiv-publications", "biorxiv_pubs"}:
        return BioRxivPublicationSourceAdapter(
            name=name,
            url=url,
            server=_required_text(expanded, "server"),
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 90),
            **injected,
        )

    if adapter in {"osf_preprints", "osf-preprints"}:
        return OsfPreprintSourceAdapter(
            name=name,
            url=url,
            artifact_kind=artifact_kind or "paper",
            page_size=_integer(expanded.get("page_size"), 100),
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            **injected,
        )

    if adapter in {"eartharxiv", "eartharxiv_oai", "eartharxiv-oai"}:
        return EarthArxivSourceAdapter(
            name=name,
            url=url,
            web_base_url=_required_text(expanded, "web_base_url"),
            artifact_kind=artifact_kind or "paper",
            initial_lookback_days=_nonnegative_integer(expanded.get("initial_lookback_days"), 7),
            overlap_days=_nonnegative_integer(expanded.get("overlap_days"), 2),
            **injected,
        )

    if adapter == "tensorflow_model_garden":
        return TensorFlowGardenSourceAdapter(
            name=name,
            max_bytes=_integer(expanded.get("max_bytes"), 8 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "jax_registry":
        return JaxRegistrySourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "google-research/t5x",
            branch=_text(expanded.get("branch")) or "main",
            source_path=_text(expanded.get("source_path")) or "docs/models.md",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "openml_flow_registry":
        return OpenMLFlowRegistrySourceAdapter(
            name=name,
            url=_text(expanded.get("url")) or "https://www.openml.org/api/v1/json/flow/list",
            page_size=_integer(expanded.get("page_size"), 100),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            client=injected["client"],
        )

    if adapter == "proteinmpnn_checkpoints":
        return ProteinMpnSourceAdapter(
            name=name,
            repository=_text(expanded.get("repository")) or "dauparas/ProteinMPNN",
            branch=_text(expanded.get("branch")) or "main",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter == "acl_anthology":
        return AclAnthologySourceAdapter(
            name=name,
            url=_text(expanded.get("url")) or None,
            repository=_text(expanded.get("repository")) or "acl-org/acl-anthology",
            branch=_text(expanded.get("branch")) or "master",
            manifest_url=_text(expanded.get("manifest_url")) or None,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 64 * 1024 * 1024),
            max_papers=_integer(expanded.get("max_papers"), 10_000),
            max_collections=_integer(expanded.get("max_collections"), 20_000),
            max_abstract_chars=_integer(expanded.get("max_abstract_chars"), 100_000),
            **injected,
        )

    if adapter == "stardist_pretrained_registry":
        return StarDistPretrainedRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 2 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 1_000),
            **injected,
        )

    if adapter == "jax_extra_registry":
        return JaxExtraRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "coqui_tts_registry":
        return CoquiTtsRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "dev",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 20_000),
            max_depth=_integer(expanded.get("max_depth"), 8),
            **injected,
        )

    if adapter == "graph_ml_registry":
        return GraphMLRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            mapping_variable=_required_text(expanded, "mapping_variable"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            checkpoint_base_url=_required_text(expanded, "checkpoint_base_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "dgl_lifesci_checkpoint_registry":
        return DglLifeSciCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            mapping_variable=_required_text(expanded, "mapping_variable"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            checkpoint_base_url=_required_text(expanded, "checkpoint_base_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "oci_generative_ai_model_catalog":
        return OciGenerativeAIModelCatalog(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            client=injected["client"],
        )

    if adapter == "openfold_checkpoints":
        return OpenFoldCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "openfold3_parameter_registry":
        return OpenFold3ParameterRegistryAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 512 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "argus_checkpoint_inventory":
        return ArgusCheckpointInventorySourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "unimof_checkpoint_registry":
        return UniMofCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "chemprop_chemeleon_checkpoint":
        if (
            _required_text(expanded, "documentation_url")
            != ChempropCheMeleonCheckpointSourceAdapter.documentation_url
        ):
            raise ValueError(f"{name}: unexpected Chemprop documentation URL")
        if (
            _required_text(expanded, "weight_url")
            != ChempropCheMeleonCheckpointSourceAdapter.weight_url
        ):
            raise ValueError(f"{name}: unexpected CheMeleon checkpoint URL")
        return ChempropCheMeleonCheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            **injected,
        )

    if adapter == "cellpose_registry":
        return CellposeRegistrySourceAdapter(
            name=name,
            model_url_base=_required_text(expanded, "model_url_base"),
            **({"clock": clock} if clock is not None else {}),
        )

    if adapter == "pyg_gpse_registry":
        return PyGGPSECheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "nnunet_v1_pretrained_registry":
        return NnUNetV1PretrainedRegistryAdapter(
            name=name,
            record_id=_required_text(expanded, "record_id"),
            max_record_bytes=_integer(expanded.get("max_record_bytes"), 4 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 500),
            **injected,
        )

    if adapter == "compvis_latent_diffusion_downloads":
        return CompVisLatentDiffusionDownloadsSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "compvis_stable_diffusion_first_stages":
        return CompVisStableDiffusionFirstStagesSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "compvis_latent_diffusion_readme_downloads":
        return CompVisLatentDiffusionReadmeDownloadsSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "mace_off23_checkpoint_registry":
        return MaceOff23CheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            mapping_variable=_required_text(expanded, "mapping_variable"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "mace_foundation_checkpoint_registry":
        return MaceFoundationCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            **injected,
        )

    if adapter == "mace_omol_checkpoint":
        return MaceOmolCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "moler_default_checkpoint":
        return MoLeRCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "neuralgcm_checkpoint_registry":
        return NeuralGCMCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "pangu_weather_checkpoint_registry":
        return PanguWeatherCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10),
            **injected,
        )

    if adapter == "fengwu_checkpoint_registry":
        return FengWuCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10),
            **injected,
        )

    if adapter == "nvidia_groot_n17_checkpoints":
        return NvidiaGR00TN17CheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 8),
            **injected,
        )

    if adapter == "paddlenlp_taskflow_text_similarity":
        return PaddleNlpTaskflowTextSimilaritySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "paddlex_model_list":
        if _required_text(expanded, "provider_namespace") != "paddlex:model":
            raise ValueError(f"{name}: unexpected PaddleX namespace")
        return PaddleXModelListSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_required_text(expanded, "branch"),
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_rows=_integer(expanded.get("max_rows"), 10_000),
            **injected,
        )

    if adapter == "wenet_pretrained_models":
        return WenetPretrainedModelSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            document_path=_required_text(expanded, "document_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_rows=_integer(expanded.get("max_rows"), 2_000),
            **injected,
        )

    if adapter == "rl_clarity_checkpoint_index":
        return RlClarityCheckpointIndexAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "rfdiffusion_checkpoint_registry":
        return RFDiffusionCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "nltk_data_models":
        return NltkDataModelIndexSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "gh-pages",
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            **injected,
        )

    if adapter == "pytorch_hub_load_calls":
        return PyTorchHubLoadCallSourceAdapter(
            name=name,
            index_url=_required_text(expanded, "index_url"),
            page_batch_size=_integer(expanded.get("page_batch_size"), 10),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_code_chars=_integer(expanded.get("max_code_chars"), 256 * 1024),
            max_pages=_integer(expanded.get("max_pages"), 1_000),
            client=injected["client"],
        )

    if adapter == "paddlenlp_taskflow_uie":
        return PaddleNlpTaskflowUieSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            **injected,
        )

    if adapter == "deepchem_mol2vec_checkpoint":
        return DeepChemMol2VecCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "grover_checkpoint_registry":
        return GroverCheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "microsoft_vq_diffusion_checkpoint_manifest":
        return MicrosoftVqDiffusionCheckpointManifestSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            **injected,
        )

    if adapter == "openvla_checkpoints":
        return OpenVLACheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 20),
            **injected,
        )

    if adapter == "sherpa_source_separation":
        return SherpaSourceSeparationSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            document_path=_required_text(expanded, "document_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 500),
            **injected,
        )

    if adapter == "alphachip_rl_checkpoint":
        return AlphaChipRlCheckpointAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "paddlenlp_taskflow_sentiment":
        return PaddleNlpTaskflowSentimentSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "gitlab_public_release_assets":
        return GitLabPublicReleaseAssetsSourceAdapter(
            name=name,
            initial_project_id=(
                _nonnegative_integer(expanded["initial_project_id"], 0)
                if "initial_project_id" in expanded
                else None
            ),
            max_project_id=(
                _integer(expanded["max_project_id"], 1)
                if "max_project_id" in expanded
                else None
            ),
            max_projects=(
                _integer(expanded["max_projects"], 1)
                if "max_projects" in expanded
                else None
            ),
            page_size=_integer(expanded.get("page_size"), 100),
            max_projects_per_page=_integer(expanded.get("max_projects_per_page"), 100),
            max_releases_per_page=_integer(expanded.get("max_releases_per_page"), 100),
            max_assets_per_release=_integer(expanded.get("max_assets_per_release"), 1_000),
            client=injected["client"],
        )

    if adapter == "ngc_cli_model_versions":
        targets = expanded.get("targets")
        if not isinstance(targets, list) or not all(isinstance(value, str) for value in targets):
            raise ValueError(f"{name}: targets must be a list of model identifiers")
        return NgcCliModelVersionsSourceAdapter(
            targets=targets,
            name=name,
            executable=_text(expanded.get("executable")) or "ngc",
            max_versions_per_target=_integer(expanded.get("max_versions_per_target"), 5_000),
        )

    if adapter == "admet_ai_checkpoint_registry":
        return ADMETAICheckpointRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 16 * 1024 * 1024),
            **injected,
        )

    if adapter == "dipy_pretrained_registry":
        return DipyPretrainedRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            source_path=_required_text(expanded, "source_path"),
            max_source_bytes=_integer(expanded.get("max_source_bytes"), 8 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 100),
            **injected,
        )

    if adapter == "paddlenlp_taskflow_knowledge_mining":
        return PaddleNlpTaskflowKnowledgeMiningSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "develop",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "octo_checkpoints":
        return OctoCheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10),
            **injected,
        )

    if adapter == "sherpa_audio_tagging":
        return SherpaAudioTaggingSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            document_path=_required_text(expanded, "document_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_models=_integer(expanded.get("max_models"), 200),
            **injected,
        )

    if adapter == "pelican_vla_checkpoint_registry":
        return PelicanVLACheckpointRegistrySourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 20),
            **injected,
        )

    if adapter == "ocp_model_registry":
        if _required_text(expanded, "page_url") != (
            "https://facebookresearch.github.io/fairchem/models-1/"
        ):
            raise ValueError(f"{name}: unexpected FAIR Chemistry model page")
        return OCPModelRegistrySourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            **injected,
        )

    if adapter == "dopamine_checkpoint_bundles":
        return DopamineCheckpointBundleAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            **injected,
        )

    if adapter == "kaldi_model_index":
        return KaldiModelIndexSourceAdapter(
            name=name,
            index_url=_required_text(expanded, "index_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_resources=_integer(expanded.get("max_resources"), 100),
            max_archives=_integer(expanded.get("max_archives"), 2_000),
            **injected,
        )

    if adapter == "galaxea_vla_checkpoints":
        return GalaxeaVLACheckpointSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 50),
            **injected,
        )

    if adapter == "gpt4all_model_catalog":
        return Gpt4AllModelCatalogSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            source_path=_required_text(expanded, "source_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "mindspore_modelzoo":
        return MindSporeModelZooSourceAdapter(
            name=name,
            root_url=_text(expanded.get("root_url"))
            or "https://download.mindspore.cn/model_zoo/official/",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_total_bytes=_integer(expanded.get("max_total_bytes"), 64 * 1024 * 1024),
            max_pages=_integer(expanded.get("max_pages"), 2_000),
            max_links_per_page=_integer(expanded.get("max_links_per_page"), 20_000),
            max_artifacts=_integer(expanded.get("max_artifacts"), 20_000),
            max_depth=_integer(expanded.get("max_depth"), 8),
            **injected,
        )

    if adapter == "google_robotics_transformer_checkpoints":
        return RoboticsTransformerCheckpointSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 100_000),
            **injected,
        )

    if adapter == "diffusion_policy_checkpoints":
        return DiffusionPolicyCheckpointIndexAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_directories=_integer(expanded.get("max_directories"), 1_024),
            max_entries=_integer(expanded.get("max_entries"), 20_000),
            **injected,
        )

    if adapter == "torch_hub_listing_extra":
        return TorchHubListingSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            max_archive_bytes=_integer(expanded.get("max_archive_bytes"), 32 * 1024 * 1024),
            max_file_bytes=_integer(expanded.get("max_file_bytes"), 2 * 1024 * 1024),
            max_files=_integer(expanded.get("max_files"), 5_000),
            **injected,
        )

    if adapter == "alphafold_parameter_archive":
        return AlphaFoldParameterArchiveSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            max_script_bytes=_integer(expanded.get("max_script_bytes"), 256 * 1024),
            **injected,
        )

    if adapter == "graphgps_release_asset":
        return GraphGPSReleaseAssetSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "main",
            document_path=_required_text(expanded, "document_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "google_mediapipe_model_catalog":
        return MediaPipeModelCatalogSourceAdapter(
            name=name,
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 200),
            **injected,
        )

    if adapter == "gensim_downloader_model_registry":
        if _required_text(expanded, "manifest_path") != "list.json":
            raise ValueError(f"{name}: Gensim manifest path must be list.json")
        return GensimDownloaderModelRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            **injected,
        )

    if adapter == "bpemb_pretrained_vector_registry":
        return BPEmbPretrainedVectorRegistrySourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 10_000),
            max_languages=_integer(expanded.get("max_languages"), 300),
            languages_per_page=_integer(expanded.get("languages_per_page"), 10),
            **injected,
        )

    if adapter == "conceptnet_numberbatch":
        return ConceptNetNumberbatchSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_text(expanded.get("branch")) or "master",
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "aws_bedrock_region_matrix":
        return AwsBedrockRegionMatrixAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 5_000),
            max_cells=_integer(expanded.get("max_cells"), 250_000),
            client=injected["client"],
        )

    if adapter == "opencv_dnn_model_index":
        return OpenCVDnnModelIndexSourceAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            repository_url=_required_text(expanded, "repository_url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 2 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 1_000),
            **injected,
        )

    if adapter == "allennlp_model_archives":
        return AllenNLPModelArchiveSourceAdapter(
            name=name,
            repository=_required_text(expanded, "repository"),
            branch=_required_text(expanded, "branch"),
            modelcards_path=_required_text(expanded, "modelcards_path"),
            provider_namespace=_required_text(expanded, "provider_namespace"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 4 * 1024 * 1024),
            max_entries=_integer(expanded.get("max_entries"), 100),
            **injected,
        )

    if adapter == "fairseq_pretrained_language_models":
        return FairseqPretrainedLanguageModelSourceAdapter(
            name=name,
            document_path=_required_text(expanded, "document_path"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_rows=_integer(expanded.get("max_rows"), 10_000),
            **injected,
        )

    if adapter == "ollama_library_tags":
        return OllamaLibraryTagCatalogAdapter(
            name=name,
            url=_required_text(expanded, "url"),
            max_response_bytes=_integer(expanded.get("max_response_bytes"), 8 * 1024 * 1024),
            max_families=_integer(expanded.get("max_families"), 5_000),
            max_tags_per_family=_integer(expanded.get("max_tags_per_family"), 10_000),
            max_pages_per_family=_integer(expanded.get("max_pages_per_family"), 100),
            max_anchors_per_page=_integer(expanded.get("max_anchors_per_page"), 100_000),
            client=injected["client"],
        )

    raise ValueError(f"{name}: unknown source adapter {adapter!r}")


def load_sources(
    path: str | Path | None = None,
    *,
    client: HttpClient | Any | None = None,
    clock: Clock | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, SourceAdapter]:
    """Load enabled source adapters, keyed by their unique catalog name."""

    environment = os.environ if environ is None else environ
    shared_client = client or HttpClient()
    sources: dict[str, SourceAdapter] = {}
    for config in load_source_configs(path):
        if config.get("enabled", True) is False:
            continue
        activation_env = _activation_env(config)
        if activation_env and not _text(environment.get(activation_env)):
            continue
        source = create_source(
            config,
            client=shared_client,
            clock=clock,
            environ=environment,
        )
        sources[source.name] = source
    return sources


def _activation_env(config: Mapping[str, Any]) -> str:
    if "activation_env" not in config:
        return ""
    name = _text(config.get("activation_env"))
    if not name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError("source.activation_env must name an environment variable")
    return name


def _credential(
    config: Mapping[str, Any],
    environ: Mapping[str, str],
    *,
    defaults: tuple[str, ...],
) -> str | None:
    if token := _text(config.get("token")):
        return token
    names = []
    if configured := _text(config.get("token_env")):
        names.append(configured)
    names.extend(defaults)
    return next((value for name in names if (value := _text(environ.get(name)))), None)


def _expand_environment(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_REFERENCE.sub(lambda match: environ.get(match.group(1), ""), value)
    if isinstance(value, Mapping):
        return {key: _expand_environment(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, environ) for item in value]
    return value


def _required_text(config: Mapping[str, Any], key: str, *, section: str = "source") -> str:
    value = _text(config.get(key))
    if not value:
        raise ValueError(f"{section}.{key} is required")
    return value


def _integer(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if result <= 0:
        raise ValueError("source numeric settings must be positive")
    return result


def _nonnegative_integer(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    if result < 0:
        raise ValueError("source numeric settings must be nonnegative")
    return result


def _boolean(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError("source boolean settings must be true or false")
    return value


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "create_source",
    "load_benchmark_configs",
    "load_source_configs",
    "load_sources",
]
