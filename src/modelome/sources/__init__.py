"""Source adapters."""

from modelome.sources.arxiv import ArxivSourceAdapter
from modelome.sources.bioimageio import BioImageIoSourceAdapter
from modelome.sources.biorxiv import BioRxivPublicationSourceAdapter, BioRxivSourceAdapter
from modelome.sources.commoncrawl import CommonCrawlWetSourceAdapter
from modelome.sources.crossref import CrossrefSourceAdapter
from modelome.sources.datacite import DataCiteSourceAdapter
from modelome.sources.eartharxiv import EarthArxivSourceAdapter
from modelome.sources.europe_pmc import EuropePmcSourceAdapter
from modelome.sources.gharchive import GhArchiveSourceAdapter
from modelome.sources.github_repositories import GitHubPublicRepositoriesSourceAdapter
from modelome.sources.hal import HalSourceAdapter
from modelome.sources.html_catalog import HtmlCatalogSourceAdapter
from modelome.sources.monai_model_zoo import MonaiModelZooSourceAdapter
from modelome.sources.onnx_model_zoo import OnnxModelZooSourceAdapter
from modelome.sources.openai_models import OpenAIModelsSourceAdapter
from modelome.sources.openaire import OpenAireGraphSourceAdapter
from modelome.sources.openmmlab import OpenMMLabModelIndexSourceAdapter
from modelome.sources.openreview import OpenReviewSourceAdapter
from modelome.sources.osf_preprints import OsfPreprintSourceAdapter
from modelome.sources.paperswithcode import (
    PapersWithCodeEvaluationMethodsSourceAdapter,
    PapersWithCodeLinksSourceAdapter,
    PapersWithCodeValidatedMethodsSourceAdapter,
)
from modelome.sources.plos import PlosSourceAdapter
from modelome.sources.pmc import PmcSourceAdapter
from modelome.sources.pubmed import PubMedBulkSourceAdapter
from modelome.sources.semantic_scholar import SemanticScholarDatasetSourceAdapter
from modelome.sources.software_heritage import SoftwareHeritageOriginSourceAdapter

__all__ = [
    "ArxivSourceAdapter",
    "BioImageIoSourceAdapter",
    "BioRxivPublicationSourceAdapter",
    "BioRxivSourceAdapter",
    "CommonCrawlWetSourceAdapter",
    "CrossrefSourceAdapter",
    "DataCiteSourceAdapter",
    "EarthArxivSourceAdapter",
    "EuropePmcSourceAdapter",
    "GhArchiveSourceAdapter",
    "GitHubPublicRepositoriesSourceAdapter",
    "HalSourceAdapter",
    "HtmlCatalogSourceAdapter",
    "MonaiModelZooSourceAdapter",
    "OnnxModelZooSourceAdapter",
    "OpenAIModelsSourceAdapter",
    "OpenMMLabModelIndexSourceAdapter",
    "OpenAireGraphSourceAdapter",
    "OpenReviewSourceAdapter",
    "OsfPreprintSourceAdapter",
    "PlosSourceAdapter",
    "PmcSourceAdapter",
    "PapersWithCodeEvaluationMethodsSourceAdapter",
    "PapersWithCodeLinksSourceAdapter",
    "PapersWithCodeValidatedMethodsSourceAdapter",
    "PubMedBulkSourceAdapter",
    "SemanticScholarDatasetSourceAdapter",
    "SoftwareHeritageOriginSourceAdapter",
]
