import pytest

from modelome.extract import IntroductionCueExtractor
from modelome.models import ArtifactKind, ModelStatus, SourceRecord


def paper(title: str, text: str) -> SourceRecord:
    return SourceRecord(
        source_record_id="paper-1",
        kind=ArtifactKind.PAPER,
        canonical_url="https://example.test/paper-1",
        title=title,
        text=text,
        raw={"title": title, "abstract": text},
    )


def test_discovers_names_from_language_not_a_seed_list() -> None:
    record = paper(
        "A general method",
        "We introduce QuasarNet-17, a deep neural network for an unfamiliar task.",
    )

    hints = IntroductionCueExtractor().extract(record)

    assert [hint.name for hint in hints] == ["QuasarNet-17"]
    assert hints[0].status is ModelStatus.CANDIDATE
    assert hints[0].locator is not None


def test_title_prefix_is_only_a_scoped_candidate() -> None:
    hints = IntroductionCueExtractor().extract(
        paper("MoleculeForge: A deep-learning model for chemical space", "")
    )

    assert len(hints) == 1
    assert hints[0].name == "MoleculeForge"
    assert hints[0].confidence < 1


def test_does_not_turn_ordinary_prose_into_a_model() -> None:
    hints = IntroductionCueExtractor().extract(
        paper("An empirical study", "We propose a simple approach for evaluation.")
    )

    assert hints == ()


def test_repository_can_use_document_level_neural_scope() -> None:
    record = SourceRecord(
        source_record_id="owner/repository",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://github.com/owner/repository",
        title="owner/repository",
        text=(
            "This work presents NebulaForge-9, a new scientific surrogate. "
            "The implementation is a deep neural network trained end to end."
        ),
        raw={"full_name": "owner/repository"},
    )

    hints = IntroductionCueExtractor().extract(record)

    assert [hint.name for hint in hints] == ["NebulaForge-9"]
    assert hints[0].locator == "text:19:32"


def test_repository_readme_heading_can_name_a_code_only_neural_model() -> None:
    record = SourceRecord(
        source_record_id="owner/repository",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://github.com/owner/repository",
        title="owner/repository",
        text=(
            "Reference implementation.\n\n"
            "# EventHorizonNet-4\n\n"
            "A deep neural network for reconstructing sparse detector events."
        ),
        raw={"full_name": "owner/repository"},
    )

    hints = IntroductionCueExtractor().extract(record)

    assert [hint.name for hint in hints] == ["EventHorizonNet-4"]
    assert hints[0].locator == "text:29:46"


def test_repository_readme_heading_is_not_a_model_without_neural_scope() -> None:
    record = SourceRecord(
        source_record_id="owner/toolkit",
        kind=ArtifactKind.CODE_REPOSITORY,
        canonical_url="https://github.com/owner/toolkit",
        title="owner/toolkit",
        text="# DataToolkit\n\nUtilities for converting tabular files.",
        raw={"full_name": "owner/toolkit"},
    )

    assert IntroductionCueExtractor().extract(record) == ()


def test_discovers_aiayn_style_appositive_without_a_name_list() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "A general architecture",
            "We propose a new simple network architecture, the LatticeFormer-X, "
            "based solely on attention mechanisms.",
        )
    )

    assert [hint.name for hint in hints] == ["LatticeFormer-X"]
    assert hints[0].locator is not None


def test_discovers_self_named_neural_model_without_a_name_list() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Forecasting with learned representations",
            "We call our neural model AuroraNet-2. It predicts future observations.",
        )
    )

    assert [hint.name for hint in hints] == ["AuroraNet-2"]
    assert hints[0].confidence == 0.78


def test_pmc_style_adjacent_sentence_repeats_name_before_neural_scope() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "An approach for forecasting",
            "We introduce HelioNet-3 for long-range prediction. "
            "HelioNet-3 is a deep neural network trained on historical data.",
        )
    )

    assert [hint.name for hint in hints] == ["HelioNet-3"]


def test_adjacent_neural_evidence_without_repeated_name_does_not_scope_candidate() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "An approach for forecasting",
            "We introduce HelioNet-3 for long-range prediction. "
            "The task is solved with a deep neural network trained on historical data.",
        )
    )

    assert hints == ()


def test_discovers_model_name_in_possessive_apposition() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "A forecasting system",
            "Our deep neural model, AuroraNet-2, predicts future observations.",
        )
    )

    assert [hint.name for hint in hints] == ["AuroraNet-2"]


def test_repository_deposit_metadata_needs_local_neural_evidence() -> None:
    record = SourceRecord(
        source_record_id="10.1234/deposit",
        kind=ArtifactKind.OTHER,
        canonical_url="https://repository.example/deposit",
        title="A research deposit",
        text="We introduce DepositNet-X, a deep neural network with trained parameters.",
        raw={},
    )

    assert [hint.name for hint in IntroductionCueExtractor().extract(record)] == [
        "DepositNet-X"
    ]


def test_biomedical_here_we_developed_and_lowercase_quoted_name() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Mapping single cells",
            'Here, we developed "cell atlas mapper", a deep neural network for '
            "cell-state annotation.",
        )
    )

    assert [hint.name for hint in hints] == ["cell atlas mapper"]
    assert hints[0].locator == "text:20:37"


def test_chemistry_long_scientific_name() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Molecular property prediction",
            "In this work, we present Molecular Orbital Message Passing Network, "
            "a neural network for quantum-chemical prediction.",
        )
    )

    assert [hint.name for hint in hints] == ["Molecular Orbital Message Passing Network"]


def test_physics_name_first_appositive() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Closure modelling",
            "PlasmaClosure-X, a neural operator for turbulent closure prediction, "
            "was evaluated across several regimes.",
        )
    )

    assert [hint.name for hint in hints] == ["PlasmaClosure-X"]


def test_materials_long_title_prefix() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Crystal Graph Interatomic Potential Network: "
            "A deep-learning model for materials simulation",
            "",
        )
    )

    assert [hint.name for hint in hints] == [
        "Crystal Graph Interatomic Potential Network"
    ]


def test_title_family_is_a_candidate_without_a_name_seed() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "Spectral Neural Operator for irregular scientific fields",
            "We train a neural operator for data-driven simulation.",
        )
    )

    assert [hint.name for hint in hints] == ["Spectral Neural Operator"]
    assert hints[0].locator == "title:0:24"


def test_generic_title_family_is_not_a_model_identity() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "A Deep Neural Network for image reconstruction",
            "We train a deep neural network on the reconstruction task.",
        )
    )

    assert hints == ()


def test_compact_named_title_can_be_scoped_by_abstract() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "NovaFold 3 predicts molecular conformations",
            "NovaFold 3 is a deep learning system for molecular prediction.",
        )
    )

    assert [hint.name for hint in hints] == ["NovaFold 3"]


def test_generative_diffusion_scope_does_not_require_name_dictionary() -> None:
    hints = IntroductionCueExtractor().extract(
        paper(
            "NebulaDiff: A sampling method",
            "We introduce NebulaDiff, a generative diffusion model for molecular design.",
        )
    )

    assert [hint.name for hint in hints] == ["NebulaDiff"]


@pytest.mark.parametrize(
    "text",
    [
        "We developed BayesKinetics-X, a hierarchical statistical model fitted by MCMC.",
        "Here we describe Cardiac Mechanics Model, a mechanistic model of deformation.",
        "We introduce Porous Diffusion Model, a physical diffusion model for transport.",
    ],
)
def test_non_neural_named_models_are_not_admitted(text: str) -> None:
    assert IntroductionCueExtractor().extract(paper("A scientific model", text)) == ()


def test_paper_scope_must_be_local_to_the_candidate() -> None:
    record = paper(
        "Two modelling approaches",
        "We introduce PorousFlow-X, a mechanistic physical diffusion model. "
        "Separately, we train a deep neural network as a numerical control.",
    )

    assert IntroductionCueExtractor().extract(record) == ()


def test_called_dataset_is_not_confused_with_a_model_name() -> None:
    record = paper(
        "A neural analysis",
        "We train a deep neural network on a cohort called BioSet-X.",
    )

    assert IntroductionCueExtractor().extract(record) == ()
