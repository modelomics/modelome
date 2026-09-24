from modelome.extract import IntroductionCueExtractor
from modelome.models import ArtifactKind, Link, SourceRecord


def _paper(title: str, links: tuple[Link, ...]) -> SourceRecord:
    return SourceRecord(
        source_record_id="paper-1",
        kind=ArtifactKind.PAPER,
        canonical_url="https://arxiv.org/abs/2601.00001",
        title=title,
        raw={},
        # Deliberately lacks neural wording: the source link is the evidence.
        text="We evaluate a computational approach for forecasting.",
        links=links,
    )


def test_matching_model_card_url_supports_a_named_title_candidate() -> None:
    record = _paper(
        "NovaFold 3 predicts molecular conformations",
        (
            Link(
                "https://huggingface.co/example/NovaFold-3",
                relation="model_card",
                locator="abstract:available model",
            ),
        ),
    )

    hints = IntroductionCueExtractor().extract(record)

    assert [hint.name for hint in hints] == ["NovaFold 3"]
    assert hints[0].locator == "title:0:10"


def test_matching_weights_url_supports_model_name_separated_by_slug_punctuation() -> None:
    record = _paper(
        "NovaFold 3 predicts molecular conformations",
        (Link("https://models.example.test/releases/NovaFold_3/weights.bin", relation="weights"),),
    )

    assert [hint.name for hint in IntroductionCueExtractor().extract(record)] == [
        "NovaFold 3"
    ]


def test_unrelated_model_card_or_generic_code_link_does_not_expand_candidates() -> None:
    record = _paper(
        "NovaFold 3 predicts molecular conformations",
        (
            Link("https://huggingface.co/example/OtherModel", relation="model_card"),
            Link("https://github.com/example/NovaFold-3", relation="code_reference"),
        ),
    )

    assert IntroductionCueExtractor().extract(record) == ()
