"""Regression coverage for fenced Markdown in configured model catalogs."""

import re

from modelome.sources.markdown_checkpoint_list import _parse_checkpoint_list
from modelome.sources.markdown_model_card_list import _parse_model_cards
from modelome.sources.markdown_model_table import _relation, _table_cells, _visible_markdown


def test_markdown_list_fence_closer_must_match_opening_length() -> None:
    document = """\
## Models
````markdown
- `fake`: [fake](https://weights.test/fake.pth)
```
- `also_fake`: [fake](https://weights.test/also.pth)
````
- `real`: [real](https://weights.test/real.pth)
"""
    rows = _parse_checkpoint_list(
        document,
        source="fixture",
        path="README.md",
        heading_pattern=re.compile("Models", re.I),
        handle_pattern=re.compile(r"`(?P<handle>[^`]+)`"),
        maximum=10,
    )
    assert [row.handle for row in rows] == ["real"]


def test_card_list_fence_closer_must_match_opening_length() -> None:
    document = """\
## Models
~~~markdown
- [fake](https://huggingface.co/org/fake)
```
- [also fake](https://huggingface.co/org/also-fake)
~~~
- [real](https://huggingface.co/org/real)
"""
    cards = _parse_model_cards(
        document,
        source="fixture",
        path="README.md",
        heading_pattern=re.compile("Models", re.I),
        url_pattern=re.compile(r"https://huggingface\.co/(?P<handle>[^/]+/[^/]+)"),
        maximum=10,
    )
    assert [card.handle for card in cards] == ["org/real"]


def test_model_table_ignores_tables_inside_fenced_code() -> None:
    document = """\
## Models
````markdown
Model | Download
--- | ---
Fake | [weights](https://weights.test/fake.pth)
````
Model | Download
--- | ---
Real | [weights](https://weights.test/real.pth)
"""
    visible = _visible_markdown(document)
    assert "Fake |" not in visible
    assert "Real |" in visible
    assert visible.count("\n") == document.count("\n")


def test_model_table_keeps_escaped_and_inline_code_pipes_inside_cells() -> None:
    assert _table_cells(r"| `encoder|decoder` | A\|B | download |") == (
        "`encoder|decoder`",
        "A|B",
        "download",
    )


def test_github_release_checkpoint_asset_is_not_mistaken_for_source_code() -> None:
    asset = "https://github.com/acme/models/releases/download/v1/checkpoint"
    repository = "https://github.com/acme/models"

    assert _relation(asset, "checkpoint") == "weights"
    assert _relation(repository, "source") == "source_implementation"


def test_checkpoint_lists_admit_known_suffixless_direct_storage_endpoints() -> None:
    document = """\
## Checkpoints
- `zenodo-model`: [weights](https://zenodo.org/api/records/12345/files/model/content)
- `gcs-model`: [weights](https://storage.googleapis.com/model-bucket/releases/model-v2)
"""
    rows = _parse_checkpoint_list(
        document,
        source="fixture",
        path="README.md",
        heading_pattern=re.compile("Checkpoints", re.I),
        handle_pattern=re.compile(r"`(?P<handle>[^`]+)`"),
        maximum=10,
    )
    assert [row.handle for row in rows] == ["zenodo-model", "gcs-model"]


def test_checkpoint_lists_still_reject_unrecognized_suffixless_pages() -> None:
    document = """\
## Checkpoints
- `not-a-checkpoint`: [page](https://example.test/models/weights)
"""
    import pytest

    with pytest.raises(ValueError, match="exactly one direct checkpoint"):
        _parse_checkpoint_list(
            document,
            source="fixture",
            path="README.md",
            heading_pattern=re.compile("Checkpoints", re.I),
            handle_pattern=re.compile(r"`(?P<handle>[^`]+)`"),
            maximum=10,
        )
