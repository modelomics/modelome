from __future__ import annotations

from modelome.sources.openmmlab import _parse_metafile


def test_openmmlab_model_alias_lists_are_all_retained() -> None:
    _, models = _parse_metafile(
        """Models:
  - Name: model-inline
    Alias: [short-name, 'legacy, name']
  - Name: model-block
    Alias:
      - first-alias
      - second-alias
    Config: configs/model.py
"""
    )

    assert [model.aliases for model in models] == [
        ["short-name", "legacy, name"],
        ["first-alias", "second-alias"],
    ]
    assert models[1].config == "configs/model.py"
