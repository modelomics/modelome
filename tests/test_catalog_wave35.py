from __future__ import annotations

from modelome.sources.catalog import load_source_configs, load_sources


def test_wave35_public_sources_are_enabled_and_constructible() -> None:
    expected = {
        "boardwalk-public-models",
        "flower-vla-collection",
        "github-curated-model-release-assets-17",
        "mars-zenodo-checkpoints",
        "onnx-model-zoo-hub",
        "paddlenlp-unimo-pretrained-registry",
        "pmtransformer-figshare-checkpoint",
        "sony-woosh-release-assets",
        "timm-legacy-dla-v0613",
    }
    configs = load_source_configs()
    names = [config["name"] for config in configs]
    assert len(names) == len(set(names))
    assert expected <= set(names)
    assert all(config.get("enabled") is True for config in configs if config["name"] in expected)

    sources = load_sources(environ={})
    assert expected <= set(sources)
    assert all(sources[name].name == name for name in expected)
    ngc = sources["ngc-models"]
    assert ngc.include_all_versions is True
    assert ngc.page_size == 25
