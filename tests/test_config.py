from modelome.config import Settings, default_source_catalog
from modelome.sources.catalog import load_benchmark_configs, load_source_configs


def test_catalog_is_available_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELOME_SOURCES", raising=False)
    monkeypatch.chdir(tmp_path)
    assert default_source_catalog().is_file()
    assert len(load_source_configs()) > 200
    assert load_benchmark_configs()[0]["name"] == "epoch"
    assert Settings.from_env().sources == default_source_catalog()


def test_catalog_override_is_shared_by_library_and_cli(tmp_path, monkeypatch):
    catalog = tmp_path / "custom.toml"
    catalog.write_text('[[source]]\nname = "custom"\nadapter = "csv"\n')
    monkeypatch.setenv("MODELOME_SOURCES", str(catalog))
    assert Settings.from_env().sources == catalog
    assert load_source_configs()[0]["name"] == "custom"
