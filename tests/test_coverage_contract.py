import ast
import csv
import re
from pathlib import Path

from modelome.http import HttpResponse
from modelome.sources.csv_source import CsvSourceAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "coverage_models.csv"
RUNTIME_ROOT = PROJECT_ROOT / "src" / "modelome"
FIXTURE_URL = "https://fixtures.example.test/coverage_models.csv"

REQUIRED_BUCKETS = {
    "classic_cv",
    "gan",
    "transformer",
    "image_diffusion",
    "chemistry_materials_diffusion",
    "physics_earth_diffusion",
    "biology_diffusion",
    "biology_neural",
    "medicine_neural",
    "chemistry_neural",
    "physics_neural",
    "materials_neural",
}

MINIMUM_BUCKET_SIZES = {
    "classic_cv": 4,
    "gan": 7,
    "transformer": 7,
    "image_diffusion": 5,
    "chemistry_materials_diffusion": 3,
    "physics_earth_diffusion": 2,
    "biology_diffusion": 4,
    "biology_neural": 2,
    "medicine_neural": 2,
    "chemistry_neural": 2,
    "physics_neural": 2,
    "materials_neural": 2,
}


class FixtureClient:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def get(self, url: str, *, headers=None) -> HttpResponse:
        assert url == FIXTURE_URL
        return HttpResponse(
            status=200,
            headers={"content-type": "text/csv", "etag": '"coverage-v1"'},
            body=self.body,
            url=url,
        )


def _fixture_rows() -> list[dict[str, str]]:
    with FIXTURE_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _executable_string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings: set[int] = set()
    documented_nodes = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for owner in ast.walk(tree):
        if not isinstance(owner, documented_nodes) or not owner.body:
            continue
        first = owner.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _normalized_words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def test_generic_csv_adapter_discovers_every_fixture_row_and_bucket() -> None:
    rows = _fixture_rows()
    adapter = CsvSourceAdapter(
        name="coverage-contract",
        url=FIXTURE_URL,
        mapping={
            "id_fields": ["Model", "Bucket", "Domain"],
            "title_field": "Model",
            "body_fields": ["Bucket", "Domain"],
            "link_fields": ["Reference"],
            "model_field": "Model",
        },
        client=FixtureClient(FIXTURE_PATH.read_bytes()),
    )

    page = adapter.fetch_page({})

    assert page.complete
    assert page.upstream_count == len(rows)
    assert len(page.records) == len(rows)
    assert len({record.source_record_id for record in page.records}) == len(rows)
    assert all(len(record.models) == 1 for record in page.records)
    assert [record.models[0].name for record in page.records] == [
        row["Model"] for row in rows
    ]
    assert all(record.models[0].locator for record in page.records)
    assert {record.raw["Bucket"] for record in page.records} >= REQUIRED_BUCKETS
    bucket_sizes = {
        bucket: sum(row["Bucket"] == bucket for row in rows)
        for bucket in REQUIRED_BUCKETS
    }
    assert all(
        bucket_sizes[bucket] >= minimum
        for bucket, minimum in MINIMUM_BUCKET_SIZES.items()
    )


def test_fixture_model_names_are_not_runtime_seed_constants() -> None:
    # Only Model-column values are prohibited. Generic schema strings such as
    # "model", "name", "category", and "domain" remain legitimate runtime metadata.
    model_names = [row["Model"] for row in _fixture_rows()]
    runtime_literals = {
        path: _executable_string_literals(path) for path in RUNTIME_ROOT.rglob("*.py")
    }
    # These occurrences are source identifiers or generic model-context terms,
    # not fixture seeds.
    source_identifier_literals = {
        "Transformer": {
            "src/modelome/sources/catalog.py",
            "src/modelome/sources/github_release_assets.py",
            "src/modelome/sources/gitlab_release_assets.py",
            "src/modelome/sources/robotics_registry_v3.py",
        },
        "nnU-Net": {"src/modelome/sources/nnunet_registry.py"},
        "RFdiffusion": {
            "src/modelome/sources/catalog.py",
            "src/modelome/sources/rfdiffusion_registry.py",
        },
        "DimeNet": {
            "src/modelome/sources/catalog.py",
            "src/modelome/sources/pyg_dimenet_checkpoints.py",
        },
        "SchNet": {
            "src/modelome/sources/catalog.py",
            "src/modelome/sources/pyg_schnet_qm9_registry.py",
            "src/modelome/sources/cgschnet_pretrained_bundle.py",
        },
    }
    violations: dict[str, list[str]] = {}

    for name in model_names:
        needle = f" {_normalized_words(name)} "
        offenders = [
            str(path.relative_to(PROJECT_ROOT))
            for path, literals in runtime_literals.items()
            if any(needle in f" {_normalized_words(literal)} " for literal in literals)
            and str(path.relative_to(PROJECT_ROOT))
            not in source_identifier_literals.get(name, set())
        ]
        if offenders:
            violations[name] = offenders

    assert not violations, f"coverage examples leaked into runtime seed constants: {violations}"


def test_aiayn_is_an_independent_transformer_evaluation_sentinel() -> None:
    row = next(item for item in _fixture_rows() if item["Model"] == "Transformer")

    assert row["Bucket"] == "transformer"
    assert row["Reference"] == (
        "https://proceedings.neurips.cc/paper/2017/file/"
        "3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    )
