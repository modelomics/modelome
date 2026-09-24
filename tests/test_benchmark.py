from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from modelome.benchmark import evaluate_epoch_benchmark
from modelome.http import HttpResponse
from modelome.models import ArtifactKind, Identifier, ModelHint, SourcePage, SourceRecord
from modelome.storage import Database


class StaticHttpClient:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        response_url: str = "https://cdn.epoch.example/all-models.csv",
    ) -> None:
        self.response = HttpResponse(
            status=status,
            headers=headers or {},
            body=body,
            url=response_url,
        )
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, *, headers: dict[str, str]) -> HttpResponse:
        self.calls.append({"url": url, "headers": headers})
        return self.response


@pytest.fixture
def database(tmp_path: Path) -> Database:
    result = Database(tmp_path / "store")
    result.initialize()
    return result


@pytest.fixture
def benchmark_config() -> dict[str, Any]:
    return {
        "name": "epoch",
        "adapter": "csv",
        "url": "https://epoch.example/all-models.csv",
        "mapping": {"model_field": "Model"},
    }


def _record(
    record_id: str,
    name: str,
    *,
    identifier: Identifier | None = None,
    aliases: tuple[str, ...] = (),
    version: int = 1,
) -> SourceRecord:
    identifiers = () if identifier is None else (identifier,)
    return SourceRecord(
        source_record_id=record_id,
        kind=ArtifactKind.MODEL_CARD,
        canonical_url=f"https://records.example/models/{record_id}",
        title=name,
        raw={"record_id": record_id, "version": version},
        models=(
            ModelHint(
                local_id="model",
                name=name,
                identifiers=identifiers,
                aliases=aliases,
            ),
        ),
    )


def _evaluate(
    database: Database,
    benchmark_config: dict[str, Any],
    body: bytes,
    *,
    client: StaticHttpClient | None = None,
) -> dict[str, Any]:
    return evaluate_epoch_benchmark(
        database,
        benchmark_config,
        client=client or StaticHttpClient(body),
        clock=lambda: datetime(2026, 9, 2, 12, 34, 56, tzinfo=UTC),
    )


def test_exact_names_and_aliases_require_current_active_name_evidence(
    database: Database,
    benchmark_config: dict[str, Any],
) -> None:
    database.ingest_page(
        "huggingface",
        (
            _record(
                "vision-backbone",
                "Vision Backbone",
                aliases=("VBNet",),
            ),
        ),
        {},
        extractor="fixture",
    )

    detail = database.model_detail(database.search_models("Vision Backbone")[0]["id"])
    assert detail is not None
    predicates_by_value = {
        (evidence["predicate"], str(evidence["value"]))
        for evidence in detail["evidence"]
    }
    assert ("alias", "VBNet") in predicates_by_value
    assert any(
        predicate == "documented_by" and "Vision Backbone" in value
        for predicate, value in predicates_by_value
    )

    report = _evaluate(
        database,
        benchmark_config,
        b"Model\nVision Backbone\nVBNet\nUnseen Model\n",
    )

    by_name = {item["name"]: item for item in report["expectations"]}
    assert report["total"] == 3
    assert report["found"] == 2
    assert report["missing"] == 1
    assert report["recall"] == pytest.approx(2 / 3)
    assert by_name["Vision Backbone"]["found"] is True
    assert by_name["Vision Backbone"]["supporting_sources"] == ["huggingface"]
    assert by_name["VBNet"]["found"] is True
    assert by_name["VBNet"]["supporting_sources"] == ["huggingface"]
    assert by_name["Unseen Model"]["found"] is False


def test_stale_and_inactive_supporting_revisions_do_not_count(
    database: Database,
    benchmark_config: dict[str, Any],
) -> None:
    stale_identifier = Identifier("provider:model", "lab/renamed")
    database.ingest_page(
        "openalex",
        (
            _record(
                "renamed-paper",
                "Stale Expected Name",
                identifier=stale_identifier,
                version=1,
            ),
        ),
        {},
        extractor="fixture",
    )
    database.ingest_page(
        "openalex",
        (
            _record(
                "renamed-paper",
                "Current Replacement Name",
                identifier=stale_identifier,
                version=2,
            ),
        ),
        {},
        extractor="fixture",
    )

    inactive = _record("removed-card", "Inactive Expected Name")
    first_run = database.start_run("provider-snapshot")
    database.ingest_page(
        "provider-snapshot",
        SourcePage(
            records=(inactive,),
            next_state={"snapshot": 1},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        ),
        run_id=first_run,
        extractor="fixture",
    )
    database.finish_run(first_run, "complete")

    second_run = database.start_run("provider-snapshot")
    database.ingest_page(
        "provider-snapshot",
        SourcePage(
            records=(_record("surviving-card", "Different Active Model"),),
            next_state={"snapshot": 2},
            complete=True,
            upstream_count=1,
            authoritative_snapshot=True,
        ),
        run_id=second_run,
        extractor="fixture",
    )
    database.finish_run(second_run, "complete")

    report = _evaluate(
        database,
        benchmark_config,
        b"Model\nStale Expected Name\nInactive Expected Name\n",
    )

    assert report["found"] == 0
    assert {item["name"] for item in report["expectations"] if not item["found"]} == {
        "Stale Expected Name",
        "Inactive Expected Name",
    }


def test_report_exposes_duplicates_invalid_rows_and_retrieval_metadata(
    database: Database,
    benchmark_config: dict[str, Any],
) -> None:
    body = (
        b"Model,Domain\n"
        b"Alpha,vision\n"
        b" alpha ,vision\n"
        b",unknown\n"
        b"!!!,unknown\n"
        b"Beta,vision,unexpected\n"
    )
    client = StaticHttpClient(
        body,
        headers={
            "ETag": '"fixture-etag"',
            "Last-Modified": "Wed, 02 Sep 2026 12:00:00 GMT",
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Length": str(len(body)),
        },
    )

    report = _evaluate(database, benchmark_config, body, client=client)

    assert client.calls == [
        {
            "url": "https://epoch.example/all-models.csv",
            "headers": {"Accept": "text/csv"},
        }
    ]
    assert report["benchmark"] == "epoch"
    assert report["kind"] == "external_csv_recall"
    assert report["model_column"] == "Model"
    assert report["raw_rows"] == 5
    assert report["valid_rows"] == 2
    assert report["unique_expectations"] == 1
    assert report["duplicate_count"] == 1
    assert report["invalid_count"] == 3
    assert report["integrity_ok"] is False
    assert report["total"] == 1
    assert report["row_total"] == 2
    assert report["duplicates"] == [
        {
            "row_number": 3,
            "name": "alpha",
            "normalized_name": "alpha",
            "duplicate_of_row": 2,
        }
    ]
    assert report["invalid"] == [
        {"row_number": 4, "reason": "empty_model_name"},
        {
            "row_number": 5,
            "name": "!!!",
            "reason": "model_name_has_no_letters_or_numbers",
        },
        {"row_number": 6, "reason": "unexpected_extra_columns"},
    ]
    assert report["expectations"][0]["row_numbers"] == [2, 3]
    assert report["expectations"][0]["names"] == ["Alpha", "alpha"]

    retrieval = report["retrieval"]
    assert retrieval["requested_url"] == "https://epoch.example/all-models.csv"
    assert retrieval["response_url"] == "https://cdn.epoch.example/all-models.csv"
    assert retrieval["status"] == 200
    assert retrieval["retrieved_at"] == "2026-09-02T12:34:56Z"
    assert retrieval["bytes"] == len(body)
    assert retrieval["sha256"] == hashlib.sha256(body).hexdigest()
    assert len(retrieval["config_sha256"]) == 64
    assert retrieval["etag"] == '"fixture-etag"'
    assert retrieval["last_modified"] == "Wed, 02 Sep 2026 12:00:00 GMT"
    assert retrieval["content_type"] == "text/csv; charset=utf-8"
    assert retrieval["content_length"] == str(len(body))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"", "has no header"),
        (b"Other\nAlpha\n", "missing configured model column 'Model'"),
        (b"Model,Model\nAlpha,Beta\n", "duplicate column(s): Model"),
        (b"Model\n\xff\n", "must be UTF-8"),
        (b'Model\n"unterminated\n', "malformed benchmark CSV"),
        (b"Model\nAlpha,unexpected\n", "has no valid model expectations"),
    ],
)
def test_malformed_or_missing_model_csv_is_rejected(
    database: Database,
    benchmark_config: dict[str, Any],
    body: bytes,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _evaluate(database, benchmark_config, body)


def test_evaluation_does_not_mutate_the_parquet_store(
    database: Database,
    benchmark_config: dict[str, Any],
) -> None:
    database.ingest_page(
        "huggingface",
        (_record("immutable", "Immutable Benchmark Match"),),
        {},
        extractor="fixture",
    )
    head_path = database.root / "HEAD.json"
    commits_path = database.root / "commits"
    head_before = head_path.read_bytes()
    commits_before = sorted(path.name for path in commits_path.iterdir())
    stats_before = database.stats()

    report = _evaluate(
        database,
        benchmark_config,
        b"Model\nImmutable Benchmark Match\nMissing Match\n",
    )

    assert report["found"] == 1
    assert head_path.read_bytes() == head_before
    assert sorted(path.name for path in commits_path.iterdir()) == commits_before
    assert database.stats() == stats_before
