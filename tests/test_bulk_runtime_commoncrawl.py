from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelome.bulk_runtime import _load_and_project_commoncrawl_shard


def test_commoncrawl_bulk_handoff_materializes_and_finishes_search_pages(
    monkeypatch,
) -> None:
    shard_receipt = object()
    bulk_receipt = SimpleNamespace(shard_receipt=shard_receipt)
    discovery_receipt = object()
    loader = SimpleNamespace(load=lambda record: bulk_receipt)
    materialized: list[object] = []
    projector = SimpleNamespace(
        materialize=lambda receipt: materialized.append(receipt) or discovery_receipt
    )
    outcomes = iter(
        (
            SimpleNamespace(status="partial", complete=False, next_row=10, error=None),
            SimpleNamespace(status="complete", complete=True, next_row=12, error=None),
        )
    )
    calls: list[tuple[object, object, object]] = []

    def ingest(database, selected_projector, receipt):
        calls.append((database, selected_projector, receipt))
        return next(outcomes)

    monkeypatch.setattr("modelome.bulk_runtime.run_commoncrawl_search_ingestion", ingest)
    database = object()

    result = _load_and_project_commoncrawl_shard(
        database,  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        projector,  # type: ignore[arg-type]
        object(),
    )

    assert result is bulk_receipt
    assert materialized == [shard_receipt]
    assert calls == [
        (database, projector, discovery_receipt),
        (database, projector, discovery_receipt),
    ]


def test_commoncrawl_bulk_handoff_fails_closed_on_search_failure(monkeypatch) -> None:
    loader = SimpleNamespace(load=lambda record: SimpleNamespace(shard_receipt=object()))
    projector = SimpleNamespace(materialize=lambda receipt: object())
    monkeypatch.setattr(
        "modelome.bulk_runtime.run_commoncrawl_search_ingestion",
        lambda *args: SimpleNamespace(
            status="failed",
            complete=False,
            next_row=0,
            error="invalid sealed evidence",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid sealed evidence"):
        _load_and_project_commoncrawl_shard(
            object(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            projector,  # type: ignore[arg-type]
            object(),
        )


def test_commoncrawl_bulk_handoff_requires_exact_shard_receipt() -> None:
    loader = SimpleNamespace(load=lambda record: SimpleNamespace(shard_receipt=None))

    with pytest.raises(ValueError, match="missing its shard receipt"):
        _load_and_project_commoncrawl_shard(
            object(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            object(),
        )
