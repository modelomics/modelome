from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelome.bulk_runtime import _load_and_project_software_heritage_shard


def test_software_heritage_bulk_handoff_finishes_github_projection_pages(
    monkeypatch,
) -> None:
    shard_receipt = object()
    bulk_receipt = SimpleNamespace(shard_receipt=shard_receipt)
    loader = SimpleNamespace(load=lambda record: bulk_receipt)
    outcomes = iter(
        (
            SimpleNamespace(
                status="partial",
                complete=False,
                start_row=0,
                next_row=10,
                error=None,
            ),
            SimpleNamespace(
                status="complete",
                complete=True,
                start_row=10,
                next_row=12,
                error=None,
            ),
        )
    )
    calls: list[tuple[object, object, object]] = []

    def project(database, landing_zone, *, source_receipt):
        calls.append((database, landing_zone, source_receipt))
        return next(outcomes)

    monkeypatch.setattr(
        "modelome.bulk_runtime.run_software_heritage_github_shard_projection",
        project,
    )
    database = object()
    landing_zone = object()

    result = _load_and_project_software_heritage_shard(
        database,  # type: ignore[arg-type]
        landing_zone,  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        object(),
    )

    assert result is bulk_receipt
    assert calls == [
        (database, landing_zone, shard_receipt),
        (database, landing_zone, shard_receipt),
    ]


def test_software_heritage_bulk_handoff_fails_closed_on_projection_failure(
    monkeypatch,
) -> None:
    loader = SimpleNamespace(load=lambda record: SimpleNamespace(shard_receipt=object()))
    monkeypatch.setattr(
        "modelome.bulk_runtime.run_software_heritage_github_shard_projection",
        lambda *args, **kwargs: SimpleNamespace(
            status="failed",
            complete=False,
            start_row=0,
            next_row=0,
            error="invalid committed evidence",
        ),
    )

    with pytest.raises(RuntimeError, match="invalid committed evidence"):
        _load_and_project_software_heritage_shard(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            object(),
        )


def test_software_heritage_bulk_handoff_fails_closed_without_progress(
    monkeypatch,
) -> None:
    loader = SimpleNamespace(load=lambda record: SimpleNamespace(shard_receipt=object()))
    monkeypatch.setattr(
        "modelome.bulk_runtime.run_software_heritage_github_shard_projection",
        lambda *args, **kwargs: SimpleNamespace(
            status="partial",
            complete=False,
            start_row=20,
            next_row=20,
            error=None,
        ),
    )

    with pytest.raises(RuntimeError, match="did not advance"):
        _load_and_project_software_heritage_shard(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            object(),
        )


def test_software_heritage_bulk_handoff_requires_exact_shard_receipt() -> None:
    loader = SimpleNamespace(load=lambda record: SimpleNamespace(shard_receipt=None))

    with pytest.raises(ValueError, match="missing its shard receipt"):
        _load_and_project_software_heritage_shard(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            loader,  # type: ignore[arg-type]
            object(),
        )
