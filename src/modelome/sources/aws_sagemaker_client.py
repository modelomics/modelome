"""Create a signed SageMaker client using boto3's standard credential chain."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

_REGION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d+$")


def create_signed_sagemaker_client(
    *,
    region_name: str | None = None,
    environ: Mapping[str, str] | None = None,
    boto3_module: Any | None = None,
) -> Any:
    """Build a SageMaker client without making an API request.

    ``region_name`` should normally come from source configuration. If omitted,
    an explicitly configured ``AWS_REGION`` or ``AWS_DEFAULT_REGION`` is used.
    Credentials are resolved by boto3/botocore's standard chain when a request
    is signed; this helper never reads, logs, or returns credential values.

    ``boto3_module`` is an injection point for tests and embedded runtimes. The
    normal path imports boto3 lazily, so importing this module has no boto3
    dependency.
    """

    environment = os.environ if environ is None else environ
    region = _text(region_name) or _text(environment.get("AWS_REGION")) or _text(
        environment.get("AWS_DEFAULT_REGION")
    )
    if not region or not _REGION.fullmatch(region):
        raise ValueError(
            "an explicit AWS region is required via region_name, AWS_REGION, "
            "or AWS_DEFAULT_REGION"
        )

    if boto3_module is None:
        try:
            import boto3 as boto3_module  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError(
                "boto3 is required to create a signed SageMaker client"
            ) from error

    session = boto3_module.Session(region_name=region)
    return session.client("sagemaker")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


__all__ = ["create_signed_sagemaker_client"]
