from __future__ import annotations

from typing import Any

import pytest

from modelome.sources.aws_sagemaker_client import create_signed_sagemaker_client


class FakeSageMakerClient:
    def list_hub_contents(self, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("client construction must not make API requests")


class FakeSession:
    def __init__(self, *, region_name: str) -> None:
        self.region_name = region_name
        self.client_services: list[str] = []
        self.client_instance = FakeSageMakerClient()

    def client(self, service_name: str) -> FakeSageMakerClient:
        self.client_services.append(service_name)
        return self.client_instance


class FakeBoto3:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def Session(self, *, region_name: str) -> FakeSession:
        session = FakeSession(region_name=region_name)
        self.sessions.append(session)
        return session


def test_sagemaker_client_uses_explicit_config_region_without_network_calls() -> None:
    boto3 = FakeBoto3()

    client = create_signed_sagemaker_client(
        region_name="us-west-2",
        environ={"AWS_REGION": "us-east-1"},
        boto3_module=boto3,
    )

    assert client is boto3.sessions[0].client_instance
    assert boto3.sessions[0].region_name == "us-west-2"
    assert boto3.sessions[0].client_services == ["sagemaker"]


def test_sagemaker_client_does_not_forward_or_log_credential_values(caplog) -> None:
    boto3 = FakeBoto3()
    secrets = {
        "AWS_ACCESS_KEY_ID": "test-access-key-secret",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key-secret",
        "AWS_SESSION_TOKEN": "test-session-token-secret",
    }

    client = create_signed_sagemaker_client(
        region_name="us-west-2",
        environ={"AWS_REGION": "us-east-1", **secrets},
        boto3_module=boto3,
    )

    assert client is boto3.sessions[0].client_instance
    assert boto3.sessions[0].region_name == "us-west-2"
    assert all(secret not in caplog.text for secret in secrets.values())


@pytest.mark.parametrize(
    ("environ", "expected_region"),
    [
        ({"AWS_REGION": "eu-west-1"}, "eu-west-1"),
        ({"AWS_DEFAULT_REGION": "us-gov-west-1"}, "us-gov-west-1"),
    ],
)
def test_sagemaker_client_uses_explicit_environment_region(
    environ: dict[str, str], expected_region: str
) -> None:
    boto3 = FakeBoto3()

    create_signed_sagemaker_client(environ=environ, boto3_module=boto3)

    assert boto3.sessions[0].region_name == expected_region


@pytest.mark.parametrize("region", [None, "", "not a region", "us-west"])
def test_sagemaker_client_rejects_missing_or_invalid_region(region: str | None) -> None:
    with pytest.raises(ValueError, match="explicit AWS region is required"):
        create_signed_sagemaker_client(region_name=region, environ={})
