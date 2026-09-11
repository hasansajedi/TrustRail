"""Bypass-oriented corpus for MCP server onboarding controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustrail import (
    MCPApprovedServerPolicy,
    MCPSecretDelivery,
    MCPServerCommand,
    MCPServerFilesystemAccess,
    MCPServerManifest,
    MCPServerNetworkEndpoint,
    MCPServerOnboardingCode,
    MCPServerOnboardingGuard,
    MCPServerOnboardingOperation,
    MCPServerOnboardingPolicy,
    MCPServerOnboardingRequest,
    MCPServerPublisher,
    MCPServerRequestOrigin,
    MCPServerSandboxControl,
    MCPServerSandboxRequirements,
    MCPServerSecretRequirement,
    MCPServerSource,
    MCPServerTransport,
    MCPServerTransportKind,
    onboarding_reference,
)

CORPUS_PATH = Path(__file__).parent.parent / "security_corpus" / "mcp_server_onboarding.json"
CASES: list[dict[str, str]] = json.loads(CORPUS_PATH.read_text())
NOW = datetime(2026, 9, 9, 12, tzinfo=UTC)
CONTROLS = frozenset(
    {
        MCPServerSandboxControl.PROCESS_ISOLATION,
        MCPServerSandboxControl.FILESYSTEM_POLICY,
        MCPServerSandboxControl.NETWORK_POLICY,
        MCPServerSandboxControl.SECRET_BROKER,
    }
)


def _guard() -> MCPServerOnboardingGuard:
    approved = MCPApprovedServerPolicy(
        server_id="acme-search",
        publisher_id="acme",
        publisher_verification_refs=frozenset({onboarding_reference("verified-publisher")}),
        package_name="acme-mcp-search",
        source_uri_prefixes=("https://registry.example/packages",),
        allowed_versions=frozenset({"1.0.0"}),
        allowed_revisions=frozenset({"release-1.0.0"}),
        allowed_artifact_digests=frozenset({"b" * 64}),
        allowed_executables=frozenset({"/opt/acme/search"}),
        allowed_transports=frozenset(
            {MCPServerTransportKind.STDIO, MCPServerTransportKind.STREAMABLE_HTTP}
        ),
        allowed_scopes=frozenset({"search.read"}),
        readable_path_prefixes=frozenset({"/srv/acme/input"}),
        allowed_outbound_hosts=frozenset({"api.example.com"}),
        allowed_secret_names=frozenset({"SEARCH_TOKEN"}),
        allowed_sandbox_profiles=frozenset({"restricted"}),
        required_sandbox_controls=CONTROLS,
    )
    return MCPServerOnboardingGuard(
        MCPServerOnboardingPolicy(
            approved_servers=(approved,),
            trusted_publisher_ids=frozenset({"acme"}),
        )
    )


def _manifest() -> MCPServerManifest:
    return MCPServerManifest(
        server_id="acme-search",
        publisher=MCPServerPublisher(
            publisher_id="acme",
            display_name="Acme",
            verification_ref=onboarding_reference("verified-publisher"),
        ),
        source=MCPServerSource(
            package_name="acme-mcp-search",
            source_uri="https://registry.example/packages/acme-search",
            version="1.0.0",
            revision="release-1.0.0",
            artifact_digest="b" * 64,
        ),
        command=MCPServerCommand(executable="/opt/acme/search"),
        transport=MCPServerTransport(kind=MCPServerTransportKind.STDIO),
        scopes=frozenset({"search.read"}),
        filesystem=MCPServerFilesystemAccess(read_paths=frozenset({"/srv/acme/input/config.json"})),
        outbound_network=(MCPServerNetworkEndpoint(host="api.example.com", port=443),),
        secrets=(
            MCPServerSecretRequirement(
                name="SEARCH_TOKEN",
                secret_ref=onboarding_reference("search-token-record"),
                delivery=MCPSecretDelivery.ENVIRONMENT,
                target="SEARCH_TOKEN",
            ),
        ),
        sandbox=MCPServerSandboxRequirements(
            profile_id="restricted",
            required_controls=CONTROLS,
            max_memory_bytes=134_217_728,
            max_cpu_seconds=60,
            max_processes=8,
        ),
    )


def _mutated_request(case: dict[str, str]) -> MCPServerOnboardingRequest:
    manifest = _manifest()
    mutation = case["mutation"]
    value = case["value"]
    origin = MCPServerRequestOrigin.USER_ACTION
    if mutation == "origin":
        origin = MCPServerRequestOrigin(value)
    elif mutation == "publisher_verification":
        manifest = manifest.model_copy(
            update={"publisher": manifest.publisher.model_copy(update={"verification_ref": None})}
        )
    elif mutation == "publisher_id":
        manifest = manifest.model_copy(
            update={"publisher": manifest.publisher.model_copy(update={"publisher_id": value})}
        )
    elif mutation == "source_uri":
        manifest = manifest.model_copy(
            update={"source": manifest.source.model_copy(update={"source_uri": value})}
        )
    elif mutation == "package_name":
        manifest = manifest.model_copy(
            update={"source": manifest.source.model_copy(update={"package_name": value})}
        )
    elif mutation == "artifact_digest":
        manifest = manifest.model_copy(
            update={"source": manifest.source.model_copy(update={"artifact_digest": value})}
        )
    elif mutation == "version":
        manifest = manifest.model_copy(
            update={"source": manifest.source.model_copy(update={"version": value})}
        )
    elif mutation == "command_argument":
        manifest = manifest.model_copy(
            update={"command": manifest.command.model_copy(update={"arguments": (value,)})}
        )
    elif mutation == "bind_host":
        manifest = manifest.model_copy(
            update={
                "transport": MCPServerTransport(
                    kind=MCPServerTransportKind.STREAMABLE_HTTP,
                    bind_host=value,
                    bind_port=8080,
                )
            }
        )
    elif mutation == "scope":
        manifest = manifest.model_copy(update={"scopes": frozenset({"search.read", value})})
    elif mutation == "read_path":
        manifest = manifest.model_copy(
            update={
                "filesystem": manifest.filesystem.model_copy(
                    update={"read_paths": frozenset({value})}
                )
            }
        )
    elif mutation == "network_host":
        manifest = manifest.model_copy(
            update={"outbound_network": (MCPServerNetworkEndpoint(host=value, port=443),)}
        )
    elif mutation == "secret_name":
        manifest = manifest.model_copy(
            update={
                "secrets": (
                    manifest.secrets[0].model_copy(update={"name": value, "target": value}),
                )
            }
        )
    elif mutation == "sandbox_controls":
        manifest = manifest.model_copy(
            update={
                "sandbox": manifest.sandbox.model_copy(
                    update={"required_controls": frozenset({MCPServerSandboxControl(value)})}
                )
            }
        )
    return MCPServerOnboardingRequest(
        request_id=f"corpus-{case['id']}",
        operation=MCPServerOnboardingOperation.INSTALL,
        origin=origin,
        initiated_by="security-user",
        tenant_id="tenant-a",
        manifest=manifest,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_onboarding_bypass_corpus_is_blocked_without_echoing_payload(case):
    result = _guard().prepare_consent(_mutated_request(case), now=NOW)

    assert result.is_blocked
    assert MCPServerOnboardingCode(case["expected_code"]) in {
        finding.code for finding in result.findings
    }
    assert case["value"] not in result.model_dump_json()
