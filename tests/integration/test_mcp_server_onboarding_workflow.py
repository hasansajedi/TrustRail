"""Integration coverage for onboarding before MCP tool exposure."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from trustrail import (
    MCPApprovedServerPolicy,
    MCPServerCommand,
    MCPServerConsentGrant,
    MCPServerManifest,
    MCPServerOnboardingGuard,
    MCPServerOnboardingOperation,
    MCPServerOnboardingPolicy,
    MCPServerOnboardingRequest,
    MCPServerPublisher,
    MCPServerRequestOrigin,
    MCPServerSandboxAttestation,
    MCPServerSandboxControl,
    MCPServerSandboxRequirements,
    MCPServerSource,
    MCPServerTransport,
    MCPServerTransportKind,
    MCPToolDefinition,
    MCPToolDefinitionGuard,
    StaticMCPServerConsentVerifier,
    StaticMCPServerSandboxAttestationVerifier,
    onboarding_reference,
)

NOW = datetime(2026, 9, 9, 14, tzinfo=UTC)
CONTROL = MCPServerSandboxControl.PROCESS_ISOLATION


def test_server_is_verified_before_its_tools_are_exposed():
    approved = MCPApprovedServerPolicy(
        server_id="calendar-server",
        publisher_id="calendar-inc",
        publisher_verification_refs=frozenset({onboarding_reference("publisher-proof")}),
        package_name="calendar-mcp",
        source_uri_prefixes=("https://packages.example/mcp",),
        allowed_versions=frozenset({"2.1.0"}),
        allowed_revisions=frozenset({"v2.1.0"}),
        allowed_artifact_digests=frozenset({"c" * 64}),
        allowed_executables=frozenset({"/opt/mcp/calendar"}),
        allowed_transports=frozenset({MCPServerTransportKind.STDIO}),
        allowed_sandbox_profiles=frozenset({"mcp-local"}),
        required_sandbox_controls=frozenset({CONTROL}),
    )
    policy = MCPServerOnboardingPolicy(
        approved_servers=(approved,),
        trusted_publisher_ids=frozenset({"calendar-inc"}),
    )
    manifest = MCPServerManifest(
        server_id="calendar-server",
        publisher=MCPServerPublisher(
            publisher_id="calendar-inc",
            display_name="Calendar Inc.",
            verification_ref=onboarding_reference("publisher-proof"),
        ),
        source=MCPServerSource(
            package_name="calendar-mcp",
            source_uri="https://packages.example/mcp/calendar",
            version="2.1.0",
            revision="v2.1.0",
            artifact_digest="c" * 64,
        ),
        command=MCPServerCommand(executable="/opt/mcp/calendar", arguments=("serve",)),
        transport=MCPServerTransport(kind=MCPServerTransportKind.STDIO),
        sandbox=MCPServerSandboxRequirements(
            profile_id="mcp-local",
            required_controls=frozenset({CONTROL}),
            max_memory_bytes=134_217_728,
            max_cpu_seconds=60,
            max_processes=4,
        ),
    )
    request = MCPServerOnboardingRequest(
        request_id="calendar-install",
        operation=MCPServerOnboardingOperation.INSTALL,
        origin=MCPServerRequestOrigin.USER_ACTION,
        initiated_by="user-1",
        tenant_id="tenant-1",
        manifest=manifest,
    )
    bootstrap = MCPServerOnboardingGuard(policy)
    prompt = bootstrap.require_consent_prompt(request, now=NOW)
    grant = MCPServerConsentGrant(
        grant_ref=onboarding_reference("calendar-consent"),
        consent_digest=prompt.consent_digest,
        manifest_digest=manifest.manifest_digest,
        capability_digest=manifest.effective_capabilities.capability_digest,
        approved_by="user-1",
        approved_at=NOW + timedelta(seconds=1),
        expires_at=NOW + timedelta(minutes=5),
    )
    attestation = MCPServerSandboxAttestation(
        evidence_ref=onboarding_reference("calendar-sandbox"),
        manifest_digest=manifest.manifest_digest,
        capability_digest=manifest.effective_capabilities.capability_digest,
        sandbox_profile_id="mcp-local",
        enforced_controls=frozenset({CONTROL}),
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    onboarding = MCPServerOnboardingGuard(
        policy,
        consent_verifier=StaticMCPServerConsentVerifier((grant,)),
        attestation_verifier=StaticMCPServerSandboxAttestationVerifier((attestation,)),
    )
    permit = onboarding.require_authorization(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=2),
    )
    connect_request = request.model_copy(update={"operation": MCPServerOnboardingOperation.CONNECT})
    onboarding.require_connection(
        connect_request,
        permit,
        attestation=attestation,
        now=NOW + timedelta(seconds=3),
    )

    definition = MCPToolDefinition(
        server_id="calendar-server",
        name="calendar.list",
        description="List the user's calendar events.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
        outputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    definitions = MCPToolDefinitionGuard(b"onboarding-integration-signing-key")
    discovery = definitions.require_discovery((definition,), now=NOW + timedelta(seconds=4))

    assert len(discovery.fingerprints) == 1
    assert permit.server_id == definition.server_id
