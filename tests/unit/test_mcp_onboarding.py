"""Unit coverage for secure MCP server onboarding and consent."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from trustrail import (
    GuardAction,
    MCPApprovedServerPolicy,
    MCPSecretDelivery,
    MCPServerCommand,
    MCPServerConsentGrant,
    MCPServerExternalPolicyDecision,
    MCPServerFilesystemAccess,
    MCPServerManifest,
    MCPServerNetworkEndpoint,
    MCPServerOnboardingCode,
    MCPServerOnboardingError,
    MCPServerOnboardingGuard,
    MCPServerOnboardingOperation,
    MCPServerOnboardingPolicy,
    MCPServerOnboardingRequest,
    MCPServerPublisher,
    MCPServerRequestOrigin,
    MCPServerSandboxAttestation,
    MCPServerSandboxControl,
    MCPServerSandboxRequirements,
    MCPServerSecretRequirement,
    MCPServerSource,
    MCPServerTransport,
    MCPServerTransportKind,
    MemoryMCPServerOnboardingAuditSink,
    StaticMCPServerConsentVerifier,
    StaticMCPServerSandboxAttestationVerifier,
    onboarding_reference,
)

NOW = datetime(2026, 9, 9, 10, tzinfo=UTC)
CONTROLS = frozenset(
    {
        MCPServerSandboxControl.PROCESS_ISOLATION,
        MCPServerSandboxControl.FILESYSTEM_POLICY,
        MCPServerSandboxControl.NETWORK_POLICY,
        MCPServerSandboxControl.SECRET_BROKER,
        MCPServerSandboxControl.RESOURCE_LIMITS,
    }
)


def _policy(**updates: object) -> MCPServerOnboardingPolicy:
    server = MCPApprovedServerPolicy(
        server_id="acme-search",
        publisher_id="acme",
        publisher_verification_refs=frozenset({onboarding_reference("verified-acme-publisher")}),
        package_name="@acme/mcp-search",
        source_uri_prefixes=("https://registry.example/packages",),
        allowed_versions=frozenset({"1.4.2"}),
        allowed_revisions=frozenset({"release-1.4.2"}),
        allowed_artifact_digests=frozenset({"a" * 64}),
        allowed_executables=frozenset({"/usr/local/bin/acme-search"}),
        allowed_transports=frozenset({MCPServerTransportKind.STDIO}),
        allowed_scopes=frozenset({"search.read"}),
        readable_path_prefixes=frozenset({"/srv/search/input"}),
        writable_path_prefixes=frozenset({"/srv/search/output"}),
        allowed_outbound_hosts=frozenset({"api.example.com"}),
        allowed_secret_names=frozenset({"SEARCH_TOKEN"}),
        allowed_sandbox_profiles=frozenset({"mcp-restricted"}),
        required_sandbox_controls=CONTROLS,
    )
    values: dict[str, object] = {
        "approved_servers": (server,),
        "trusted_publisher_ids": frozenset({"acme"}),
    }
    values.update(updates)
    return MCPServerOnboardingPolicy(**values)


def _manifest(**updates: object) -> MCPServerManifest:
    values: dict[str, object] = {
        "server_id": "acme-search",
        "publisher": MCPServerPublisher(
            publisher_id="acme",
            display_name="Acme Security",
            verification_ref=onboarding_reference("verified-acme-publisher"),
        ),
        "source": MCPServerSource(
            package_name="@acme/mcp-search",
            source_uri="https://registry.example/packages/acme-search",
            version="1.4.2",
            revision="release-1.4.2",
            artifact_digest="a" * 64,
        ),
        "command": MCPServerCommand(
            executable="/usr/local/bin/acme-search",
            arguments=("--mode", "read only", "--config", "/srv/search/input/config.json"),
            working_directory="/srv/search/input",
        ),
        "transport": MCPServerTransport(kind=MCPServerTransportKind.STDIO),
        "scopes": frozenset({"search.read"}),
        "filesystem": MCPServerFilesystemAccess(
            read_paths=frozenset({"/srv/search/input/config.json"}),
            write_paths=frozenset({"/srv/search/output/cache"}),
        ),
        "outbound_network": (MCPServerNetworkEndpoint(host="api.example.com", port=443),),
        "secrets": (
            MCPServerSecretRequirement(
                name="SEARCH_TOKEN",
                secret_ref=onboarding_reference("vault-record-search-token"),
                delivery=MCPSecretDelivery.ENVIRONMENT,
                target="SEARCH_TOKEN",
            ),
        ),
        "sandbox": MCPServerSandboxRequirements(
            profile_id="mcp-restricted",
            required_controls=CONTROLS,
            max_memory_bytes=256 * 1024 * 1024,
            max_cpu_seconds=120,
            max_processes=16,
        ),
    }
    values.update(updates)
    return MCPServerManifest(**values)


def _request(**updates: object) -> MCPServerOnboardingRequest:
    values: dict[str, object] = {
        "request_id": "request-31",
        "operation": MCPServerOnboardingOperation.INSTALL,
        "origin": MCPServerRequestOrigin.USER_ACTION,
        "initiated_by": "user-42",
        "tenant_id": "tenant-a",
        "manifest": _manifest(),
    }
    values.update(updates)
    return MCPServerOnboardingRequest(**values)


def _grant(request: MCPServerOnboardingRequest, prompt) -> MCPServerConsentGrant:
    return MCPServerConsentGrant(
        grant_ref=onboarding_reference("authenticated-consent-record"),
        consent_digest=prompt.consent_digest,
        manifest_digest=request.manifest.manifest_digest,
        capability_digest=request.manifest.effective_capabilities.capability_digest,
        approved_by="user-42",
        approved_at=NOW + timedelta(seconds=5),
        expires_at=NOW + timedelta(minutes=5),
    )


def _attestation(request: MCPServerOnboardingRequest) -> MCPServerSandboxAttestation:
    return MCPServerSandboxAttestation(
        evidence_ref=onboarding_reference("sandbox-deployment-42"),
        manifest_digest=request.manifest.manifest_digest,
        capability_digest=request.manifest.effective_capabilities.capability_digest,
        sandbox_profile_id="mcp-restricted",
        enforced_controls=CONTROLS,
        issued_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


class _AllowPolicyHook:
    def evaluate(self, request, attestation):
        assert attestation is not None
        return MCPServerExternalPolicyDecision(
            decision_ref=onboarding_reference("opa-decision-31"),
            request_digest=request.request_digest,
            manifest_digest=request.manifest.manifest_digest,
            capability_digest=request.manifest.effective_capabilities.capability_digest,
            allowed=True,
            expires_at=NOW + timedelta(minutes=30),
        )


def test_complete_onboarding_binds_consent_sandbox_policy_and_connection():
    request = _request()
    audit = MemoryMCPServerOnboardingAuditSink()
    prompt_result = MCPServerOnboardingGuard(_policy(), audit_sink=audit).prepare_consent(
        request, now=NOW
    )

    assert prompt_result.action == GuardAction.REQUIRE_APPROVAL
    assert prompt_result.consent_prompt is not None
    prompt = prompt_result.consent_prompt
    assert prompt.command_argv == request.manifest.command.argv
    assert prompt.command_display == (
        "/usr/local/bin/acme-search --mode 'read only' --config /srv/search/input/config.json"
    )
    assert prompt.command_truncated is False
    assert prompt.publisher == request.manifest.publisher
    assert prompt.source == request.manifest.source
    assert prompt.capabilities == request.manifest.effective_capabilities

    grant = _grant(request, prompt)
    attestation = _attestation(request)
    guard = MCPServerOnboardingGuard(
        _policy(),
        consent_verifier=StaticMCPServerConsentVerifier((grant,)),
        attestation_verifier=StaticMCPServerSandboxAttestationVerifier((attestation,)),
        policy_hook=_AllowPolicyHook(),
        audit_sink=audit,
    )
    permit = guard.require_authorization(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=10),
    )
    verified = guard.require_connection(
        request.model_copy(update={"operation": MCPServerOnboardingOperation.CONNECT}),
        permit,
        attestation=attestation,
        now=NOW + timedelta(seconds=20),
    )

    assert verified.manifest_digest == request.manifest.manifest_digest
    assert verified.capability_digest == request.manifest.effective_capabilities.capability_digest
    assert verified.sandbox_evidence_ref == attestation.evidence_ref
    assert verified.policy_decision_ref == onboarding_reference("opa-decision-31")
    assert [event.action for event in audit.events] == [
        GuardAction.REQUIRE_APPROVAL,
        GuardAction.ALLOW,
        GuardAction.ALLOW,
    ]
    assert "acme-search" not in audit.events[-1].model_dump_json()
    assert request.manifest.command.display not in audit.events[-1].model_dump_json()


@pytest.mark.parametrize(
    ("onboarding_request", "expected"),
    [
        (
            _request(origin=MCPServerRequestOrigin.WEB_CONTENT),
            MCPServerOnboardingCode.ORIGIN_DENIED,
        ),
        (
            _request(
                manifest=_manifest(
                    command=MCPServerCommand(
                        executable="/usr/local/bin/acme-search",
                        arguments=("--token", "plaintext-value"),
                    )
                )
            ),
            MCPServerOnboardingCode.PLAINTEXT_CREDENTIAL,
        ),
        (
            _request(
                manifest=_manifest(
                    source=MCPServerSource(
                        package_name="@acme/mcp-searhc",
                        source_uri="https://registry.example/packages/acme-search",
                        version="1.4.2",
                        revision="release-1.4.2",
                        artifact_digest="a" * 64,
                    )
                )
            ),
            MCPServerOnboardingCode.TYPOSQUATTING_SUSPECTED,
        ),
    ],
)
def test_unsafe_onboarding_is_blocked_without_echoing_values(onboarding_request, expected):
    result = MCPServerOnboardingGuard(_policy()).prepare_consent(onboarding_request, now=NOW)

    assert result.is_blocked
    assert expected in {finding.code for finding in result.findings}
    assert result.consent_prompt is None
    assert "plaintext-value" not in result.model_dump_json()


def test_authorization_fails_closed_without_consent_or_attestation_verifiers():
    request = _request()
    guard = MCPServerOnboardingGuard(_policy())
    prompt = guard.require_consent_prompt(request, now=NOW)
    grant = _grant(request, prompt)
    attestation = _attestation(request)

    result = guard.authorize(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=10),
    )

    assert result.is_blocked
    assert {finding.code for finding in result.findings} == {
        MCPServerOnboardingCode.CONSENT_UNVERIFIED,
        MCPServerOnboardingCode.ATTESTATION_INVALID,
    }
    with pytest.raises(MCPServerOnboardingError):
        guard.require_authorization(
            request,
            prompt,
            grant,
            attestation=attestation,
            now=NOW + timedelta(seconds=10),
        )


def test_consent_grant_is_single_use():
    request = _request()
    bootstrap = MCPServerOnboardingGuard(_policy())
    prompt = bootstrap.require_consent_prompt(request, now=NOW)
    grant = _grant(request, prompt)
    attestation = _attestation(request)
    guard = MCPServerOnboardingGuard(
        _policy(),
        consent_verifier=StaticMCPServerConsentVerifier((grant,)),
        attestation_verifier=StaticMCPServerSandboxAttestationVerifier((attestation,)),
    )
    first = guard.authorize(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=10),
    )
    replay = guard.authorize(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=11),
    )

    assert first.is_allowed
    assert replay.is_blocked
    assert {finding.code for finding in replay.findings} == {
        MCPServerOnboardingCode.CONSENT_INVALID
    }


def test_connection_rejects_post_consent_capability_growth():
    request = _request()
    bootstrap = MCPServerOnboardingGuard(_policy())
    prompt = bootstrap.require_consent_prompt(request, now=NOW)
    grant = _grant(request, prompt)
    attestation = _attestation(request)
    guard = MCPServerOnboardingGuard(
        _policy(),
        consent_verifier=StaticMCPServerConsentVerifier((grant,)),
        attestation_verifier=StaticMCPServerSandboxAttestationVerifier((attestation,)),
    )
    permit = guard.require_authorization(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=10),
    )
    changed_manifest = request.manifest.model_copy(
        update={"scopes": frozenset({"search.read", "admin.write"})}
    )
    changed_request = request.model_copy(update={"manifest": changed_manifest})

    result = guard.verify_connection(
        changed_request,
        permit,
        attestation=attestation,
        now=NOW + timedelta(seconds=20),
    )

    codes = {finding.code for finding in result.findings}
    assert result.is_blocked
    assert MCPServerOnboardingCode.SCOPE_DENIED in codes
    assert MCPServerOnboardingCode.PERMIT_INVALID in codes
    assert MCPServerOnboardingCode.ATTESTATION_MISMATCH in codes


def test_previous_permit_surfaces_manifest_and_capability_reconsent():
    request = _request()
    bootstrap = MCPServerOnboardingGuard(_policy())
    prompt = bootstrap.require_consent_prompt(request, now=NOW)
    grant = _grant(request, prompt)
    attestation = _attestation(request)
    guard = MCPServerOnboardingGuard(
        _policy(),
        consent_verifier=StaticMCPServerConsentVerifier((grant,)),
        attestation_verifier=StaticMCPServerSandboxAttestationVerifier((attestation,)),
    )
    permit = guard.require_authorization(
        request,
        prompt,
        grant,
        attestation=attestation,
        now=NOW + timedelta(seconds=10),
    )
    expanded_policy = _policy(
        approved_servers=(
            _policy()
            .approved_servers[0]
            .model_copy(update={"allowed_scopes": frozenset({"search.read", "search.write"})}),
        )
    )
    changed = request.model_copy(
        update={
            "manifest": request.manifest.model_copy(
                update={"scopes": frozenset({"search.read", "search.write"})}
            )
        }
    )

    result = MCPServerOnboardingGuard(expanded_policy).prepare_consent(
        changed, previous_permit=permit, now=NOW + timedelta(minutes=1)
    )

    assert result.requires_consent
    assert {finding.code for finding in result.findings} == {
        MCPServerOnboardingCode.MANIFEST_CHANGED,
        MCPServerOnboardingCode.SCOPE_GROWTH,
        MCPServerOnboardingCode.CONSENT_REQUIRED,
    }


def test_manifest_models_reject_ambiguous_transports_and_plain_secret_values():
    with pytest.raises(ValidationError):
        MCPServerTransport(
            kind=MCPServerTransportKind.STREAMABLE_HTTP,
            endpoint_uri="https://mcp.example/api",
            bind_host="127.0.0.1",
            bind_port=8080,
        )
    with pytest.raises(ValidationError):
        MCPServerSecretRequirement(
            name="SEARCH_TOKEN",
            secret_ref="plaintext-token",
            delivery=MCPSecretDelivery.ENVIRONMENT,
            target="SEARCH_TOKEN",
        )
