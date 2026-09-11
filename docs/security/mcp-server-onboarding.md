# MCP server onboarding, consent, and sandbox declarations

An MCP server can gain access to host files, credentials, networks, tools, and
model-visible data as soon as a client installs or connects it. Treat onboarding
as a control-plane operation: server-supplied metadata and requests originating
from web pages, retrieved content, model output, or tool output are untrusted.

`MCPServerOnboardingGuard` validates a typed server manifest before any install,
process launch, connection, credential resolution, tool discovery, or data
exposure. It provides:

- a closed application-owned inventory of server, publisher, canonical package,
  source, executable, transport, and maximum capability policy;
- exact version, revision, artifact-digest, structured command, filesystem,
  outbound-network, brokered-secret, and sandbox declarations;
- rejection of web/model/tool-triggered installation, unverified or substituted
  publishers, unapproved sources, similar package names, non-TLS endpoints,
  non-loopback listeners, and plaintext command/URL credentials;
- a consent prompt containing the complete untruncated command vector and exact
  effective capabilities, all bound to deterministic digests;
- fresh consent when the manifest, scopes, files, network, secrets, transport,
  or sandbox changes;
- external consent, sandbox-attestation, and deployment-policy hooks that fail
  closed on missing, stale, mismatched, invalid, or unavailable evidence; and
- short-lived, integrity-bound permits plus content-free audit events.

Consent grant references are consumed once per guard instance. Distributed or
restart-safe replay prevention belongs in the trusted consent verifier/store and
must use an atomic claim operation.

## Declare the maximum policy

Build policy from reviewed application configuration. Never populate publisher
trust, package identity, sources, scopes, paths, hosts, or secret names from an
MCP server or model response.

```python
from trustrail import (
    MCPApprovedServerPolicy,
    MCPServerOnboardingPolicy,
    MCPServerSandboxControl,
    MCPServerTransportKind,
)

controls = frozenset(
    {
        MCPServerSandboxControl.PROCESS_ISOLATION,
        MCPServerSandboxControl.FILESYSTEM_POLICY,
        MCPServerSandboxControl.NETWORK_POLICY,
        MCPServerSandboxControl.SECRET_BROKER,
        MCPServerSandboxControl.RESOURCE_LIMITS,
    }
)

policy = MCPServerOnboardingPolicy(
    approved_servers=(
        MCPApprovedServerPolicy(
            server_id="search-production",
            publisher_id="reviewed-publisher",
            publisher_verification_refs=frozenset({f"sha256:{'d' * 64}"}),
            package_name="@example/mcp-search",
            source_uri_prefixes=("https://registry.example/packages",),
            allowed_versions=frozenset({"1.4.2"}),
            allowed_revisions=frozenset({"release-1.4.2"}),
            allowed_artifact_digests=frozenset({"a" * 64}),
            allowed_executables=frozenset({"/opt/mcp/search"}),
            allowed_transports=frozenset({MCPServerTransportKind.STDIO}),
            allowed_scopes=frozenset({"search.read"}),
            readable_path_prefixes=frozenset({"/srv/search/input"}),
            writable_path_prefixes=frozenset({"/srv/search/output"}),
            allowed_outbound_hosts=frozenset({"api.example.com"}),
            allowed_secret_names=frozenset({"SEARCH_TOKEN"}),
            allowed_sandbox_profiles=frozenset({"mcp-restricted"}),
            required_sandbox_controls=controls,
        ),
    ),
    trusted_publisher_ids=frozenset({"reviewed-publisher"}),
    require_publisher_verification=True,
    require_source_digest=True,
    require_local_sandbox_attestation=True,
    allow_non_loopback_bind=False,
    consent_ttl_seconds=600,
    permit_ttl_seconds=3600,
)
```

The policy stores secret *names*, never values. A manifest uses
`MCPServerSecretRequirement.secret_ref` to identify an application-owned secret
record and declares a brokered delivery mechanism. Do not serialize credentials
into manifests, commands, endpoint query strings, findings, or audit events.

## Enforce the lifecycle

The host integration owns the order of operations:

1. Parse the complete proposed manifest into `MCPServerManifest`.
2. Call `require_consent_prompt()` before displaying installation UI.
3. Render `command_argv`, `command_display`, publisher/source identity, version,
   and every field in `capabilities` without truncation or hidden expansion.
4. Authenticate the user decision outside model-controlled state and return an
   `MCPServerConsentGrant` through `MCPServerConsentVerifier`.
5. Provision the declared restrictions in an external sandbox. Return an
   `MCPServerSandboxAttestation` through
   `MCPServerSandboxAttestationVerifier`.
6. Optionally call an external admission engine through
   `MCPServerPolicyDecisionHook`; any exception, stale decision, mismatched
   digest, or denial blocks authorization.
7. Call `require_authorization()` before downloading, launching, connecting,
   resolving a secret, or exposing a host capability.
8. Call `require_connection()` with the current manifest and live attestation
   before exposing tools or data. A changed manifest cannot reuse the permit.

```python
from trustrail import MCPServerOnboardingGuard

guard = MCPServerOnboardingGuard(
    policy,
    consent_verifier=application_consent_store,
    attestation_verifier=sandbox_broker,
    policy_hook=deployment_admission,
    audit_sink=restricted_audit_sink,
)

prompt = guard.require_consent_prompt(onboarding_request)
# Render the exact prompt; the application authenticates the resulting grant.
permit = guard.require_authorization(
    onboarding_request,
    prompt,
    authenticated_grant,
    attestation=sandbox_attestation,
)

# Re-read/rebuild the current manifest rather than trusting an earlier object.
guard.require_connection(
    connection_request,
    permit,
    attestation=current_sandbox_attestation,
)
expose_server_tools()
```

The bundled `StaticMCPServerConsentVerifier` and
`StaticMCPServerSandboxAttestationVerifier` are exact-match helpers for tests and
protected in-process state. Production adapters should authenticate records from
the application's identity/consent service and sandbox broker.

## Attestation and deployment boundaries

trustrail defines evidence contracts and verifies their binding. It does not
install packages, start processes, create containers, restrict syscalls, filter
network traffic, mount filesystems, store secrets, verify package signatures, or
authenticate publishers. Those controls must be implemented by the host,
package pipeline, operating system, sandbox, and deployment platform.

The attestation must cover the exact manifest and capability digests, declared
sandbox profile, every required control, and a short validity interval. The
external policy result must cover the exact request, manifest, and capabilities.
Keep verifier keys and trusted records outside model/server state, rotate and
revoke them, synchronize clocks, and use durable audit storage.

## Assumptions, limits, and residual risk

- Manifest fields are assertions until independently resolved and verified by
  trusted host adapters. A SHA-256 value proves equality to an expected value,
  not publisher quality or absence of vulnerabilities.
- Similarity checks catch short edit-distance package substitutions but are not
  a complete Unicode, registry-specific, or brand-abuse detector. Use a private
  allowlisted registry, signature/transparency verification, SBOM and dependency
  scanning, and independent source review.
- Path and host allowlists do not enforce runtime behavior. Apply OS/container
  filesystem policy, DNS/IP-aware egress controls, redirect restrictions,
  resource limits, process isolation, no-new-privileges, and cleanup.
- Loopback binding reduces remote exposure but does not authenticate local
  clients. Use authenticated transports, per-server credentials, and process or
  user isolation.
- Consent can be manipulated by a deceptive surrounding UI. Keep installation
  UI outside untrusted content, identify the publisher/source prominently, avoid
  pre-checked approval, authenticate the user, and re-consent on every material
  definition or capability change.
- In-memory verifiers and audit sinks are not shared across workers and are not
  durable. Production integrations need atomic, persistent, access-controlled
  state and replay/revocation handling.

Combine onboarding with [MCP tool-definition integrity](mcp-tool-integrity.md),
[MCP message integrity](mcp-message-integrity.md), and
[MCP server isolation](mcp-server-isolation.md). Onboarding authorizes a server;
it does not make its tool implementation or outputs trustworthy.
