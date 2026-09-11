"""Fail-closed MCP server installation and connection onboarding."""

from __future__ import annotations

import ipaddress
import re
import threading
import unicodedata
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Protocol
from urllib.parse import parse_qsl, urlsplit

from trustrail.exceptions import MCPServerOnboardingError
from trustrail.models.enums import GuardAction, Severity
from trustrail.models.mcp_onboarding import (
    MCPApprovedServerPolicy,
    MCPServerConsentGrant,
    MCPServerConsentPrompt,
    MCPServerExternalPolicyDecision,
    MCPServerInstallationPermit,
    MCPServerOnboardingAuditEvent,
    MCPServerOnboardingCode,
    MCPServerOnboardingFinding,
    MCPServerOnboardingPolicy,
    MCPServerOnboardingRequest,
    MCPServerOnboardingResult,
    MCPServerSandboxAttestation,
    MCPServerTransportKind,
    onboarding_reference,
    utcnow,
)

_CREDENTIAL_NAME_RE = re.compile(
    r"(?:api[-_]?key|access[-_]?token|auth[-_]?token|client[-_]?secret|password|passwd|secret|token)",
    re.IGNORECASE,
)
_CREDENTIAL_OPTION_RE = re.compile(
    r"^--?(?:api[-_]?key|access[-_]?token|auth[-_]?token|authorization|client[-_]?secret|credential|password|passwd|secret|token)(?:=|$)",
    re.IGNORECASE,
)
_PLAINTEXT_AUTH_RE = re.compile(
    r"(?:^|\s)(?:authorization\s*[:=]\s*)?(?:basic|bearer)\s+\S+",
    re.IGNORECASE,
)


class MCPServerConsentVerifier(Protocol):
    """Authenticate a consent grant issued outside model-controlled state."""

    def verify_consent(
        self,
        grant: MCPServerConsentGrant,
        prompt: MCPServerConsentPrompt,
        request: MCPServerOnboardingRequest,
    ) -> bool:
        """Return whether trusted application state issued the exact grant."""
        ...


class MCPServerSandboxAttestationVerifier(Protocol):
    """Verify evidence returned by an external sandbox or deployment system."""

    def verify_attestation(
        self,
        attestation: MCPServerSandboxAttestation,
        request: MCPServerOnboardingRequest,
    ) -> bool:
        """Return whether the attestation is authentic for this request."""
        ...


class MCPServerPolicyDecisionHook(Protocol):
    """Evaluate a deployment using an application-owned external policy engine."""

    def evaluate(
        self,
        request: MCPServerOnboardingRequest,
        attestation: MCPServerSandboxAttestation | None,
    ) -> MCPServerExternalPolicyDecision:
        """Return an integrity-bound allow or deny decision."""
        ...


class MCPServerOnboardingAuditSink(Protocol):
    """Persist content-free MCP server onboarding events."""

    def emit(self, event: MCPServerOnboardingAuditEvent) -> None:
        """Persist one event without receiving commands, paths, or credentials."""
        ...


class MemoryMCPServerOnboardingAuditSink:
    """Thread-safe bounded onboarding audit sink for tests and development."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[MCPServerOnboardingAuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, event: MCPServerOnboardingAuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[MCPServerOnboardingAuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class StaticMCPServerConsentVerifier:
    """Exact-match consent verifier for tests and protected application state."""

    def __init__(self, grants: Iterable[MCPServerConsentGrant]) -> None:
        self._grants = tuple(grant.model_copy(deep=True) for grant in grants)

    def verify_consent(
        self,
        grant: MCPServerConsentGrant,
        prompt: MCPServerConsentPrompt,
        request: MCPServerOnboardingRequest,
    ) -> bool:
        del prompt, request
        return any(grant == expected for expected in self._grants)


class StaticMCPServerSandboxAttestationVerifier:
    """Exact-match attestation verifier for tests and protected application state."""

    def __init__(self, attestations: Iterable[MCPServerSandboxAttestation]) -> None:
        self._attestations = tuple(item.model_copy(deep=True) for item in attestations)

    def verify_attestation(
        self,
        attestation: MCPServerSandboxAttestation,
        request: MCPServerOnboardingRequest,
    ) -> bool:
        del request
        return any(attestation == expected for expected in self._attestations)


def _identity_skeleton(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() and not unicodedata.combining(character)
    )


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        next_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            next_row.append(
                min(
                    next_row[-1] + 1,
                    row[right_index] + 1,
                    row[right_index - 1] + (left_character != right_character),
                )
            )
        row = next_row
    return row[-1]


def _uri_within_prefix(uri: str, prefix: str) -> bool:
    normalized_prefix = prefix.rstrip("/")
    return uri == normalized_prefix or uri.startswith(f"{normalized_prefix}/")


def _path_within_prefix(path: str, prefix: str) -> bool:
    try:
        PurePosixPath(path).relative_to(PurePosixPath(prefix))
    except ValueError:
        return False
    return True


def _is_loopback(host: str) -> bool:
    normalized = host.casefold().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _contains_plaintext_credentials(request: MCPServerOnboardingRequest) -> bool:
    manifest = request.manifest
    command = manifest.command
    if command is not None:
        arguments = command.arguments
        for index, argument in enumerate(arguments):
            if _PLAINTEXT_AUTH_RE.search(argument):
                return True
            if _CREDENTIAL_OPTION_RE.match(argument) and (
                "=" in argument or index + 1 < len(arguments)
            ):
                return True
            if "=" in argument:
                name, _, value = argument.partition("=")
                if value and _CREDENTIAL_NAME_RE.fullmatch(name.lstrip("-")):
                    return True

    for uri in (manifest.source.source_uri, manifest.transport.endpoint_uri):
        if uri is None:
            continue
        parsed = urlsplit(uri)
        if parsed.username is not None or parsed.password is not None:
            return True
        if any(_CREDENTIAL_NAME_RE.search(name) for name, _ in parse_qsl(parsed.query)):
            return True
    return False


class MCPServerOnboardingGuard:
    """Authorize MCP server installation and connection before exposure.

    Manifests, consent, sandbox evidence, and optional external policy decisions
    are all bound to the exact effective capability digest. External verifiers
    are application trust-boundary hooks and failures are closed.
    """

    def __init__(
        self,
        policy: MCPServerOnboardingPolicy,
        *,
        consent_verifier: MCPServerConsentVerifier | None = None,
        attestation_verifier: MCPServerSandboxAttestationVerifier | None = None,
        policy_hook: MCPServerPolicyDecisionHook | None = None,
        audit_sink: MCPServerOnboardingAuditSink | None = None,
    ) -> None:
        self._policy = policy.model_copy(deep=True)
        self._servers = {item.server_id: item for item in self._policy.approved_servers}
        self._consent_verifier = consent_verifier
        self._attestation_verifier = attestation_verifier
        self._policy_hook = policy_hook
        self._audit_sink = audit_sink
        self._used_consent_refs: set[str] = set()
        self._lock = threading.Lock()

    @property
    def policy(self) -> MCPServerOnboardingPolicy:
        """Return a defensive copy of the onboarding policy."""
        return self._policy.model_copy(deep=True)

    def prepare_consent(
        self,
        request: MCPServerOnboardingRequest,
        *,
        previous_permit: MCPServerInstallationPermit | None = None,
        now: datetime | None = None,
    ) -> MCPServerOnboardingResult:
        """Validate a request and create its exact, untruncated consent prompt."""
        checked_at = now or utcnow()
        findings = self._manifest_findings(request)
        if findings:
            return self._result(request, GuardAction.BLOCK, findings, checked_at)

        notice_findings: list[MCPServerOnboardingFinding] = []
        if previous_permit is not None:
            notice_findings.extend(self._change_findings(request, previous_permit))
        notice_findings.append(
            self._finding(
                request,
                MCPServerOnboardingCode.CONSENT_REQUIRED,
                Severity.HIGH,
                "Explicit consent is required for the exact command and effective capabilities",
                "consent",
            )
        )
        manifest = request.manifest
        prompt = MCPServerConsentPrompt.create(
            consent_id=manifest.manifest_digest,
            server_id=manifest.server_id,
            operation=request.operation,
            publisher=manifest.publisher,
            source=manifest.source,
            request_digest=request.request_digest,
            manifest_digest=manifest.manifest_digest,
            capability_digest=manifest.effective_capabilities.capability_digest,
            command_argv=None if manifest.command is None else manifest.command.argv,
            command_display=None if manifest.command is None else manifest.command.display,
            command_truncated=False,
            capabilities=manifest.effective_capabilities,
            issued_at=checked_at,
            expires_at=checked_at + timedelta(seconds=self._policy.consent_ttl_seconds),
        )
        return self._result(
            request,
            GuardAction.REQUIRE_APPROVAL,
            notice_findings,
            checked_at,
            consent_prompt=prompt,
        )

    def require_consent_prompt(
        self,
        request: MCPServerOnboardingRequest,
        *,
        previous_permit: MCPServerInstallationPermit | None = None,
        now: datetime | None = None,
    ) -> MCPServerConsentPrompt:
        """Return a safe consent prompt or raise before installation UI exposure."""
        result = self.prepare_consent(request, previous_permit=previous_permit, now=now)
        if not result.requires_consent or result.consent_prompt is None:
            raise MCPServerOnboardingError(result)
        return result.consent_prompt

    def authorize(
        self,
        request: MCPServerOnboardingRequest,
        prompt: MCPServerConsentPrompt,
        grant: MCPServerConsentGrant,
        *,
        attestation: MCPServerSandboxAttestation | None = None,
        now: datetime | None = None,
    ) -> MCPServerOnboardingResult:
        """Issue a short-lived permit after consent and deployment verification."""
        checked_at = now or utcnow()
        findings = self._manifest_findings(request)
        findings.extend(self._consent_findings(request, prompt, grant, checked_at))
        findings.extend(self._attestation_findings(request, attestation, checked_at))
        decision, policy_findings = self._external_policy(request, attestation, checked_at)
        findings.extend(policy_findings)
        if findings:
            return self._result(request, GuardAction.BLOCK, findings, checked_at)

        with self._lock:
            if grant.grant_ref in self._used_consent_refs:
                replay = self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_INVALID,
                    Severity.CRITICAL,
                    "Consent grant has already been consumed",
                    "consent",
                )
                return self._result(request, GuardAction.BLOCK, [replay], checked_at)
            self._used_consent_refs.add(grant.grant_ref)

        manifest = request.manifest
        capability_digest = manifest.effective_capabilities.capability_digest
        permit_id = onboarding_reference(
            f"{request.request_digest}:{grant.grant_ref}:{checked_at.isoformat()}"
        ).removeprefix("sha256:")
        permit = MCPServerInstallationPermit.create(
            permit_id=permit_id,
            server_id=manifest.server_id,
            manifest_digest=manifest.manifest_digest,
            capabilities=manifest.effective_capabilities,
            capability_digest=capability_digest,
            consent_grant_ref=grant.grant_ref,
            sandbox_evidence_ref=None if attestation is None else attestation.evidence_ref,
            policy_decision_ref=None if decision is None else decision.decision_ref,
            issued_at=checked_at,
            expires_at=checked_at + timedelta(seconds=self._policy.permit_ttl_seconds),
        )
        allowed = self._finding(
            request,
            MCPServerOnboardingCode.ALLOWED,
            Severity.INFO,
            "Exact MCP server manifest and effective capabilities are authorized",
            "manifest",
        )
        return self._result(
            request,
            GuardAction.ALLOW,
            [allowed],
            checked_at,
            permit=permit,
        )

    def require_authorization(
        self,
        request: MCPServerOnboardingRequest,
        prompt: MCPServerConsentPrompt,
        grant: MCPServerConsentGrant,
        *,
        attestation: MCPServerSandboxAttestation | None = None,
        now: datetime | None = None,
    ) -> MCPServerInstallationPermit:
        """Return an installation permit or raise before exposing host capabilities."""
        result = self.authorize(
            request,
            prompt,
            grant,
            attestation=attestation,
            now=now,
        )
        if not result.is_allowed or result.permit is None:
            raise MCPServerOnboardingError(result)
        return result.permit

    def verify_connection(
        self,
        request: MCPServerOnboardingRequest,
        permit: MCPServerInstallationPermit,
        *,
        attestation: MCPServerSandboxAttestation | None = None,
        now: datetime | None = None,
    ) -> MCPServerOnboardingResult:
        """Revalidate the manifest, permit, sandbox, and policy before connection."""
        checked_at = now or utcnow()
        findings = self._manifest_findings(request)
        findings.extend(self._permit_findings(request, permit, attestation, checked_at))
        findings.extend(self._attestation_findings(request, attestation, checked_at))
        _, policy_findings = self._external_policy(request, attestation, checked_at)
        findings.extend(policy_findings)
        if findings:
            return self._result(request, GuardAction.BLOCK, findings, checked_at)
        allowed = self._finding(
            request,
            MCPServerOnboardingCode.ALLOWED,
            Severity.INFO,
            "MCP server connection remains bound to the authorized capabilities",
            "permit",
        )
        return self._result(
            request,
            GuardAction.ALLOW,
            [allowed],
            checked_at,
            permit=permit.model_copy(deep=True),
        )

    def require_connection(
        self,
        request: MCPServerOnboardingRequest,
        permit: MCPServerInstallationPermit,
        *,
        attestation: MCPServerSandboxAttestation | None = None,
        now: datetime | None = None,
    ) -> MCPServerInstallationPermit:
        """Return a verified permit or raise before tool or data exposure."""
        result = self.verify_connection(request, permit, attestation=attestation, now=now)
        if not result.is_allowed or result.permit is None:
            raise MCPServerOnboardingError(result)
        return result.permit

    def _manifest_findings(
        self,
        request: MCPServerOnboardingRequest,
    ) -> list[MCPServerOnboardingFinding]:
        findings: list[MCPServerOnboardingFinding] = []
        manifest = request.manifest
        approved = self._servers.get(manifest.server_id)
        if request.origin not in self._policy.trusted_origins:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.ORIGIN_DENIED,
                    Severity.CRITICAL,
                    "Installation and connection requests cannot originate from untrusted content",
                    "origin",
                )
            )
        if approved is None:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SERVER_NOT_APPROVED,
                    Severity.CRITICAL,
                    "MCP server is not present in the closed onboarding inventory",
                    "manifest.server_id",
                )
            )
            if self._is_typosquatting(manifest.source.package_name):
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.TYPOSQUATTING_SUSPECTED,
                        Severity.CRITICAL,
                        "Package identity is suspiciously similar to an approved server package",
                        "manifest.source.package_name",
                    )
                )
            return findings

        findings.extend(self._identity_findings(request, approved))
        findings.extend(self._command_and_transport_findings(request, approved))
        findings.extend(self._capability_findings(request, approved))
        return findings

    def _identity_findings(
        self,
        request: MCPServerOnboardingRequest,
        approved: MCPApprovedServerPolicy,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        findings: list[MCPServerOnboardingFinding] = []
        publisher = manifest.publisher
        if publisher.publisher_id != approved.publisher_id:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.PUBLISHER_MISMATCH,
                    Severity.CRITICAL,
                    "Publisher identity does not match the approved server record",
                    "manifest.publisher.publisher_id",
                )
            )
        if publisher.publisher_id not in self._policy.trusted_publisher_ids or (
            self._policy.require_publisher_verification
            and publisher.verification_ref not in approved.publisher_verification_refs
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.PUBLISHER_UNVERIFIED,
                    Severity.CRITICAL,
                    "Publisher identity lacks required trusted verification evidence",
                    "manifest.publisher.verification_ref",
                )
            )

        source = manifest.source
        if source.package_name != approved.package_name:
            code = (
                MCPServerOnboardingCode.TYPOSQUATTING_SUSPECTED
                if self._is_typosquatting(source.package_name)
                else MCPServerOnboardingCode.SOURCE_NOT_APPROVED
            )
            findings.append(
                self._finding(
                    request,
                    code,
                    Severity.CRITICAL,
                    "Package identity does not match the approved canonical package",
                    "manifest.source.package_name",
                )
            )
        parsed_source = urlsplit(source.source_uri)
        if parsed_source.scheme.casefold() not in {"https", "file"}:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SOURCE_INSECURE,
                    Severity.CRITICAL,
                    "Package source must use an approved integrity-protected scheme",
                    "manifest.source.source_uri",
                )
            )
        if not any(
            _uri_within_prefix(source.source_uri, prefix) for prefix in approved.source_uri_prefixes
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SOURCE_NOT_APPROVED,
                    Severity.CRITICAL,
                    "Package source is outside approved source locations",
                    "manifest.source.source_uri",
                )
            )
        if self._policy.require_source_digest and source.artifact_digest is None:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SOURCE_DIGEST_MISSING,
                    Severity.CRITICAL,
                    "Package source requires an expected artifact digest",
                    "manifest.source.artifact_digest",
                )
            )
        elif (
            source.artifact_digest is not None
            and source.artifact_digest not in approved.allowed_artifact_digests
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SOURCE_DIGEST_MISMATCH,
                    Severity.CRITICAL,
                    "Artifact digest does not match an application-approved release",
                    "manifest.source.artifact_digest",
                )
            )
        if (
            source.version not in approved.allowed_versions
            or source.revision not in approved.allowed_revisions
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SOURCE_VERSION_DENIED,
                    Severity.CRITICAL,
                    "Package version or revision is outside approved releases",
                    "manifest.source.version",
                    "manifest.source.revision",
                )
            )
        return findings

    def _command_and_transport_findings(
        self,
        request: MCPServerOnboardingRequest,
        approved: MCPApprovedServerPolicy,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        findings: list[MCPServerOnboardingFinding] = []
        command = manifest.command
        if manifest.transport.kind == MCPServerTransportKind.STDIO and command is None:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.COMMAND_REQUIRED,
                    Severity.CRITICAL,
                    "The stdio transport requires an exact structured local command",
                    "manifest.command",
                )
            )
        if command is not None:
            if command.executable not in approved.allowed_executables:
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.COMMAND_NOT_ALLOWED,
                        Severity.CRITICAL,
                        "Local executable is outside the approved server command inventory",
                        "manifest.command.executable",
                    )
                )
            if (
                len(command.arguments) > self._policy.max_command_arguments
                or sum(len(item) for item in command.argv) > self._policy.max_command_chars
            ):
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.COMMAND_LIMIT_EXCEEDED,
                        Severity.HIGH,
                        "Local command exceeds configured onboarding limits",
                        "manifest.command.arguments",
                    )
                )
        if _contains_plaintext_credentials(request):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.PLAINTEXT_CREDENTIAL,
                    Severity.CRITICAL,
                    "Credentials must use declared brokered references, not command or "
                    "endpoint text",
                    "manifest.command",
                    "manifest.source.source_uri",
                    "manifest.transport.endpoint_uri",
                )
            )

        transport = manifest.transport
        if transport.kind not in approved.allowed_transports:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.TRANSPORT_DENIED,
                    Severity.CRITICAL,
                    "MCP transport is outside the approved server policy",
                    "manifest.transport.kind",
                )
            )
        if transport.endpoint_uri is not None:
            parsed_endpoint = urlsplit(transport.endpoint_uri)
            if parsed_endpoint.scheme.casefold() != "https" or parsed_endpoint.hostname is None:
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.TRANSPORT_DENIED,
                        Severity.CRITICAL,
                        "Remote MCP transports require a valid TLS endpoint",
                        "manifest.transport.endpoint_uri",
                    )
                )
            elif (
                parsed_endpoint.hostname.casefold().rstrip(".")
                not in approved.allowed_endpoint_hosts
            ):
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.TRANSPORT_DENIED,
                        Severity.CRITICAL,
                        "Remote MCP endpoint host is outside the approved inventory",
                        "manifest.transport.endpoint_uri",
                    )
                )
        if (
            transport.bind_host is not None
            and not self._policy.allow_non_loopback_bind
            and not _is_loopback(transport.bind_host)
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.UNSAFE_BIND_ADDRESS,
                    Severity.CRITICAL,
                    "Local MCP listeners must bind to an exact loopback address",
                    "manifest.transport.bind_host",
                )
            )
        return findings

    def _capability_findings(
        self,
        request: MCPServerOnboardingRequest,
        approved: MCPApprovedServerPolicy,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        findings: list[MCPServerOnboardingFinding] = []
        if not manifest.scopes.issubset(approved.allowed_scopes):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SCOPE_DENIED,
                    Severity.CRITICAL,
                    "Requested OAuth or application scopes exceed approved bounds",
                    "manifest.scopes",
                )
            )
        if any(
            not any(_path_within_prefix(path, prefix) for prefix in approved.readable_path_prefixes)
            for path in manifest.filesystem.read_paths
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.FILESYSTEM_ACCESS_DENIED,
                    Severity.CRITICAL,
                    "Requested filesystem read access exceeds approved prefixes",
                    "manifest.filesystem.read_paths",
                )
            )
        if any(
            not any(_path_within_prefix(path, prefix) for prefix in approved.writable_path_prefixes)
            for path in manifest.filesystem.write_paths
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.FILESYSTEM_ACCESS_DENIED,
                    Severity.CRITICAL,
                    "Requested filesystem write access exceeds approved prefixes",
                    "manifest.filesystem.write_paths",
                )
            )
        if any(
            endpoint.host not in approved.allowed_outbound_hosts
            for endpoint in manifest.outbound_network
        ):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.NETWORK_ACCESS_DENIED,
                    Severity.CRITICAL,
                    "Requested outbound network access exceeds approved hosts",
                    "manifest.outbound_network",
                )
            )
        if any(secret.name not in approved.allowed_secret_names for secret in manifest.secrets):
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SECRET_ACCESS_DENIED,
                    Severity.CRITICAL,
                    "Requested brokered secret exceeds the approved secret inventory",
                    "manifest.secrets",
                )
            )

        sandbox = manifest.sandbox
        if manifest.command is not None and sandbox is None:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SANDBOX_PROFILE_DENIED,
                    Severity.CRITICAL,
                    "Locally launched MCP servers must declare an approved sandbox profile",
                    "manifest.sandbox",
                )
            )
        elif sandbox is not None:
            if sandbox.profile_id not in approved.allowed_sandbox_profiles:
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.SANDBOX_PROFILE_DENIED,
                        Severity.CRITICAL,
                        "Declared sandbox profile is outside approved policy",
                        "manifest.sandbox.profile_id",
                    )
                )
            missing = approved.required_sandbox_controls - sandbox.required_controls
            if missing:
                findings.append(
                    self._finding(
                        request,
                        MCPServerOnboardingCode.SANDBOX_CONTROL_MISSING,
                        Severity.CRITICAL,
                        "Declared sandbox omits required isolation controls",
                        "manifest.sandbox.required_controls",
                    )
                )
        return findings

    def _consent_findings(
        self,
        request: MCPServerOnboardingRequest,
        prompt: MCPServerConsentPrompt,
        grant: MCPServerConsentGrant,
        now: datetime,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        capabilities = manifest.effective_capabilities
        expected_argv = None if manifest.command is None else manifest.command.argv
        expected_display = None if manifest.command is None else manifest.command.display
        prompt_matches = (
            prompt.has_valid_integrity
            and prompt.server_id == manifest.server_id
            and prompt.operation == request.operation
            and prompt.publisher == manifest.publisher
            and prompt.source == manifest.source
            and prompt.request_digest == request.request_digest
            and prompt.manifest_digest == manifest.manifest_digest
            and prompt.capability_digest == capabilities.capability_digest
            and prompt.capabilities == capabilities
            and prompt.command_argv == expected_argv
            and prompt.command_display == expected_display
            and prompt.command_truncated is False
        )
        if not prompt_matches:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_INVALID,
                    Severity.CRITICAL,
                    "Consent prompt is not bound to the complete current request",
                    "consent",
                )
            ]
        if now > prompt.expires_at or now < prompt.issued_at:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_EXPIRED,
                    Severity.CRITICAL,
                    "Consent prompt is expired or not yet valid",
                    "consent",
                )
            ]
        grant_matches = (
            grant.consent_digest == prompt.consent_digest
            and grant.manifest_digest == manifest.manifest_digest
            and grant.capability_digest == capabilities.capability_digest
            and grant.approved_at >= prompt.issued_at
            and grant.approved_at <= prompt.expires_at
            and now >= grant.approved_at
            and now <= grant.expires_at
        )
        if not grant_matches:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_INVALID,
                    Severity.CRITICAL,
                    "Consent grant is not valid for the current exact prompt",
                    "consent",
                )
            ]
        if self._consent_verifier is None:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_UNVERIFIED,
                    Severity.CRITICAL,
                    "No trusted consent verifier is configured",
                    "consent",
                )
            ]
        try:
            verified = self._consent_verifier.verify_consent(grant, prompt, request)
        except Exception:
            verified = False
        if not verified:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.CONSENT_UNVERIFIED,
                    Severity.CRITICAL,
                    "Consent grant was not authenticated by trusted application state",
                    "consent",
                )
            ]
        return []

    def _attestation_findings(
        self,
        request: MCPServerOnboardingRequest,
        attestation: MCPServerSandboxAttestation | None,
        now: datetime,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        required = self._policy.require_local_sandbox_attestation and manifest.command is not None
        if attestation is None:
            if not required:
                return []
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.ATTESTATION_REQUIRED,
                    Severity.CRITICAL,
                    "External sandbox attestation is required for a local MCP server",
                    "attestation",
                )
            ]
        sandbox = manifest.sandbox
        if (
            sandbox is None
            or attestation.manifest_digest != manifest.manifest_digest
            or attestation.capability_digest != manifest.effective_capabilities.capability_digest
            or attestation.sandbox_profile_id != sandbox.profile_id
            or not sandbox.required_controls.issubset(attestation.enforced_controls)
        ):
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.ATTESTATION_MISMATCH,
                    Severity.CRITICAL,
                    "Sandbox evidence does not cover the current manifest and controls",
                    "attestation",
                )
            ]
        if now < attestation.issued_at or now > attestation.expires_at:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.ATTESTATION_EXPIRED,
                    Severity.CRITICAL,
                    "Sandbox evidence is expired or not yet valid",
                    "attestation",
                )
            ]
        if self._attestation_verifier is None:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.ATTESTATION_INVALID,
                    Severity.CRITICAL,
                    "No trusted sandbox attestation verifier is configured",
                    "attestation",
                )
            ]
        try:
            verified = self._attestation_verifier.verify_attestation(attestation, request)
        except Exception:
            verified = False
        if not verified:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.ATTESTATION_INVALID,
                    Severity.CRITICAL,
                    "Sandbox attestation was not authenticated by trusted application state",
                    "attestation",
                )
            ]
        return []

    def _external_policy(
        self,
        request: MCPServerOnboardingRequest,
        attestation: MCPServerSandboxAttestation | None,
        now: datetime,
    ) -> tuple[
        MCPServerExternalPolicyDecision | None,
        list[MCPServerOnboardingFinding],
    ]:
        if self._policy_hook is None:
            return None, []
        try:
            decision = self._policy_hook.evaluate(request, attestation)
        except Exception:
            return None, [
                self._finding(
                    request,
                    MCPServerOnboardingCode.POLICY_HOOK_UNAVAILABLE,
                    Severity.CRITICAL,
                    "External deployment policy could not produce a decision",
                    "policy_decision",
                )
            ]
        manifest = request.manifest
        if (
            decision.request_digest != request.request_digest
            or decision.manifest_digest != manifest.manifest_digest
            or decision.capability_digest != manifest.effective_capabilities.capability_digest
            or decision.expires_at < now
        ):
            return decision, [
                self._finding(
                    request,
                    MCPServerOnboardingCode.POLICY_DECISION_INVALID,
                    Severity.CRITICAL,
                    "External policy decision is stale or bound to a different request",
                    "policy_decision",
                )
            ]
        if not decision.allowed:
            return decision, [
                self._finding(
                    request,
                    MCPServerOnboardingCode.POLICY_HOOK_REJECTED,
                    Severity.CRITICAL,
                    "External deployment policy denied the MCP server request",
                    "policy_decision",
                )
            ]
        return decision, []

    def _permit_findings(
        self,
        request: MCPServerOnboardingRequest,
        permit: MCPServerInstallationPermit,
        attestation: MCPServerSandboxAttestation | None,
        now: datetime,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        capabilities = manifest.effective_capabilities
        if (
            not permit.has_valid_integrity
            or permit.server_id != manifest.server_id
            or permit.manifest_digest != manifest.manifest_digest
            or permit.capability_digest != capabilities.capability_digest
            or permit.capabilities != capabilities
        ):
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.PERMIT_INVALID,
                    Severity.CRITICAL,
                    "Installation permit does not authorize the current exact manifest",
                    "permit",
                )
            ]
        if now < permit.issued_at or now > permit.expires_at:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.PERMIT_EXPIRED,
                    Severity.CRITICAL,
                    "Installation permit is expired or not yet valid",
                    "permit",
                )
            ]
        evidence_ref = None if attestation is None else attestation.evidence_ref
        if permit.sandbox_evidence_ref != evidence_ref:
            return [
                self._finding(
                    request,
                    MCPServerOnboardingCode.PERMIT_INVALID,
                    Severity.CRITICAL,
                    "Installation permit is bound to different sandbox evidence",
                    "permit.sandbox_evidence_ref",
                )
            ]
        return []

    def _change_findings(
        self,
        request: MCPServerOnboardingRequest,
        previous: MCPServerInstallationPermit,
    ) -> list[MCPServerOnboardingFinding]:
        manifest = request.manifest
        findings: list[MCPServerOnboardingFinding] = []
        if previous.manifest_digest != manifest.manifest_digest:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.MANIFEST_CHANGED,
                    Severity.HIGH,
                    "Server manifest changed and requires a new exact consent decision",
                    "manifest",
                )
            )
        current = manifest.effective_capabilities
        prior = previous.capabilities
        scope_grew = not current.scopes.issubset(prior.scopes)
        filesystem_grew = not current.filesystem.read_paths.issubset(
            prior.filesystem.read_paths
        ) or not current.filesystem.write_paths.issubset(prior.filesystem.write_paths)
        network_grew = not set(current.outbound_network).issubset(prior.outbound_network)
        secrets_grew = not current.secret_names.issubset(prior.secret_names)
        sandbox_changed = current.sandbox != prior.sandbox or current.transport != prior.transport
        if scope_grew or filesystem_grew or network_grew or secrets_grew or sandbox_changed:
            findings.append(
                self._finding(
                    request,
                    MCPServerOnboardingCode.SCOPE_GROWTH,
                    Severity.CRITICAL,
                    "Effective capabilities grew or changed and require fresh consent",
                    "manifest.capabilities",
                )
            )
        return findings

    def _is_typosquatting(self, package_name: str) -> bool:
        candidate = _identity_skeleton(package_name)
        for item in self._policy.approved_servers:
            canonical = _identity_skeleton(item.package_name)
            if candidate != canonical and _edit_distance(candidate, canonical) <= 2:
                return True
        return False

    @staticmethod
    def _finding(
        request: MCPServerOnboardingRequest,
        code: MCPServerOnboardingCode,
        severity: Severity,
        message: str,
        *field_paths: str,
    ) -> MCPServerOnboardingFinding:
        return MCPServerOnboardingFinding(
            code=code,
            severity=severity,
            message=message,
            server_ref=onboarding_reference(request.manifest.server_id),
            field_paths=tuple(field_paths),
        )

    def _result(
        self,
        request: MCPServerOnboardingRequest,
        action: GuardAction,
        findings: list[MCPServerOnboardingFinding],
        occurred_at: datetime,
        *,
        consent_prompt: MCPServerConsentPrompt | None = None,
        permit: MCPServerInstallationPermit | None = None,
    ) -> MCPServerOnboardingResult:
        event = MCPServerOnboardingAuditEvent(
            occurred_at=occurred_at,
            operation=request.operation,
            action=action,
            request_ref=onboarding_reference(request.request_id),
            server_ref=onboarding_reference(request.manifest.server_id),
            initiator_ref=onboarding_reference(request.initiated_by),
            tenant_ref=onboarding_reference(request.tenant_id),
            manifest_ref=onboarding_reference(request.manifest.manifest_digest),
            finding_codes=tuple(finding.code for finding in findings),
        )
        if self._audit_sink is not None:
            self._audit_sink.emit(event)
        return MCPServerOnboardingResult(
            action=action,
            findings=tuple(findings),
            consent_prompt=consent_prompt,
            permit=permit,
            audit_event=event,
        )
