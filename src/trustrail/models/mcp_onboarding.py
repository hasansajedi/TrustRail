"""Typed contracts for secure MCP server onboarding and consent."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trustrail.models.enums import GuardAction, Severity

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
_PACKAGE_PATTERN = r"^@?[A-Za-z0-9][A-Za-z0-9._@/-]{0,255}$"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_REFERENCE_PATTERN = r"^sha256:[0-9a-f]{64}$"
_ENV_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]{0,127}$"


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _digest(value: Any) -> str:
    canonical = json.dumps(
        _canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def onboarding_reference(value: str | bytes) -> str:
    """Return a content-free SHA-256 reference for audit and evidence fields."""
    raw = value.encode() if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _normalize_package_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[-_.]+", "-", normalized)


def _validate_absolute_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value:
        raise ValueError("filesystem paths must be absolute POSIX paths")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError("filesystem paths must be normalized absolute POSIX paths")
    return str(path)


class MCPServerOnboardingOperation(StrEnum):
    """Lifecycle operation requested before a server is exposed."""

    INSTALL = "install"
    CONNECT = "connect"


class MCPServerRequestOrigin(StrEnum):
    """Origin of an installation or connection request."""

    USER_ACTION = "user_action"
    ADMIN_POLICY = "admin_policy"
    LOCAL_CONFIG = "local_config"
    WEB_CONTENT = "web_content"
    MODEL_OUTPUT = "model_output"
    TOOL_OUTPUT = "tool_output"
    RETRIEVED_CONTENT = "retrieved_content"
    REMOTE_REQUEST = "remote_request"


class MCPServerTransportKind(StrEnum):
    """Supported local and remote MCP transport declarations."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"
    SSE = "sse"


class MCPServerNetworkProtocol(StrEnum):
    """Protocols a server may use for declared outbound access."""

    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"


class MCPSecretDelivery(StrEnum):
    """Brokered mechanisms for delivering a referenced secret."""

    ENVIRONMENT = "environment"
    FILE_DESCRIPTOR = "file_descriptor"
    FILE_MOUNT = "file_mount"
    OAUTH_BROKER = "oauth_broker"


class MCPServerSandboxControl(StrEnum):
    """Isolation controls asserted by an external sandbox provider."""

    PROCESS_ISOLATION = "process_isolation"
    FILESYSTEM_POLICY = "filesystem_policy"
    NETWORK_POLICY = "network_policy"
    SECRET_BROKER = "secret_broker"  # noqa: S105 - capability name, not a credential
    RESOURCE_LIMITS = "resource_limits"
    NO_NEW_PRIVILEGES = "no_new_privileges"
    EPHEMERAL_WORKSPACE = "ephemeral_workspace"
    CLEANUP_ON_EXIT = "cleanup_on_exit"


class MCPServerOnboardingCode(StrEnum):
    """Stable machine-readable onboarding outcomes."""

    ALLOWED = "allowed"
    CONSENT_REQUIRED = "consent_required"
    ORIGIN_DENIED = "origin_denied"
    SERVER_NOT_APPROVED = "server_not_approved"
    PUBLISHER_UNVERIFIED = "publisher_unverified"
    PUBLISHER_MISMATCH = "publisher_mismatch"
    SOURCE_NOT_APPROVED = "source_not_approved"
    SOURCE_INSECURE = "source_insecure"
    SOURCE_DIGEST_MISSING = "source_digest_missing"
    SOURCE_DIGEST_MISMATCH = "source_digest_mismatch"
    SOURCE_VERSION_DENIED = "source_version_denied"
    TYPOSQUATTING_SUSPECTED = "typosquatting_suspected"
    COMMAND_REQUIRED = "command_required"
    COMMAND_NOT_ALLOWED = "command_not_allowed"
    COMMAND_LIMIT_EXCEEDED = "command_limit_exceeded"
    PLAINTEXT_CREDENTIAL = "plaintext_credential"
    TRANSPORT_DENIED = "transport_denied"
    UNSAFE_BIND_ADDRESS = "unsafe_bind_address"
    SCOPE_DENIED = "scope_denied"
    SCOPE_GROWTH = "scope_growth"
    FILESYSTEM_ACCESS_DENIED = "filesystem_access_denied"
    NETWORK_ACCESS_DENIED = "network_access_denied"
    SECRET_ACCESS_DENIED = "secret_access_denied"  # noqa: S105 - outcome code
    SANDBOX_PROFILE_DENIED = "sandbox_profile_denied"
    SANDBOX_CONTROL_MISSING = "sandbox_control_missing"
    ATTESTATION_REQUIRED = "attestation_required"
    ATTESTATION_INVALID = "attestation_invalid"
    ATTESTATION_EXPIRED = "attestation_expired"
    ATTESTATION_MISMATCH = "attestation_mismatch"
    POLICY_HOOK_REJECTED = "policy_hook_rejected"
    POLICY_HOOK_UNAVAILABLE = "policy_hook_unavailable"
    POLICY_DECISION_INVALID = "policy_decision_invalid"
    CONSENT_INVALID = "consent_invalid"
    CONSENT_EXPIRED = "consent_expired"
    CONSENT_UNVERIFIED = "consent_unverified"
    MANIFEST_CHANGED = "manifest_changed"
    PERMIT_INVALID = "permit_invalid"
    PERMIT_EXPIRED = "permit_expired"


class MCPServerPublisher(BaseModel):
    """Publisher identity and out-of-band verification evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publisher_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    display_name: str = Field(min_length=1, max_length=256)
    verification_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)

    @field_validator("display_name")
    @classmethod
    def reject_control_text(cls, value: str) -> str:
        if not value.strip() or any(unicodedata.category(char) == "Cc" for char in value):
            raise ValueError("publisher display name must be visible non-blank text")
        return value


class MCPServerSource(BaseModel):
    """Pinned package and source location proposed for installation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    package_name: str = Field(pattern=_PACKAGE_PATTERN)
    source_uri: str = Field(min_length=1, max_length=2_048)
    version: str = Field(pattern=_IDENTIFIER_PATTERN)
    revision: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_digest: str | None = Field(default=None, pattern=_DIGEST_PATTERN)

    @field_validator("package_name")
    @classmethod
    def normalize_package_name(cls, value: str) -> str:
        return _normalize_package_name(value)


class MCPServerCommand(BaseModel):
    """Structured local command retained exactly for consent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    executable: str = Field(min_length=1, max_length=2_048)
    arguments: tuple[str, ...] = Field(default_factory=tuple, max_length=1_000)
    working_directory: str | None = Field(default=None, max_length=2_048)

    @field_validator("executable", "arguments")
    @classmethod
    def reject_command_controls(cls, value: str | tuple[str, ...]) -> str | tuple[str, ...]:
        values = (value,) if isinstance(value, str) else value
        if any(
            not item or "\x00" in item or any(unicodedata.category(char) == "Cc" for char in item)
            for item in values
        ):
            raise ValueError("command values must be non-empty and contain no control characters")
        return value

    @field_validator("working_directory")
    @classmethod
    def validate_working_directory(cls, value: str | None) -> str | None:
        return None if value is None else _validate_absolute_path(value)

    @property
    def argv(self) -> tuple[str, ...]:
        """Return the exact unmodified command vector."""
        return (self.executable, *self.arguments)

    @property
    def display(self) -> str:
        """Return the complete shell-escaped display string without truncation."""
        return shlex.join(self.argv)


class MCPServerTransport(BaseModel):
    """Transport endpoint or local bind declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: MCPServerTransportKind
    endpoint_uri: str | None = Field(default=None, max_length=2_048)
    bind_host: str | None = Field(default=None, max_length=253)
    bind_port: int | None = Field(default=None, ge=1, le=65_535)

    @field_validator("bind_host")
    @classmethod
    def normalize_bind_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.casefold().rstrip(".")
        if not normalized or any(char.isspace() for char in normalized):
            raise ValueError("bind_host must be an exact hostname or address")
        return normalized

    @model_validator(mode="after")
    def validate_shape(self) -> MCPServerTransport:
        if self.kind == MCPServerTransportKind.STDIO:
            if (
                self.endpoint_uri is not None
                or self.bind_host is not None
                or self.bind_port is not None
            ):
                raise ValueError("stdio transport cannot declare an endpoint or bind address")
        elif (self.endpoint_uri is None) == (self.bind_host is None):
            raise ValueError("network transport requires exactly one endpoint or local bind")
        elif (self.bind_host is None) != (self.bind_port is None):
            raise ValueError("local bind host and port must be declared together")
        return self


class MCPServerFilesystemAccess(BaseModel):
    """Host filesystem paths requested by the MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    read_paths: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    write_paths: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)

    @field_validator("read_paths", "write_paths")
    @classmethod
    def validate_paths(cls, values: frozenset[str]) -> frozenset[str]:
        return frozenset(_validate_absolute_path(value) for value in values)


class MCPServerNetworkEndpoint(BaseModel):
    """One explicit outbound network capability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65_535)
    protocol: MCPServerNetworkProtocol = MCPServerNetworkProtocol.HTTPS

    @field_validator("host")
    @classmethod
    def normalize_host(cls, value: str) -> str:
        normalized = value.casefold().rstrip(".")
        if (
            not normalized
            or any(char.isspace() for char in normalized)
            or any(char in normalized for char in "/*@[]")
        ):
            raise ValueError("network host must be an exact name or address")
        return normalized


class MCPServerSecretRequirement(BaseModel):
    """Reference to a secret delivered by a trusted broker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=_ENV_NAME_PATTERN)
    secret_ref: str = Field(pattern=_REFERENCE_PATTERN)
    delivery: MCPSecretDelivery
    target: str | None = Field(default=None, max_length=2_048)

    @model_validator(mode="after")
    def validate_target(self) -> MCPServerSecretRequirement:
        if self.delivery == MCPSecretDelivery.ENVIRONMENT:
            if self.target is None or re.fullmatch(_ENV_NAME_PATTERN, self.target) is None:
                raise ValueError("environment delivery requires an environment variable target")
        elif self.delivery == MCPSecretDelivery.FILE_MOUNT:
            if self.target is None:
                raise ValueError("file-mount delivery requires an absolute target")
            _validate_absolute_path(self.target)
        elif self.target is not None:
            raise ValueError("selected secret delivery mechanism does not accept a target")
        return self


class MCPServerSandboxRequirements(BaseModel):
    """Declared external sandbox profile and minimum controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    required_controls: frozenset[MCPServerSandboxControl] = Field(min_length=1)
    max_memory_bytes: int = Field(ge=1, le=1_099_511_627_776)
    max_cpu_seconds: int = Field(ge=1, le=86_400)
    max_processes: int = Field(ge=1, le=65_536)


class MCPServerEffectiveCapabilities(BaseModel):
    """Exact effective capabilities displayed for consent and pinned in permits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scopes: frozenset[str]
    filesystem: MCPServerFilesystemAccess
    outbound_network: tuple[MCPServerNetworkEndpoint, ...]
    secret_names: frozenset[str]
    transport: MCPServerTransport
    sandbox: MCPServerSandboxRequirements | None

    @property
    def capability_digest(self) -> str:
        """Return a deterministic digest of every effective capability."""
        return _digest(self)


class MCPServerManifest(BaseModel):
    """Complete, integrity-bound MCP server installation declaration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    publisher: MCPServerPublisher
    source: MCPServerSource
    command: MCPServerCommand | None = None
    transport: MCPServerTransport
    scopes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    filesystem: MCPServerFilesystemAccess = Field(default_factory=MCPServerFilesystemAccess)
    outbound_network: tuple[MCPServerNetworkEndpoint, ...] = Field(
        default_factory=tuple,
        max_length=1_000,
    )
    secrets: tuple[MCPServerSecretRequirement, ...] = Field(
        default_factory=tuple,
        max_length=1_000,
    )
    sandbox: MCPServerSandboxRequirements | None = None

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_IDENTIFIER_PATTERN, value) is None for value in values):
            raise ValueError("scopes contain an unsupported identifier")
        return values

    @model_validator(mode="after")
    def validate_unique_capabilities(self) -> MCPServerManifest:
        endpoints = [
            (endpoint.host, endpoint.port, endpoint.protocol) for endpoint in self.outbound_network
        ]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("outbound network endpoints must be unique")
        secret_names = [secret.name for secret in self.secrets]
        if len(secret_names) != len(set(secret_names)):
            raise ValueError("secret requirements must use unique names")
        return self

    @property
    def effective_capabilities(self) -> MCPServerEffectiveCapabilities:
        """Return the exact capability set a user must approve."""
        return MCPServerEffectiveCapabilities(
            scopes=self.scopes,
            filesystem=self.filesystem,
            outbound_network=self.outbound_network,
            secret_names=frozenset(secret.name for secret in self.secrets),
            transport=self.transport,
            sandbox=self.sandbox,
        )

    @property
    def manifest_digest(self) -> str:
        """Return a deterministic digest of the complete manifest."""
        return _digest(self)


class MCPApprovedServerPolicy(BaseModel):
    """Maximum installable capabilities for one canonical server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    publisher_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    publisher_verification_refs: frozenset[str] = Field(min_length=1, max_length=100)
    package_name: str = Field(pattern=_PACKAGE_PATTERN)
    source_uri_prefixes: tuple[str, ...] = Field(min_length=1, max_length=100)
    allowed_versions: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_revisions: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_artifact_digests: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_executables: frozenset[str] = Field(default_factory=frozenset, max_length=100)
    allowed_transports: frozenset[MCPServerTransportKind] = Field(min_length=1)
    allowed_endpoint_hosts: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    allowed_scopes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    readable_path_prefixes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    writable_path_prefixes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    allowed_outbound_hosts: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    allowed_secret_names: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)
    allowed_sandbox_profiles: frozenset[str] = Field(default_factory=frozenset, max_length=100)
    required_sandbox_controls: frozenset[MCPServerSandboxControl] = Field(default_factory=frozenset)

    @field_validator("package_name")
    @classmethod
    def normalize_package_name(cls, value: str) -> str:
        return _normalize_package_name(value)

    @field_validator("publisher_verification_refs")
    @classmethod
    def validate_publisher_references(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_REFERENCE_PATTERN, value) is None for value in values):
            raise ValueError("publisher verification references must be SHA-256 references")
        return values

    @field_validator("allowed_artifact_digests")
    @classmethod
    def validate_artifact_digests(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_DIGEST_PATTERN, value) is None for value in values):
            raise ValueError("approved artifact digests must be SHA-256 digests")
        return values

    @field_validator("readable_path_prefixes", "writable_path_prefixes")
    @classmethod
    def validate_path_prefixes(cls, values: frozenset[str]) -> frozenset[str]:
        return frozenset(_validate_absolute_path(value) for value in values)

    @field_validator("allowed_endpoint_hosts", "allowed_outbound_hosts")
    @classmethod
    def normalize_hosts(cls, values: frozenset[str]) -> frozenset[str]:
        return frozenset(value.casefold().rstrip(".") for value in values)


def _trusted_origins() -> frozenset[MCPServerRequestOrigin]:
    return frozenset(
        {
            MCPServerRequestOrigin.USER_ACTION,
            MCPServerRequestOrigin.ADMIN_POLICY,
            MCPServerRequestOrigin.LOCAL_CONFIG,
        }
    )


class MCPServerOnboardingPolicy(BaseModel):
    """Closed inventory and bounds for onboarding decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approved_servers: tuple[MCPApprovedServerPolicy, ...] = Field(min_length=1, max_length=10_000)
    trusted_publisher_ids: frozenset[str] = Field(min_length=1, max_length=10_000)
    trusted_origins: frozenset[MCPServerRequestOrigin] = Field(default_factory=_trusted_origins)
    require_publisher_verification: bool = True
    require_source_digest: bool = True
    require_local_sandbox_attestation: bool = True
    allow_non_loopback_bind: bool = False
    consent_ttl_seconds: int = Field(default=600, ge=1, le=86_400)
    permit_ttl_seconds: int = Field(default=3_600, ge=1, le=604_800)
    max_command_arguments: int = Field(default=256, ge=0, le=10_000)
    max_command_chars: int = Field(default=32_768, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> MCPServerOnboardingPolicy:
        server_ids = [item.server_id for item in self.approved_servers]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("approved server IDs must be unique")
        if any(
            item.publisher_id not in self.trusted_publisher_ids for item in self.approved_servers
        ):
            raise ValueError("approved servers must reference trusted publishers")
        return self


class MCPServerOnboardingRequest(BaseModel):
    """One application-originated installation or connection request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    operation: MCPServerOnboardingOperation
    origin: MCPServerRequestOrigin
    initiated_by: str = Field(pattern=_IDENTIFIER_PATTERN)
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    manifest: MCPServerManifest

    @property
    def request_digest(self) -> str:
        """Return an exact request binding."""
        return _digest(self)


class MCPServerConsentPrompt(BaseModel):
    """Complete, untruncated command and capability presentation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    consent_id: str = Field(pattern=_DIGEST_PATTERN)
    server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    operation: MCPServerOnboardingOperation
    publisher: MCPServerPublisher
    source: MCPServerSource
    request_digest: str = Field(pattern=_DIGEST_PATTERN)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    capability_digest: str = Field(pattern=_DIGEST_PATTERN)
    command_argv: tuple[str, ...] | None
    command_display: str | None
    command_truncated: Literal[False] = False
    capabilities: MCPServerEffectiveCapabilities
    issued_at: datetime
    expires_at: datetime
    consent_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(cls, **values: Any) -> MCPServerConsentPrompt:
        """Create an integrity-bound consent presentation."""
        values = dict(values)
        values.pop("consent_digest", None)
        candidate = cls.model_construct(**values, consent_digest="0" * 64)
        return cls(**values, consent_digest=_digest(candidate.integrity_payload))

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_prompt(self) -> MCPServerConsentPrompt:
        if self.expires_at <= self.issued_at:
            raise ValueError("consent expiry must follow issuance")
        if (self.command_argv is None) != (self.command_display is None):
            raise ValueError("command vector and display must be supplied together")
        if self.command_argv is not None and shlex.join(self.command_argv) != self.command_display:
            raise ValueError("command display must contain the complete exact command")
        if self.capability_digest != self.capabilities.capability_digest:
            raise ValueError("consent capabilities do not match their digest")
        if not self.has_valid_integrity:
            raise ValueError("consent prompt integrity check failed")
        return self

    @property
    def integrity_payload(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "capability_digest": self.capability_digest,
            "command_argv": self.command_argv,
            "command_display": self.command_display,
            "command_truncated": self.command_truncated,
            "consent_id": self.consent_id,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "manifest_digest": self.manifest_digest,
            "operation": self.operation,
            "publisher": self.publisher,
            "request_digest": self.request_digest,
            "server_id": self.server_id,
            "source": self.source,
        }

    @property
    def has_valid_integrity(self) -> bool:
        try:
            return self.consent_digest == _digest(self.integrity_payload)
        except (TypeError, ValueError):
            return False


class MCPServerConsentGrant(BaseModel):
    """Authenticated user decision bound to one exact consent prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grant_ref: str = Field(pattern=_REFERENCE_PATTERN)
    consent_digest: str = Field(pattern=_DIGEST_PATTERN)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    capability_digest: str = Field(pattern=_DIGEST_PATTERN)
    approved_by: str = Field(pattern=_IDENTIFIER_PATTERN)
    approved_at: datetime
    expires_at: datetime

    @field_validator("approved_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_expiry(self) -> MCPServerConsentGrant:
        if self.expires_at <= self.approved_at:
            raise ValueError("consent grant expiry must follow approval")
        return self


class MCPServerSandboxAttestation(BaseModel):
    """External evidence that a manifest is deployed in its declared sandbox."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(pattern=_REFERENCE_PATTERN)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    capability_digest: str = Field(pattern=_DIGEST_PATTERN)
    sandbox_profile_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    enforced_controls: frozenset[MCPServerSandboxControl] = Field(min_length=1)
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_expiry(self) -> MCPServerSandboxAttestation:
        if self.expires_at <= self.issued_at:
            raise ValueError("attestation expiry must follow issuance")
        return self


class MCPServerExternalPolicyDecision(BaseModel):
    """Bound result returned by an external deployment policy engine."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_ref: str = Field(pattern=_REFERENCE_PATTERN)
    request_digest: str = Field(pattern=_DIGEST_PATTERN)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    capability_digest: str = Field(pattern=_DIGEST_PATTERN)
    allowed: bool
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def require_aware_expiry(cls, value: datetime) -> datetime:
        return _require_aware(value, "expires_at")


class MCPServerInstallationPermit(BaseModel):
    """Short-lived authorization for one exact manifest and capability set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: str = Field(pattern=_DIGEST_PATTERN)
    server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    manifest_digest: str = Field(pattern=_DIGEST_PATTERN)
    capabilities: MCPServerEffectiveCapabilities
    capability_digest: str = Field(pattern=_DIGEST_PATTERN)
    consent_grant_ref: str = Field(pattern=_REFERENCE_PATTERN)
    sandbox_evidence_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    policy_decision_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    issued_at: datetime
    expires_at: datetime
    permit_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(cls, **values: Any) -> MCPServerInstallationPermit:
        """Create an integrity-bound installation or connection permit."""
        values = dict(values)
        values.pop("permit_digest", None)
        candidate = cls.model_construct(**values, permit_digest="0" * 64)
        return cls(**values, permit_digest=_digest(candidate.integrity_payload))

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_permit(self) -> MCPServerInstallationPermit:
        if self.expires_at <= self.issued_at:
            raise ValueError("permit expiry must follow issuance")
        if self.capability_digest != self.capabilities.capability_digest:
            raise ValueError("permit capabilities do not match their digest")
        if not self.has_valid_integrity:
            raise ValueError("installation permit integrity check failed")
        return self

    @property
    def integrity_payload(self) -> dict[str, Any]:
        return {
            "capabilities": self.capabilities,
            "capability_digest": self.capability_digest,
            "consent_grant_ref": self.consent_grant_ref,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "manifest_digest": self.manifest_digest,
            "permit_id": self.permit_id,
            "policy_decision_ref": self.policy_decision_ref,
            "sandbox_evidence_ref": self.sandbox_evidence_ref,
            "server_id": self.server_id,
        }

    @property
    def has_valid_integrity(self) -> bool:
        try:
            return self.permit_digest == _digest(self.integrity_payload)
        except (TypeError, ValueError):
            return False


class MCPServerOnboardingFinding(BaseModel):
    """Content-free explanation of an onboarding decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: MCPServerOnboardingCode
    severity: Severity
    message: str = Field(min_length=1, max_length=500)
    server_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    field_paths: tuple[str, ...] = ()


class MCPServerOnboardingAuditEvent(BaseModel):
    """Content-free onboarding event for a restricted audit sink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    occurred_at: datetime
    operation: MCPServerOnboardingOperation
    action: GuardAction
    request_ref: str = Field(pattern=_REFERENCE_PATTERN)
    server_ref: str = Field(pattern=_REFERENCE_PATTERN)
    initiator_ref: str = Field(pattern=_REFERENCE_PATTERN)
    tenant_ref: str = Field(pattern=_REFERENCE_PATTERN)
    manifest_ref: str = Field(pattern=_REFERENCE_PATTERN)
    finding_codes: tuple[MCPServerOnboardingCode, ...]

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        return _require_aware(value, "occurred_at")


class MCPServerOnboardingResult(BaseModel):
    """Block, consent, or authorization result from the onboarding guard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: GuardAction
    findings: tuple[MCPServerOnboardingFinding, ...] = ()
    consent_prompt: MCPServerConsentPrompt | None = None
    permit: MCPServerInstallationPermit | None = None
    audit_event: MCPServerOnboardingAuditEvent

    @property
    def is_allowed(self) -> bool:
        return self.action == GuardAction.ALLOW and self.permit is not None

    @property
    def is_blocked(self) -> bool:
        return self.action == GuardAction.BLOCK

    @property
    def requires_consent(self) -> bool:
        return self.action == GuardAction.REQUIRE_APPROVAL and self.consent_prompt is not None
