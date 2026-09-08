"""Typed contracts for MCP server isolation and cross-origin data flows."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from trustrail.models.enums import GuardAction, Severity

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
_TOOL_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_NAMESPACE_PATTERN = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"
_LABEL_PATTERN = r"^[a-z][a-z0-9_.:-]{0,127}$"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_REFERENCE_PATTERN = r"^sha256:[0-9a-f]{64}$"


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        values = [_canonicalize(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True))
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


def mcp_credential_reference(credential: str | bytes) -> str:
    """Hash a credential for policy matching without retaining the secret."""
    value = credential.encode() if isinstance(credential, str) else credential
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


class MCPDataLabel(StrEnum):
    """Built-in data sensitivity labels; policies may also use custom labels."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    CREDENTIAL = "credential"


class MCPIsolationOperation(StrEnum):
    """Gateway operation represented by an isolation audit event."""

    DEFINITION_INSPECTION = "definition_inspection"
    INVOCATION = "invocation"


class MCPIsolationCode(StrEnum):
    """Stable machine-readable MCP isolation outcomes."""

    ALLOWED = "allowed"
    UNKNOWN_SERVER = "unknown_server"
    NAMESPACE_COLLISION = "namespace_collision"
    TOOL_SHADOWING = "tool_shadowing"
    TOOL_ROUTE_MISMATCH = "tool_route_mismatch"
    CROSS_SERVER_INSTRUCTION = "cross_server_instruction"
    CONFUSED_DEPUTY = "confused_deputy"
    CREDENTIAL_CROSSOVER = "credential_crossover"
    RESULT_INTEGRITY_INVALID = "result_integrity_invalid"
    DATA_LABEL_REQUIRED = "data_label_required"
    DATA_FLOW_DENIED = "data_flow_denied"
    REDACTION_REQUIRED = "redaction_required"
    REDACTION_FAILED = "redaction_failed"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_INVALID = "approval_invalid"
    APPROVAL_REPLAYED = "approval_replayed"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"


class MCPServerTrustDomain(BaseModel):
    """One independently authenticated MCP server and its owned tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tool_namespace: str = Field(pattern=_NAMESPACE_PATTERN)
    allowed_tools: frozenset[str] = Field(min_length=1, max_length=1_000)
    credential_ref: str = Field(pattern=_REFERENCE_PATTERN)
    allowed_tenant_ids: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_agent_ids: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_user_ids: frozenset[str] = Field(default_factory=frozenset, max_length=10_000)
    required_scopes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)

    @field_validator(
        "allowed_tools",
        "allowed_tenant_ids",
        "allowed_agent_ids",
        "allowed_user_ids",
        "required_scopes",
    )
    @classmethod
    def validate_values(cls, values: frozenset[str]) -> frozenset[str]:
        if any(not value or len(value) > 256 for value in values):
            raise ValueError("domain policy values must contain 1 to 256 characters")
        return values

    @model_validator(mode="after")
    def validate_tool_ownership(self) -> MCPServerTrustDomain:
        prefix = f"{self.tool_namespace}."
        if any(not tool.startswith(prefix) for tool in self.allowed_tools):
            raise ValueError("every allowed tool must use the server's namespace")
        return self


class MCPDataFlowEdge(BaseModel):
    """Explicit cross-server route and label handling policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_tools: frozenset[str] = Field(min_length=1, max_length=1_000)
    destination_tools: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_labels: frozenset[str] = Field(default_factory=frozenset, max_length=256)
    redact_labels: frozenset[str] = Field(default_factory=frozenset, max_length=256)
    approval_labels: frozenset[str] = Field(default_factory=frozenset, max_length=256)

    @field_validator("source_tools", "destination_tools")
    @classmethod
    def validate_tools(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_TOOL_PATTERN, value) is None for value in values):
            raise ValueError("data-flow tools must be valid qualified tool names")
        return values

    @field_validator("allowed_labels", "redact_labels", "approval_labels")
    @classmethod
    def validate_labels(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_LABEL_PATTERN, value) is None for value in values):
            raise ValueError("data labels must be lowercase policy identifiers")
        return values

    @model_validator(mode="after")
    def validate_edge(self) -> MCPDataFlowEdge:
        if self.source_server_id == self.destination_server_id:
            raise ValueError("cross-server data-flow edges must connect different servers")
        groups = (self.allowed_labels, self.redact_labels, self.approval_labels)
        if not any(groups):
            raise ValueError("a data-flow edge must declare at least one label policy")
        if any(
            groups[index] & groups[other] for index in range(3) for other in range(index + 1, 3)
        ):
            raise ValueError("allowed, redacted, and approval labels must be disjoint")
        if MCPDataLabel.CREDENTIAL.value in set().union(*groups):
            raise ValueError("credentials cannot be allowed across MCP server trust domains")
        return self


class MCPServerIsolationPolicy(BaseModel):
    """Closed inventory of MCP trust domains and cross-server data-flow edges."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domains: tuple[MCPServerTrustDomain, ...] = Field(min_length=1, max_length=1_000)
    data_flow_edges: tuple[MCPDataFlowEdge, ...] = Field(default_factory=tuple, max_length=10_000)
    max_definitions: int = Field(default=1_000, ge=1, le=10_000)
    max_inputs_per_request: int = Field(default=32, ge=0, le=1_000)
    max_nodes_per_request: int = Field(default=10_000, ge=16, le=1_000_000)
    max_string_chars_per_request: int = Field(default=1_000_000, ge=1_024, le=100_000_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> MCPServerIsolationPolicy:
        server_ids = [domain.server_id for domain in self.domains]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("MCP server IDs must be unique trust domains")
        namespaces = [domain.tool_namespace.casefold() for domain in self.domains]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("MCP tool namespaces must be unique")
        credentials = [domain.credential_ref for domain in self.domains]
        if len(credentials) != len(set(credentials)):
            raise ValueError("MCP servers cannot share credential bindings")

        domains = {domain.server_id: domain for domain in self.domains}
        tool_owners: dict[str, str] = {}
        for domain in self.domains:
            for tool in domain.allowed_tools:
                normalized = tool.casefold()
                owner = tool_owners.setdefault(normalized, domain.server_id)
                if owner != domain.server_id:
                    raise ValueError("MCP tools cannot be owned by multiple trust domains")
        edge_routes: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
        prior_edges: list[MCPDataFlowEdge] = []
        for edge in self.data_flow_edges:
            source = domains.get(edge.source_server_id)
            destination = domains.get(edge.destination_server_id)
            if source is None or destination is None:
                raise ValueError("data-flow edges must reference declared MCP servers")
            if not edge.source_tools.issubset(source.allowed_tools):
                raise ValueError("data-flow source tools must be owned by the source server")
            if not edge.destination_tools.issubset(destination.allowed_tools):
                raise ValueError(
                    "data-flow destination tools must be owned by the destination server"
                )
            route = (
                edge.source_server_id,
                edge.destination_server_id,
                tuple(sorted(edge.source_tools)),
                tuple(sorted(edge.destination_tools)),
            )
            if route in edge_routes:
                raise ValueError("duplicate MCP data-flow routes are not allowed")
            if any(
                previous.source_server_id == edge.source_server_id
                and previous.destination_server_id == edge.destination_server_id
                and bool(previous.source_tools & edge.source_tools)
                and bool(previous.destination_tools & edge.destination_tools)
                for previous in prior_edges
            ):
                raise ValueError("overlapping MCP data-flow routes are ambiguous")
            edge_routes.add(route)
            prior_edges.append(edge)
        return self


class MCPGatewayPrincipal(BaseModel):
    """Authenticated user, agent, tenant, and scope context for a gateway call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    scopes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)


class MCPToolResultEnvelope(BaseModel):
    """Integrity-bound, source-labeled output from one MCP server tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    source_tool: str = Field(pattern=_TOOL_PATTERN)
    labels: frozenset[str] = Field(min_length=1, max_length=256)
    content: JsonValue = Field(exclude=True, repr=False)
    result_digest: str = Field(pattern=_DIGEST_PATTERN)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_LABEL_PATTERN, value) is None for value in values):
            raise ValueError("data labels must be lowercase policy identifiers")
        return values

    @model_validator(mode="after")
    def validate_integrity(self) -> MCPToolResultEnvelope:
        if not self.has_valid_integrity:
            raise ValueError("MCP tool result integrity check failed")
        return self

    @classmethod
    def create(cls, **values: Any) -> MCPToolResultEnvelope:
        """Create an envelope with a digest over source, labels, and content."""
        candidate = cls.model_construct(**values, result_digest="0" * 64)
        return cls(**values, result_digest=_digest(candidate.integrity_payload))

    @property
    def integrity_payload(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "labels": sorted(self.labels),
            "result_id": self.result_id,
            "source_server_id": self.source_server_id,
            "source_tool": self.source_tool,
        }

    @property
    def has_valid_integrity(self) -> bool:
        try:
            return self.result_digest == _digest(self.integrity_payload)
        except (TypeError, ValueError):
            return False


class MCPGatewayRequest(BaseModel):
    """Complete trusted context for one MCP gateway dispatch decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_tool: str = Field(pattern=_TOOL_PATTERN)
    principal: MCPGatewayPrincipal
    session_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    credential_ref: str = Field(pattern=_REFERENCE_PATTERN)
    initiating_server_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    initiating_tool: str | None = Field(default=None, pattern=_TOOL_PATTERN)
    arguments: dict[str, JsonValue] = Field(default_factory=dict, exclude=True, repr=False)
    inputs: tuple[MCPToolResultEnvelope, ...] = ()

    @model_validator(mode="after")
    def validate_initiator(self) -> MCPGatewayRequest:
        if (self.initiating_server_id is None) != (self.initiating_tool is None):
            raise ValueError("initiating_server_id and initiating_tool must be provided together")
        return self

    @property
    def request_digest(self) -> str:
        """Return a content-free binding for approvals and authorizations."""
        return _digest(
            {
                "arguments_digest": _digest(self.arguments),
                "credential_ref": self.credential_ref,
                "destination_server_id": self.destination_server_id,
                "destination_tool": self.destination_tool,
                "initiating_server_id": self.initiating_server_id,
                "initiating_tool": self.initiating_tool,
                "input_digests": sorted(item.result_digest for item in self.inputs),
                "principal": self.principal,
                "request_id": self.request_id,
                "session_id": self.session_id,
            }
        )


class MCPDataFlowApproval(BaseModel):
    """Out-of-band exception approval bound to one exact gateway request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    approval_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    request_digest: str = Field(pattern=_DIGEST_PATTERN)
    approved_labels: frozenset[str] = Field(min_length=1, max_length=256)
    approver_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    expires_at: datetime

    @field_validator("approved_labels")
    @classmethod
    def validate_labels(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_LABEL_PATTERN, value) is None for value in values):
            raise ValueError("approved labels must be lowercase policy identifiers")
        return values

    @field_validator("expires_at")
    @classmethod
    def require_aware_expiry(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval expiry must be timezone-aware")
        return value


class AuthorizedMCPGatewayRequest(BaseModel):
    """Gateway permit carrying only policy-approved or redacted inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_id: str = Field(pattern=_DIGEST_PATTERN)
    request_digest: str = Field(pattern=_DIGEST_PATTERN)
    destination_server_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_tool: str = Field(pattern=_TOOL_PATTERN)
    principal: MCPGatewayPrincipal
    session_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    arguments: dict[str, JsonValue] = Field(exclude=True, repr=False)
    inputs: tuple[MCPToolResultEnvelope, ...] = ()
    redacted_result_ids: tuple[str, ...] = ()
    approval_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)


class MCPIsolationFinding(BaseModel):
    """Content-free explanation of an MCP isolation decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: MCPIsolationCode
    severity: Severity
    message: str = Field(min_length=1, max_length=500)
    source_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    destination_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    labels: tuple[str, ...] = ()


class MCPIsolationAuditEvent(BaseModel):
    """Metadata-only evidence for one definition or dispatch decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    occurred_at: datetime
    operation: MCPIsolationOperation
    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK, GuardAction.REQUIRE_APPROVAL]
    finding_codes: tuple[MCPIsolationCode, ...]
    request_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    source_refs: tuple[str, ...] = ()
    destination_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    user_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    agent_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    session_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    labels: tuple[str, ...] = ()
    redaction_count: int = Field(default=0, ge=0)
    approval_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


class MCPIsolationResult(BaseModel):
    """Fail-closed MCP isolation result and content-free audit evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK, GuardAction.REQUIRE_APPROVAL]
    findings: tuple[MCPIsolationFinding, ...] = ()
    authorization: AuthorizedMCPGatewayRequest | None = None
    audit_event: MCPIsolationAuditEvent

    @property
    def is_allowed(self) -> bool:
        return self.action == GuardAction.ALLOW

    @property
    def is_authorized(self) -> bool:
        return self.action == GuardAction.ALLOW and self.authorization is not None

    @property
    def is_blocked(self) -> bool:
        return self.action == GuardAction.BLOCK

    @property
    def requires_approval(self) -> bool:
        return self.action == GuardAction.REQUIRE_APPROVAL
