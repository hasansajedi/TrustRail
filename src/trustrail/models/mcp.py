"""Typed models for MCP tool-definition integrity controls."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from trustrail.models.enums import GuardAction, Severity


def utcnow() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(tz=UTC)


def _canonicalize(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("object keys must remain unique after Unicode normalization")
            normalized[normalized_key] = _canonicalize(item)
        return normalized
    return value


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        _canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: JsonValue) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _field_digests(value: JsonValue, path: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        if not value:
            return {path or "/": _digest(value)}
        result: dict[str, str] = {}
        for key in sorted(value):
            result.update(_field_digests(value[key], f"{path}/{_escape_pointer(key)}"))
        return result
    if isinstance(value, list):
        if not value:
            return {path or "/": _digest(value)}
        result = {}
        for index, item in enumerate(value):
            result.update(_field_digests(item, f"{path}/{index}"))
        return result
    return {path or "/": _digest(value)}


class MCPToolDefinitionPhase(StrEnum):
    """Lifecycle phase represented by a signed definition bundle."""

    DISCOVERY = "discovery"
    APPROVAL = "approval"


class MCPToolDefinitionCode(StrEnum):
    """Stable machine-readable MCP tool-definition outcomes."""

    DUPLICATE_IDENTITY = "duplicate_identity"
    CONFUSABLE_IDENTITY = "confusable_identity"
    HIDDEN_INSTRUCTION = "hidden_instruction"
    UNICODE_SMUGGLING = "unicode_smuggling"
    CROSS_TOOL_REFERENCE = "cross_tool_reference"
    AMBIGUOUS_SEMANTICS = "ambiguous_semantics"
    DEFINITION_LIMIT_EXCEEDED = "definition_limit_exceeded"
    PIN_INVALID = "pin_invalid"
    DISCOVERY_MISMATCH = "discovery_mismatch"
    TOOL_NOT_APPROVED = "tool_not_approved"
    DEFINITION_CHANGED = "definition_changed"


class MCPDefinitionChangeKind(StrEnum):
    """Content-free classification for one changed definition field."""

    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"


class MCPToolDefinition(BaseModel):
    """Complete MCP tool metadata exposed to a model.

    ``server_id`` is supplied by the host from its authenticated connection
    configuration. The remaining fields mirror the security-relevant MCP tool
    definition and are included in the canonical digest.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    server_id: str = Field(min_length=1, max_length=256)
    name: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, max_length=512)
    description: str = Field(min_length=1, max_length=16_384)
    input_schema: dict[str, JsonValue] = Field(alias="inputSchema")
    output_schema: dict[str, JsonValue] | None = Field(default=None, alias="outputSchema")
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("server_id", "name", "description")
    @classmethod
    def reject_blank_or_control_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be blank")
        if any(unicodedata.category(character) == "Cc" for character in value):
            raise ValueError("value must not contain control characters")
        return value

    @model_validator(mode="after")
    def validate_canonical_form(self) -> MCPToolDefinition:
        # Fail early on non-finite numbers or normalized-key collisions.
        _canonical_json(self.canonical_payload)
        return self

    @property
    def identity(self) -> str:
        """Return the server-qualified identity used in pins and audit findings."""
        return f"{self.server_id}:{self.name}"

    @property
    def canonical_payload(self) -> dict[str, JsonValue]:
        """Return every security-relevant definition field in canonical shape."""
        return {
            "annotations": self.annotations,
            "description": self.description,
            "inputSchema": self.input_schema,
            "name": self.name,
            "outputSchema": self.output_schema,
            "serverId": self.server_id,
            "title": self.title,
        }

    @property
    def canonical_json(self) -> str:
        """Return deterministic Unicode-normalized JSON for cryptographic pinning."""
        return _canonical_json(self.canonical_payload)

    @property
    def definition_digest(self) -> str:
        """Return the SHA-256 digest of the complete canonical definition."""
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()

    @property
    def field_digests(self) -> dict[str, str]:
        """Return JSON-pointer-to-digest bindings for content-safe diffs."""
        return _field_digests(self.canonical_payload)


class MCPToolDefinitionPolicy(BaseModel):
    """Bounds and semantic requirements for untrusted MCP definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tools: int = Field(default=256, ge=1, le=10_000)
    max_nodes_per_definition: int = Field(default=4_096, ge=16, le=100_000)
    max_total_string_chars: int = Field(default=100_000, ge=256, le=10_000_000)
    max_depth: int = Field(default=32, ge=2, le=128)
    require_closed_object_schemas: bool = True
    require_schema_descriptions: bool = True
    reject_cross_tool_references: bool = True


class MCPToolFieldFingerprint(BaseModel):
    """Content-free fingerprint of one complete tool definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    server_id: str = Field(min_length=1, max_length=256)
    tool_name: str = Field(min_length=1, max_length=128)
    definition_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    field_digests: dict[str, str]

    @field_validator("field_digests")
    @classmethod
    def validate_field_digests(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("field_digests must not be empty")
        if any(not path.startswith("/") for path in value):
            raise ValueError("field digest paths must be JSON pointers")
        malformed = any(
            len(digest) != 64 or set(digest) - set("0123456789abcdef") for digest in value.values()
        )
        if malformed:
            raise ValueError("field digests must be lowercase SHA-256 values")
        return value

    @property
    def identity(self) -> str:
        """Return the server-qualified fingerprint identity."""
        return f"{self.server_id}:{self.tool_name}"


class MCPToolDefinitionBundle(BaseModel):
    """Signed discovery or approval snapshot for a set of MCP tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    phase: MCPToolDefinitionPhase
    fingerprints: tuple[MCPToolFieldFingerprint, ...] = Field(min_length=1)
    issued_at: datetime
    approved_by: str | None = Field(default=None, min_length=1, max_length=256)
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("issued_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("issued_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_phase(self) -> MCPToolDefinitionBundle:
        identities = [fingerprint.identity for fingerprint in self.fingerprints]
        if len(identities) != len(set(identities)):
            raise ValueError("bundle fingerprints must have unique identities")
        if self.phase == MCPToolDefinitionPhase.APPROVAL and self.approved_by is None:
            raise ValueError("approval bundles require approved_by")
        if self.phase == MCPToolDefinitionPhase.DISCOVERY and self.approved_by is not None:
            raise ValueError("discovery bundles cannot set approved_by")
        return self


class MCPDefinitionChange(BaseModel):
    """A content-free definition mutation suitable for logs and consent UIs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(pattern=r"^/")
    kind: MCPDefinitionChangeKind
    previous_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    current_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MCPToolDefinitionFinding(BaseModel):
    """Content-safe finding with hashed identity references and field paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: MCPToolDefinitionCode
    severity: Severity
    message: str
    identities: tuple[str, ...] = ()
    field_paths: tuple[str, ...] = ()
    changes: tuple[MCPDefinitionChange, ...] = ()


class MCPToolDefinitionResult(BaseModel):
    """Result of discovery, approval, or pre-execution verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: GuardAction
    findings: tuple[MCPToolDefinitionFinding, ...] = ()
    bundle: MCPToolDefinitionBundle | None = None

    @property
    def is_allowed(self) -> bool:
        """Return whether processing may continue."""
        return self.action == GuardAction.ALLOW

    @property
    def requires_approval(self) -> bool:
        """Return whether the current definition needs renewed consent."""
        return self.action == GuardAction.REQUIRE_APPROVAL


def fingerprint_definition(definition: MCPToolDefinition) -> MCPToolFieldFingerprint:
    """Build the content-free fingerprint placed in signed bundles."""
    return MCPToolFieldFingerprint(
        server_id=definition.server_id,
        tool_name=definition.name,
        definition_digest=definition.definition_digest,
        field_digests=definition.field_digests,
    )


def bundle_signing_payload(bundle: MCPToolDefinitionBundle) -> dict[str, Any]:
    """Return the exact fields covered by a bundle signature."""
    return bundle.model_dump(mode="json", exclude={"signature"})
