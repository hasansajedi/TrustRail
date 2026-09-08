"""Typed contracts for MCP message signing and replay protection."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from trustrail.models.enums import GuardAction, Severity

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_KEY_ID_PATTERN = r"^sha256:[0-9a-f]{64}$"
_NONCE_PATTERN = r"^[A-Za-z0-9_-]{22,128}$"
_SIGNATURE_PATTERN = r"^[0-9a-f]{128}$"


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
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


def canonical_mcp_json(value: JsonValue) -> str:
    """Serialize JSON using trustrail's deterministic MCP signing profile."""
    return json.dumps(
        _canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_reference(value: str | bytes) -> str:
    """Return a content-free reference suitable for verification audit events."""
    encoded = value.encode() if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class MCPMessageType(StrEnum):
    """Direction-sensitive MCP message type."""

    REQUEST = "request"
    RESPONSE = "response"


class MCPMessageVerificationCode(StrEnum):
    """Stable machine-readable message-verification outcomes."""

    VERIFIED = "verified"
    UNSIGNED_MESSAGE = "unsigned_message"
    KEY_NOT_TRUSTED = "key_not_trusted"
    KEY_NOT_ACTIVE = "key_not_active"
    SENDER_MISMATCH = "sender_mismatch"
    RECIPIENT_MISMATCH = "recipient_mismatch"
    USER_MISMATCH = "user_mismatch"
    AGENT_MISMATCH = "agent_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    TOOL_DEFINITION_MISMATCH = "tool_definition_mismatch"
    MESSAGE_TYPE_MISMATCH = "message_type_mismatch"
    MESSAGE_NOT_YET_VALID = "message_not_yet_valid"
    MESSAGE_EXPIRED = "message_expired"
    MESSAGE_TOO_OLD = "message_too_old"
    TTL_EXCEEDED = "ttl_exceeded"
    ENVELOPE_TOO_LARGE = "envelope_too_large"
    SIGNATURE_INVALID = "signature_invalid"
    REPLAY_DETECTED = "replay_detected"
    REPLAY_STORE_FULL = "replay_store_full"
    REPLAY_STORE_ERROR = "replay_store_error"


class MCPReplayClaimStatus(StrEnum):
    """Atomic replay-store claim result."""

    STORED = "stored"
    REPLAYED = "replayed"
    FULL = "full"


class MCPMessageEnvelope(BaseModel):
    """Canonical signed envelope containing one complete MCP JSON-RPC payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    algorithm: Literal["Ed25519"] = "Ed25519"
    key_id: str = Field(pattern=_KEY_ID_PATTERN)
    message_type: MCPMessageType
    sender_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    recipient_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    user_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    session_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tool_definition_digest: str = Field(pattern=_DIGEST_PATTERN)
    issued_at: datetime
    expires_at: datetime
    nonce: str = Field(pattern=_NONCE_PATTERN)
    payload: dict[str, JsonValue]
    signature: str | None = Field(default=None, pattern=_SIGNATURE_PATTERN)

    @field_validator("issued_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("message timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_envelope(self) -> MCPMessageEnvelope:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        canonical_mcp_json(self.payload)
        return self

    @property
    def signing_payload(self) -> dict[str, JsonValue]:
        """Return every protected envelope field except the signature."""
        return self.model_dump(mode="json", exclude={"signature"})

    @property
    def signing_bytes(self) -> bytes:
        """Return the canonical bytes signed by Ed25519."""
        return canonical_mcp_json(self.signing_payload).encode()


class MCPTrustedKey(BaseModel):
    """Out-of-band Ed25519 public-key binding for one authenticated sender."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sender_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    public_key: bytes = Field(min_length=32, max_length=32, exclude=True, repr=False)
    active_from: datetime | None = None
    expires_at: datetime | None = None
    revoked: bool = False

    @field_validator("active_from", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("trusted-key timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_lifetime(self) -> MCPTrustedKey:
        if (
            self.active_from is not None
            and self.expires_at is not None
            and self.expires_at <= self.active_from
        ):
            raise ValueError("trusted key expires_at must be after active_from")
        return self

    @property
    def key_id(self) -> str:
        """Return the public-key fingerprint used by signed envelopes."""
        return content_reference(self.public_key)


class MCPMessageVerificationContext(BaseModel):
    """Authenticated local expectations that must match a signed message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_type: MCPMessageType
    sender_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    recipient_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    user_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    agent_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    session_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tool_definition_digest: str = Field(pattern=_DIGEST_PATTERN)


class MCPMessageSigningPolicy(BaseModel):
    """Freshness and resource bounds for MCP signed messages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_ttl_seconds: int = Field(default=60, ge=1, le=3_600)
    max_ttl_seconds: int = Field(default=300, ge=1, le=86_400)
    max_message_age_seconds: int = Field(default=300, ge=1, le=86_400)
    clock_skew_seconds: int = Field(default=30, ge=0, le=300)
    max_envelope_bytes: int = Field(default=1_048_576, ge=1_024, le=100_000_000)

    @model_validator(mode="after")
    def validate_ttl(self) -> MCPMessageSigningPolicy:
        if self.default_ttl_seconds > self.max_ttl_seconds:
            raise ValueError("default_ttl_seconds cannot exceed max_ttl_seconds")
        return self


class MCPMessageFinding(BaseModel):
    """Content-free explanation of an MCP message verification decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: MCPMessageVerificationCode
    severity: Severity
    message: str = Field(min_length=1, max_length=500)


class MCPMessageAuditEvent(BaseModel):
    """Metadata-only evidence for one message verification attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    occurred_at: datetime
    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    code: MCPMessageVerificationCode
    message_type: MCPMessageType | None = None
    message_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    key_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    sender_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    recipient_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    user_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    agent_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    session_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    tool_definition_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)
    nonce_ref: str | None = Field(default=None, pattern=_KEY_ID_PATTERN)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value


class MCPMessageVerificationResult(BaseModel):
    """Fail-closed verification result without message content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    findings: tuple[MCPMessageFinding, ...] = ()
    audit_event: MCPMessageAuditEvent

    @property
    def is_verified(self) -> bool:
        """Return whether the signed message passed every configured check."""
        return self.action == GuardAction.ALLOW
