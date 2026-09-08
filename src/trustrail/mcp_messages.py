"""MCP message signing, identity binding, and replay protection."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import secrets
import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Literal, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic import JsonValue

from trustrail.exceptions import MCPMessageVerificationError
from trustrail.models.enums import GuardAction, Severity
from trustrail.models.mcp_messages import (
    MCPMessageAuditEvent,
    MCPMessageEnvelope,
    MCPMessageFinding,
    MCPMessageSigningPolicy,
    MCPMessageType,
    MCPMessageVerificationCode,
    MCPMessageVerificationContext,
    MCPMessageVerificationResult,
    MCPReplayClaimStatus,
    MCPTrustedKey,
    canonical_mcp_json,
    content_reference,
    utcnow,
)


class MCPReplayStore(Protocol):
    """Atomically reserve replay identifiers until their acceptance window ends."""

    def claim(
        self,
        replay_id: str,
        *,
        expires_at: datetime,
        now: datetime,
    ) -> MCPReplayClaimStatus:
        """Claim an unseen identifier, or report replay/capacity failure."""
        ...


class MCPMessageAuditSink(Protocol):
    """Persist content-free MCP verification events."""

    def emit(self, event: MCPMessageAuditEvent) -> None:
        """Persist one event without inspecting the signed payload."""
        ...


class MemoryMCPMessageAuditSink:
    """Bounded in-memory audit sink for tests and development."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[MCPMessageAuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, event: MCPMessageAuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[MCPMessageAuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class MemoryMCPReplayStore:
    """Capacity-bounded, process-local, atomic nonce replay store."""

    def __init__(self, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: dict[str, datetime] = {}
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._max_entries

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def claim(
        self,
        replay_id: str,
        *,
        expires_at: datetime,
        now: datetime,
    ) -> MCPReplayClaimStatus:
        if expires_at.tzinfo is None or now.tzinfo is None:
            raise ValueError("replay-store timestamps must be timezone-aware")
        if expires_at <= now:
            raise ValueError("replay-store expiration must be in the future")
        with self._lock:
            expired = [key for key, expiration in self._entries.items() if expiration <= now]
            for key in expired:
                del self._entries[key]
            if replay_id in self._entries:
                return MCPReplayClaimStatus.REPLAYED
            if len(self._entries) >= self._max_entries:
                return MCPReplayClaimStatus.FULL
            self._entries[replay_id] = expires_at
            return MCPReplayClaimStatus.STORED

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class MCPMessageSigner:
    """Create Ed25519-signed MCP envelopes for an authenticated sender."""

    def __init__(
        self,
        private_key: Ed25519PrivateKey,
        *,
        sender_id: str,
        policy: MCPMessageSigningPolicy | None = None,
    ) -> None:
        self._private_key = private_key
        self._sender_id = sender_id
        self._policy = (policy or MCPMessageSigningPolicy()).model_copy(deep=True)
        public_key = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        self._trusted_key = MCPTrustedKey(sender_id=sender_id, public_key=public_key)

    @classmethod
    def generate(
        cls,
        *,
        sender_id: str,
        policy: MCPMessageSigningPolicy | None = None,
    ) -> MCPMessageSigner:
        """Create a signer with a newly generated Ed25519 private key."""
        return cls(Ed25519PrivateKey.generate(), sender_id=sender_id, policy=policy)

    @property
    def key_id(self) -> str:
        return self._trusted_key.key_id

    @property
    def trusted_key(self) -> MCPTrustedKey:
        """Return the public key binding to distribute over an authenticated channel."""
        return self._trusted_key.model_copy(deep=True)

    def sign(
        self,
        payload: dict[str, JsonValue],
        *,
        message_type: MCPMessageType,
        recipient_id: str,
        user_id: str,
        agent_id: str,
        session_id: str,
        tool_definition_digest: str,
        now: datetime | None = None,
        ttl_seconds: int | None = None,
        nonce: str | None = None,
    ) -> MCPMessageEnvelope:
        """Sign one complete JSON-RPC request or response and its security context."""
        current_time = now or utcnow()
        ttl = self._policy.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl < 1 or ttl > self._policy.max_ttl_seconds:
            raise ValueError("ttl_seconds must be within the signing policy maximum")
        unsigned = MCPMessageEnvelope(
            key_id=self.key_id,
            message_type=message_type,
            sender_id=self._sender_id,
            recipient_id=recipient_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            tool_definition_digest=tool_definition_digest,
            issued_at=current_time,
            expires_at=current_time + timedelta(seconds=ttl),
            nonce=nonce if nonce is not None else secrets.token_urlsafe(32),
            payload=payload,
        )
        if len(unsigned.signing_bytes) > self._policy.max_envelope_bytes:
            raise ValueError("canonical MCP envelope exceeds max_envelope_bytes")
        signature = self._private_key.sign(unsigned.signing_bytes).hex()
        return unsigned.model_copy(update={"signature": signature}, deep=True)


class MCPMessageVerifier:
    """Fail-closed verifier for signed MCP requests and responses."""

    def __init__(
        self,
        trusted_keys: Iterable[MCPTrustedKey],
        *,
        replay_store: MCPReplayStore | None = None,
        audit_sink: MCPMessageAuditSink | None = None,
        policy: MCPMessageSigningPolicy | None = None,
    ) -> None:
        keys = tuple(key.model_copy(deep=True) for key in trusted_keys)
        by_id = {key.key_id: key for key in keys}
        if not keys:
            raise ValueError("at least one trusted key is required")
        if len(by_id) != len(keys):
            raise ValueError("trusted keys must have unique public-key fingerprints")
        self._trusted_keys = by_id
        self._replay_store = replay_store if replay_store is not None else MemoryMCPReplayStore()
        self._audit_sink = audit_sink
        self._policy = (policy or MCPMessageSigningPolicy()).model_copy(deep=True)

    @property
    def policy(self) -> MCPMessageSigningPolicy:
        return self._policy.model_copy(deep=True)

    def verify(
        self,
        envelope: MCPMessageEnvelope | None,
        context: MCPMessageVerificationContext,
        *,
        now: datetime | None = None,
    ) -> MCPMessageVerificationResult:
        """Authenticate, authorize, freshness-check, and atomically claim a message."""
        current_time = now or utcnow()
        if envelope is None or envelope.signature is None:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.UNSIGNED_MESSAGE,
                "MCP message is missing its required signature",
                current_time,
            )

        trusted_key = self._trusted_keys.get(envelope.key_id)
        if trusted_key is None:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.KEY_NOT_TRUSTED,
                "MCP message key is not present in the authenticated trust store",
                current_time,
            )
        if (
            trusted_key.revoked
            or (
                trusted_key.active_from is not None
                and (
                    current_time < trusted_key.active_from
                    or envelope.issued_at < trusted_key.active_from
                )
            )
            or (
                trusted_key.expires_at is not None
                and (
                    current_time >= trusted_key.expires_at
                    or envelope.issued_at >= trusted_key.expires_at
                )
            )
        ):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.KEY_NOT_ACTIVE,
                "MCP message key is revoked, expired, or not yet active",
                current_time,
            )

        binding_checks = (
            (
                trusted_key.sender_id == envelope.sender_id == context.sender_id,
                MCPMessageVerificationCode.SENDER_MISMATCH,
                "MCP sender does not match the trusted key and authenticated peer",
            ),
            (
                envelope.recipient_id == context.recipient_id,
                MCPMessageVerificationCode.RECIPIENT_MISMATCH,
                "MCP message audience does not match this recipient",
            ),
            (
                envelope.user_id == context.user_id,
                MCPMessageVerificationCode.USER_MISMATCH,
                "MCP message does not match the authenticated user",
            ),
            (
                envelope.agent_id == context.agent_id,
                MCPMessageVerificationCode.AGENT_MISMATCH,
                "MCP message does not match the authenticated agent",
            ),
            (
                envelope.session_id == context.session_id,
                MCPMessageVerificationCode.SESSION_MISMATCH,
                "MCP message does not match the active session",
            ),
            (
                envelope.tool_definition_digest == context.tool_definition_digest,
                MCPMessageVerificationCode.TOOL_DEFINITION_MISMATCH,
                "MCP message does not match the approved tool definition",
            ),
            (
                envelope.message_type == context.message_type,
                MCPMessageVerificationCode.MESSAGE_TYPE_MISMATCH,
                "MCP request/response type does not match the processing boundary",
            ),
        )
        for matches, code, message in binding_checks:
            if not matches:
                return self._blocked(envelope, code, message, current_time)

        if envelope.issued_at > current_time + timedelta(seconds=self._policy.clock_skew_seconds):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.MESSAGE_NOT_YET_VALID,
                "MCP message timestamp is too far in the future",
                current_time,
            )
        if envelope.issued_at < current_time - timedelta(
            seconds=self._policy.max_message_age_seconds
        ):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.MESSAGE_TOO_OLD,
                "MCP message timestamp is outside the accepted freshness window",
                current_time,
            )
        if envelope.expires_at <= current_time - timedelta(seconds=self._policy.clock_skew_seconds):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.MESSAGE_EXPIRED,
                "MCP message has expired",
                current_time,
            )
        if envelope.expires_at - envelope.issued_at > timedelta(
            seconds=self._policy.max_ttl_seconds
        ):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.TTL_EXCEEDED,
                "MCP message lifetime exceeds the accepted maximum",
                current_time,
            )
        if len(envelope.signing_bytes) > self._policy.max_envelope_bytes:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.ENVELOPE_TOO_LARGE,
                "Canonical MCP envelope exceeds the verification size limit",
                current_time,
            )

        try:
            public_key = Ed25519PublicKey.from_public_bytes(trusted_key.public_key)
            public_key.verify(bytes.fromhex(envelope.signature), envelope.signing_bytes)
        except (InvalidSignature, ValueError):
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.SIGNATURE_INVALID,
                "MCP message signature verification failed",
                current_time,
            )

        replay_id = hashlib.sha256(
            canonical_mcp_json({"keyId": envelope.key_id, "nonce": envelope.nonce}).encode()
        ).hexdigest()
        replay_until = max(
            envelope.expires_at,
            envelope.issued_at + timedelta(seconds=self._policy.max_message_age_seconds),
        ) + timedelta(seconds=self._policy.clock_skew_seconds)
        try:
            claim = self._replay_store.claim(
                replay_id,
                expires_at=replay_until,
                now=current_time,
            )
        except Exception:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.REPLAY_STORE_ERROR,
                "MCP replay state could not be checked",
                current_time,
            )
        if claim == MCPReplayClaimStatus.REPLAYED:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.REPLAY_DETECTED,
                "MCP message nonce has already been processed",
                current_time,
            )
        if claim != MCPReplayClaimStatus.STORED:
            return self._blocked(
                envelope,
                MCPMessageVerificationCode.REPLAY_STORE_FULL,
                "MCP replay store has reached its configured capacity",
                current_time,
            )
        return self._result(
            envelope,
            GuardAction.ALLOW,
            MCPMessageVerificationCode.VERIFIED,
            (),
            current_time,
        )

    def require_verified(
        self,
        envelope: MCPMessageEnvelope | None,
        context: MCPMessageVerificationContext,
        *,
        now: datetime | None = None,
    ) -> dict[str, JsonValue]:
        """Return a defensive payload copy or raise before message processing."""
        snapshot = envelope.model_copy(deep=True) if envelope is not None else None
        result = self.verify(snapshot, context, now=now)
        if not result.is_verified or snapshot is None:
            raise MCPMessageVerificationError(result)
        return copy.deepcopy(snapshot.payload)

    def _blocked(
        self,
        envelope: MCPMessageEnvelope | None,
        code: MCPMessageVerificationCode,
        message: str,
        now: datetime,
    ) -> MCPMessageVerificationResult:
        finding = MCPMessageFinding(code=code, severity=Severity.CRITICAL, message=message)
        return self._result(envelope, GuardAction.BLOCK, code, (finding,), now)

    def _result(
        self,
        envelope: MCPMessageEnvelope | None,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK],
        code: MCPMessageVerificationCode,
        findings: tuple[MCPMessageFinding, ...],
        now: datetime,
    ) -> MCPMessageVerificationResult:
        event = MCPMessageAuditEvent(
            occurred_at=now,
            action=action,
            code=code,
            message_type=envelope.message_type if envelope is not None else None,
            message_ref=(
                content_reference(envelope.signature)
                if envelope is not None and envelope.signature is not None
                else None
            ),
            key_ref=(content_reference(envelope.key_id) if envelope is not None else None),
            sender_ref=(content_reference(envelope.sender_id) if envelope is not None else None),
            recipient_ref=(
                content_reference(envelope.recipient_id) if envelope is not None else None
            ),
            user_ref=(content_reference(envelope.user_id) if envelope is not None else None),
            agent_ref=(content_reference(envelope.agent_id) if envelope is not None else None),
            session_ref=(content_reference(envelope.session_id) if envelope is not None else None),
            tool_definition_ref=(
                content_reference(envelope.tool_definition_digest) if envelope is not None else None
            ),
            nonce_ref=(content_reference(envelope.nonce) if envelope is not None else None),
        )
        if self._audit_sink is not None:
            with contextlib.suppress(Exception):
                self._audit_sink.emit(event)
        return MCPMessageVerificationResult(
            action=action,
            findings=findings,
            audit_event=event,
        )
