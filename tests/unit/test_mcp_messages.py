"""Unit coverage for MCP message signing and replay protection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

from trustrail import (
    GuardAction,
    MCPMessageEnvelope,
    MCPMessageSigner,
    MCPMessageSigningPolicy,
    MCPMessageType,
    MCPMessageVerificationCode,
    MCPMessageVerificationContext,
    MCPMessageVerificationError,
    MCPMessageVerifier,
    MCPReplayClaimStatus,
    MCPTrustedKey,
    MemoryMCPMessageAuditSink,
    MemoryMCPReplayStore,
)

NOW = datetime(2026, 9, 8, 12, tzinfo=UTC)
DIGEST = "a" * 64
PAYLOAD = {
    "jsonrpc": "2.0",
    "id": "call-1",
    "method": "tools/call",
    "params": {"arguments": {"query": "quarterly report"}, "name": "records.search"},
}


def _signer(sender_id: str = "mcp-client") -> MCPMessageSigner:
    return MCPMessageSigner.generate(sender_id=sender_id)


def _context(
    *,
    message_type: MCPMessageType = MCPMessageType.REQUEST,
    sender_id: str = "mcp-client",
    recipient_id: str = "records-server",
    user_id: str = "user-42",
    agent_id: str = "research-agent",
    session_id: str = "session-7",
    tool_definition_digest: str = DIGEST,
) -> MCPMessageVerificationContext:
    return MCPMessageVerificationContext(
        message_type=message_type,
        sender_id=sender_id,
        recipient_id=recipient_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        tool_definition_digest=tool_definition_digest,
    )


def _envelope(
    signer: MCPMessageSigner,
    *,
    payload: dict[str, object] | None = None,
    message_type: MCPMessageType = MCPMessageType.REQUEST,
    recipient_id: str = "records-server",
    user_id: str = "user-42",
    agent_id: str = "research-agent",
    session_id: str = "session-7",
    tool_definition_digest: str = DIGEST,
    now: datetime = NOW,
    ttl_seconds: int | None = None,
    nonce: str = "fixed_nonce_value_1234567890",
) -> MCPMessageEnvelope:
    return signer.sign(
        payload or PAYLOAD,  # type: ignore[arg-type]
        message_type=message_type,
        recipient_id=recipient_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        tool_definition_digest=tool_definition_digest,
        now=now,
        ttl_seconds=ttl_seconds,
        nonce=nonce,
    )


def test_valid_signed_request_is_verified_once_and_returns_defensive_payload_copy():
    signer = _signer()
    verifier = MCPMessageVerifier((signer.trusted_key,))
    envelope = _envelope(signer)

    payload = verifier.require_verified(envelope, _context(), now=NOW)
    payload["method"] = "changed-after-verification"

    assert envelope.payload["method"] == "tools/call"
    replay = verifier.verify(envelope, _context(), now=NOW)
    assert replay.action == GuardAction.BLOCK
    assert replay.findings[0].code == MCPMessageVerificationCode.REPLAY_DETECTED


def test_transport_serialization_preserves_the_verified_signature():
    signer = _signer()
    original = _envelope(signer)
    received = MCPMessageEnvelope.model_validate_json(original.model_dump_json())

    result = MCPMessageVerifier((signer.trusted_key,)).verify(received, _context(), now=NOW)

    assert received.signing_bytes == original.signing_bytes
    assert result.is_verified


def test_canonical_signatures_are_stable_across_mapping_order_and_unicode_form():
    signer = _signer()
    first = signer.sign(
        {"jsonrpc": "2.0", "result": {"b": "cafe\u0301", "a": 1}},
        message_type=MCPMessageType.RESPONSE,
        recipient_id="mcp-client",
        user_id="user-42",
        agent_id="research-agent",
        session_id="session-7",
        tool_definition_digest=DIGEST,
        now=NOW,
        nonce="canonical_nonce_123456789",
    )
    second = signer.sign(
        {"result": {"a": 1, "b": "caf\u00e9"}, "jsonrpc": "2.0"},
        message_type=MCPMessageType.RESPONSE,
        recipient_id="mcp-client",
        user_id="user-42",
        agent_id="research-agent",
        session_id="session-7",
        tool_definition_digest=DIGEST,
        now=NOW,
        nonce="canonical_nonce_123456789",
    )

    assert first.signing_bytes == second.signing_bytes
    assert first.signature == second.signature


@pytest.mark.parametrize(
    ("update", "context"),
    [
        ({"payload": {"jsonrpc": "2.0", "method": "tools/list"}}, _context()),
        ({"recipient_id": "other-server"}, _context(recipient_id="other-server")),
        ({"user_id": "user-99"}, _context(user_id="user-99")),
        ({"agent_id": "other-agent"}, _context(agent_id="other-agent")),
        ({"session_id": "session-99"}, _context(session_id="session-99")),
        (
            {"tool_definition_digest": "b" * 64},
            _context(tool_definition_digest="b" * 64),
        ),
        (
            {"message_type": MCPMessageType.RESPONSE},
            _context(message_type=MCPMessageType.RESPONSE),
        ),
        ({"issued_at": NOW + timedelta(seconds=1)}, _context()),
        ({"expires_at": NOW + timedelta(seconds=90)}, _context()),
        ({"nonce": "different_nonce_123456789"}, _context()),
    ],
)
def test_signature_covers_payload_and_every_security_binding(update, context):
    signer = _signer()
    envelope = _envelope(signer).model_copy(update=update, deep=True)

    result = MCPMessageVerifier((signer.trusted_key,)).verify(envelope, context, now=NOW)

    assert result.findings[0].code == MCPMessageVerificationCode.SIGNATURE_INVALID


@pytest.mark.parametrize(
    ("context", "code"),
    [
        (_context(sender_id="other-client"), MCPMessageVerificationCode.SENDER_MISMATCH),
        (_context(recipient_id="other-server"), MCPMessageVerificationCode.RECIPIENT_MISMATCH),
        (_context(user_id="other-user"), MCPMessageVerificationCode.USER_MISMATCH),
        (_context(agent_id="other-agent"), MCPMessageVerificationCode.AGENT_MISMATCH),
        (_context(session_id="other-session"), MCPMessageVerificationCode.SESSION_MISMATCH),
        (
            _context(tool_definition_digest="b" * 64),
            MCPMessageVerificationCode.TOOL_DEFINITION_MISMATCH,
        ),
        (
            _context(message_type=MCPMessageType.RESPONSE),
            MCPMessageVerificationCode.MESSAGE_TYPE_MISMATCH,
        ),
    ],
)
def test_authenticated_processing_context_is_enforced(context, code):
    signer = _signer()

    result = MCPMessageVerifier((signer.trusted_key,)).verify(_envelope(signer), context, now=NOW)

    assert result.findings[0].code == code


def test_unsigned_downgrade_and_unknown_key_are_rejected():
    signer = _signer()
    envelope = _envelope(signer)
    verifier = MCPMessageVerifier((signer.trusted_key,))

    unsigned = verifier.verify(envelope.model_copy(update={"signature": None}), _context(), now=NOW)
    unknown = verifier.verify(_envelope(_signer("attacker")), _context(), now=NOW)

    assert unsigned.findings[0].code == MCPMessageVerificationCode.UNSIGNED_MESSAGE
    assert unknown.findings[0].code == MCPMessageVerificationCode.KEY_NOT_TRUSTED


@pytest.mark.parametrize("state", ["revoked", "future", "expired"])
def test_revoked_expired_or_not_yet_active_keys_are_rejected(state: str):
    signer = _signer()
    values = signer.trusted_key.model_dump()
    values["public_key"] = signer.trusted_key.public_key
    if state == "revoked":
        values["revoked"] = True
    elif state == "future":
        values["active_from"] = NOW + timedelta(seconds=1)
    else:
        values["expires_at"] = NOW
    trusted_key = MCPTrustedKey(**values)

    result = MCPMessageVerifier((trusted_key,)).verify(_envelope(signer), _context(), now=NOW)

    assert result.findings[0].code == MCPMessageVerificationCode.KEY_NOT_ACTIVE


def test_key_identity_binding_prevents_sender_substitution():
    signer = _signer()
    rebound_key = MCPTrustedKey(sender_id="attacker", public_key=signer.trusted_key.public_key)

    result = MCPMessageVerifier((rebound_key,)).verify(_envelope(signer), _context(), now=NOW)

    assert result.findings[0].code == MCPMessageVerificationCode.SENDER_MISMATCH


def test_invalid_signature_cannot_poison_nonce_before_valid_message_arrives():
    signer = _signer()
    envelope = _envelope(signer)
    verifier = MCPMessageVerifier((signer.trusted_key,))
    forged = envelope.model_copy(update={"signature": "0" * 128})

    rejected = verifier.verify(forged, _context(), now=NOW)
    accepted = verifier.verify(envelope, _context(), now=NOW)

    assert rejected.findings[0].code == MCPMessageVerificationCode.SIGNATURE_INVALID
    assert accepted.is_verified


def test_replay_claim_is_atomic_under_concurrency():
    signer = _signer()
    verifier = MCPMessageVerifier((signer.trusted_key,))
    envelope = _envelope(signer)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: verifier.verify(envelope, _context(), now=NOW), range(32))
        )

    assert sum(result.is_verified for result in results) == 1
    assert (
        sum(
            result.findings[0].code == MCPMessageVerificationCode.REPLAY_DETECTED
            for result in results
            if result.findings
        )
        == 31
    )


def test_replay_store_is_capacity_bounded_and_evicts_expired_entries():
    store = MemoryMCPReplayStore(max_entries=1)

    assert store.claim("first", expires_at=NOW + timedelta(seconds=1), now=NOW) == (
        MCPReplayClaimStatus.STORED
    )
    assert store.claim("second", expires_at=NOW + timedelta(seconds=2), now=NOW) == (
        MCPReplayClaimStatus.FULL
    )
    assert (
        store.claim(
            "second",
            expires_at=NOW + timedelta(seconds=2),
            now=NOW + timedelta(seconds=1),
        )
        == MCPReplayClaimStatus.STORED
    )
    assert store.size == 1


def test_verifier_fails_closed_when_replay_store_is_full():
    first_signer = _signer()
    second_signer = _signer("second-client")
    store = MemoryMCPReplayStore(max_entries=1)
    verifier = MCPMessageVerifier(
        (first_signer.trusted_key, second_signer.trusted_key), replay_store=store
    )
    verifier.require_verified(_envelope(first_signer), _context(), now=NOW)

    result = verifier.verify(
        _envelope(second_signer, nonce="second_nonce_value_123456789"),
        _context(sender_id="second-client"),
        now=NOW,
    )

    assert result.findings[0].code == MCPMessageVerificationCode.REPLAY_STORE_FULL


@pytest.mark.parametrize(
    ("envelope_time", "verification_time", "code"),
    [
        (NOW + timedelta(seconds=31), NOW, MCPMessageVerificationCode.MESSAGE_NOT_YET_VALID),
        (NOW, NOW + timedelta(seconds=301), MCPMessageVerificationCode.MESSAGE_TOO_OLD),
        (NOW, NOW + timedelta(seconds=91), MCPMessageVerificationCode.MESSAGE_EXPIRED),
    ],
)
def test_timestamp_freshness_is_enforced(envelope_time, verification_time, code):
    signer = _signer()
    policy = MCPMessageSigningPolicy(max_message_age_seconds=600)
    verifier = MCPMessageVerifier((signer.trusted_key,), policy=policy)
    envelope = _envelope(signer, now=envelope_time)

    if code == MCPMessageVerificationCode.MESSAGE_TOO_OLD:
        verifier = MCPMessageVerifier((signer.trusted_key,))
    result = verifier.verify(envelope, _context(), now=verification_time)

    assert result.findings[0].code == code


def test_excessive_ttl_and_envelope_size_are_rejected():
    signing_policy = MCPMessageSigningPolicy(max_ttl_seconds=600, max_envelope_bytes=10_000)
    signer = MCPMessageSigner.generate(sender_id="mcp-client", policy=signing_policy)
    long_lived = _envelope(signer, ttl_seconds=301)
    oversized = _envelope(signer, payload={"jsonrpc": "2.0", "result": "x" * 2_000})

    ttl_result = MCPMessageVerifier((signer.trusted_key,)).verify(long_lived, _context(), now=NOW)
    size_result = MCPMessageVerifier(
        (signer.trusted_key,), policy=MCPMessageSigningPolicy(max_envelope_bytes=1_024)
    ).verify(oversized, _context(), now=NOW)

    assert ttl_result.findings[0].code == MCPMessageVerificationCode.TTL_EXCEEDED
    assert size_result.findings[0].code == MCPMessageVerificationCode.ENVELOPE_TOO_LARGE


class _FailingReplayStore:
    def claim(self, replay_id, *, expires_at, now):
        raise RuntimeError("backend unavailable")


def test_replay_store_failure_is_fail_closed():
    signer = _signer()
    verifier = MCPMessageVerifier((signer.trusted_key,), replay_store=_FailingReplayStore())

    result = verifier.verify(_envelope(signer), _context(), now=NOW)

    assert result.findings[0].code == MCPMessageVerificationCode.REPLAY_STORE_ERROR


def test_audit_events_are_bounded_and_contain_no_message_content_or_raw_identifiers():
    signer = _signer()
    sink = MemoryMCPMessageAuditSink(max_events=1)
    verifier = MCPMessageVerifier((signer.trusted_key,), audit_sink=sink)
    envelope = _envelope(signer)

    verifier.verify(envelope.model_copy(update={"signature": None}), _context(), now=NOW)
    verifier.verify(envelope, _context(), now=NOW)
    serialized = sink.events[0].model_dump_json()

    assert len(sink.events) == 1
    assert "quarterly report" not in serialized
    assert "user-42" not in serialized
    assert "research-agent" not in serialized
    assert "session-7" not in serialized
    assert envelope.nonce not in serialized


def test_require_verified_raises_typed_error_on_failure():
    signer = _signer()
    verifier = MCPMessageVerifier((signer.trusted_key,))

    with pytest.raises(MCPMessageVerificationError) as caught:
        verifier.require_verified(None, _context(), now=NOW)

    assert caught.value.result.findings[0].code == MCPMessageVerificationCode.UNSIGNED_MESSAGE


def test_envelope_rejects_short_nonce_and_normalized_key_collisions():
    signer = _signer()
    envelope = _envelope(signer)
    values = envelope.model_dump(exclude={"nonce", "signature"})

    with pytest.raises(ValidationError):
        MCPMessageEnvelope(**values, nonce="predictable", signature=envelope.signature)
    with pytest.raises(ValueError, match="unique after Unicode normalization"):
        signer.sign(
            {"\u00e9": 1, "e\u0301": 2},
            message_type=MCPMessageType.REQUEST,
            recipient_id="records-server",
            user_id="user-42",
            agent_id="research-agent",
            session_id="session-7",
            tool_definition_digest=DIGEST,
            now=NOW,
        )


def test_private_key_constructor_and_signing_policy_bounds():
    signer = MCPMessageSigner(Ed25519PrivateKey.generate(), sender_id="mcp-client")

    with pytest.raises(ValueError, match="ttl_seconds"):
        _envelope(signer, ttl_seconds=0)
    with pytest.raises(ValidationError):
        _envelope(signer, nonce="")
    with pytest.raises(ValidationError):
        MCPMessageSigningPolicy(default_ttl_seconds=301, max_ttl_seconds=300)
    with pytest.raises(ValueError, match="at least one trusted key"):
        MCPMessageVerifier(())
