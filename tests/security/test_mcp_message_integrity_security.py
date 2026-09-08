"""Bypass-oriented corpus for MCP message integrity and identity binding."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustrail import (
    MCPMessageSigner,
    MCPMessageType,
    MCPMessageVerificationCode,
    MCPMessageVerificationContext,
    MCPMessageVerifier,
)

CORPUS_PATH = Path(__file__).parent.parent / "security_corpus" / "mcp_message_integrity.json"
CASES: list[dict[str, str | None]] = json.loads(CORPUS_PATH.read_text())
NOW = datetime(2026, 9, 8, 16, tzinfo=UTC)
DIGEST = "a" * 64


@pytest.mark.parametrize("case", CASES, ids=[str(case["id"]) for case in CASES])
def test_signed_message_bypass_corpus_is_rejected(case: dict[str, str | None]):
    signer = MCPMessageSigner.generate(sender_id="trusted-client")
    envelope = signer.sign(
        {"jsonrpc": "2.0", "id": "call-1", "method": "tools/call"},
        message_type=MCPMessageType.REQUEST,
        recipient_id="trusted-server",
        user_id="authenticated-user",
        agent_id="authorized-agent",
        session_id="active-session",
        tool_definition_digest=DIGEST,
        now=NOW,
        nonce="security_nonce_1234567890",
    )
    context = MCPMessageVerificationContext(
        message_type=MCPMessageType.REQUEST,
        sender_id="trusted-client",
        recipient_id="trusted-server",
        user_id="authenticated-user",
        agent_id="authorized-agent",
        session_id="active-session",
        tool_definition_digest=DIGEST,
    )
    mutation = str(case["mutation"])
    if mutation == "payload":
        candidate = envelope.model_copy(
            update={"payload": {**envelope.payload, "method": case["value"]}}, deep=True
        )
    else:
        candidate = envelope.model_copy(update={mutation: case["value"]}, deep=True)

    result = MCPMessageVerifier((signer.trusted_key,)).verify(candidate, context, now=NOW)

    assert result.findings[0].code == MCPMessageVerificationCode(case["expected_code"])
    serialized = result.model_dump_json()
    assert "tools/call" not in serialized
    assert "authenticated-user" not in serialized
    assert "authorized-agent" not in serialized
    assert "active-session" not in serialized


def test_replay_and_first_contact_key_substitution_are_rejected():
    trusted_signer = MCPMessageSigner.generate(sender_id="trusted-client")
    attacker_signer = MCPMessageSigner.generate(sender_id="trusted-client")
    verifier = MCPMessageVerifier((trusted_signer.trusted_key,))
    context = MCPMessageVerificationContext(
        message_type=MCPMessageType.REQUEST,
        sender_id="trusted-client",
        recipient_id="trusted-server",
        user_id="authenticated-user",
        agent_id="authorized-agent",
        session_id="active-session",
        tool_definition_digest=DIGEST,
    )

    def sign(signer: MCPMessageSigner):
        return signer.sign(
            {"jsonrpc": "2.0", "id": "call-2", "method": "tools/call"},
            message_type=MCPMessageType.REQUEST,
            recipient_id="trusted-server",
            user_id="authenticated-user",
            agent_id="authorized-agent",
            session_id="active-session",
            tool_definition_digest=DIGEST,
            now=NOW,
        )

    trusted = sign(trusted_signer)
    assert verifier.verify(trusted, context, now=NOW).is_verified
    assert verifier.verify(trusted, context, now=NOW).findings[0].code == (
        MCPMessageVerificationCode.REPLAY_DETECTED
    )
    assert verifier.verify(sign(attacker_signer), context, now=NOW).findings[0].code == (
        MCPMessageVerificationCode.KEY_NOT_TRUSTED
    )
