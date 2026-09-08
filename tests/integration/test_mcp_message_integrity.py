"""Integration coverage for mutually signed MCP request and response boundaries."""

from __future__ import annotations

from datetime import UTC, datetime

from trustrail import (
    MCPMessageSigner,
    MCPMessageType,
    MCPMessageVerificationCode,
    MCPMessageVerificationContext,
    MCPMessageVerificationError,
    MCPMessageVerifier,
    MCPToolDefinition,
    MCPToolDefinitionGuard,
)

NOW = datetime(2026, 9, 8, 14, tzinfo=UTC)


def _definition() -> MCPToolDefinition:
    return MCPToolDefinition(
        server_id="inventory-production",
        name="inventory.lookup",
        description="Look up one inventory item by its reviewed stock identifier.",
        inputSchema={
            "type": "object",
            "properties": {
                "sku": {
                    "type": "string",
                    "pattern": "^[A-Z0-9-]{1,32}$",
                    "description": "Stock identifier selected by the user.",
                }
            },
            "required": ["sku"],
            "additionalProperties": False,
        },
        outputSchema={
            "type": "object",
            "properties": {
                "available": {
                    "type": "boolean",
                    "description": "Whether the stock item is currently available.",
                }
            },
            "required": ["available"],
            "additionalProperties": False,
        },
        annotations={"readOnlyHint": True},
    )


def _context(
    message_type: MCPMessageType,
    sender_id: str,
    recipient_id: str,
    definition_digest: str,
) -> MCPMessageVerificationContext:
    return MCPMessageVerificationContext(
        message_type=message_type,
        sender_id=sender_id,
        recipient_id=recipient_id,
        user_id="warehouse-user",
        agent_id="inventory-agent",
        session_id="inventory-session",
        tool_definition_digest=definition_digest,
    )


def test_request_and_response_are_verified_at_both_dispatch_boundaries():
    definition = _definition()
    definition_guard = MCPToolDefinitionGuard(b"integration-definition-key-at-least-32-bytes")
    discovery = definition_guard.require_discovery((definition,), now=NOW)
    approval = definition_guard.require_approval(
        (definition,), discovery, approved_by="inventory-owner", now=NOW
    )
    definition_guard.require_execution(definition, approval)

    client_signer = MCPMessageSigner.generate(sender_id="inventory-client")
    server_signer = MCPMessageSigner.generate(sender_id="inventory-server")
    server_verifier = MCPMessageVerifier((client_signer.trusted_key,))
    client_verifier = MCPMessageVerifier((server_signer.trusted_key,))
    digest = definition.definition_digest

    request = client_signer.sign(
        {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "tools/call",
            "params": {"name": definition.name, "arguments": {"sku": "ABC-123"}},
        },
        message_type=MCPMessageType.REQUEST,
        recipient_id="inventory-server",
        user_id="warehouse-user",
        agent_id="inventory-agent",
        session_id="inventory-session",
        tool_definition_digest=digest,
        now=NOW,
    )
    received_request = server_verifier.require_verified(
        request,
        _context(MCPMessageType.REQUEST, "inventory-client", "inventory-server", digest),
        now=NOW,
    )
    assert received_request["method"] == "tools/call"

    response = server_signer.sign(
        {"jsonrpc": "2.0", "id": "request-1", "result": {"available": True}},
        message_type=MCPMessageType.RESPONSE,
        recipient_id="inventory-client",
        user_id="warehouse-user",
        agent_id="inventory-agent",
        session_id="inventory-session",
        tool_definition_digest=digest,
        now=NOW,
    )
    received_response = client_verifier.require_verified(
        response,
        _context(MCPMessageType.RESPONSE, "inventory-server", "inventory-client", digest),
        now=NOW,
    )
    assert received_response["result"] == {"available": True}


def test_tampered_replayed_or_unsigned_messages_never_reach_processing():
    definition = _definition()
    signer = MCPMessageSigner.generate(sender_id="inventory-client")
    verifier = MCPMessageVerifier((signer.trusted_key,))
    context = _context(
        MCPMessageType.REQUEST,
        "inventory-client",
        "inventory-server",
        definition.definition_digest,
    )
    envelope = signer.sign(
        {"jsonrpc": "2.0", "id": "request-2", "method": "tools/call"},
        message_type=MCPMessageType.REQUEST,
        recipient_id="inventory-server",
        user_id="warehouse-user",
        agent_id="inventory-agent",
        session_id="inventory-session",
        tool_definition_digest=definition.definition_digest,
        now=NOW,
    )
    processed: list[str] = []

    def process(candidate) -> None:
        verified = verifier.require_verified(candidate, context, now=NOW)
        processed.append(str(verified["id"]))

    tampered = envelope.model_copy(
        update={"payload": {**envelope.payload, "method": "admin/delete-all"}}, deep=True
    )
    try:
        process(tampered)
    except MCPMessageVerificationError as error:
        assert error.result.findings[0].code == MCPMessageVerificationCode.SIGNATURE_INVALID
    else:
        raise AssertionError("tampered request reached message processing")

    process(envelope)
    for rejected in (envelope, envelope.model_copy(update={"signature": None})):
        try:
            process(rejected)
        except MCPMessageVerificationError:
            pass
        else:
            raise AssertionError("replayed or unsigned request reached message processing")
    assert processed == ["request-2"]
