# MCP message integrity and replay protection

TLS protects an MCP connection in transit, but a compromised proxy, middleware
component, or host process can still alter or replay JSON-RPC messages after TLS
termination. trustrail can sign the complete request or response at the
application layer and reject messages that do not match authenticated local
state.

The message controls provide:

- Ed25519 signatures over one canonical envelope containing the complete
  JSON-RPC payload;
- out-of-band public-key pinning to an authenticated sender identity;
- exact recipient, user, agent, session, message-direction, and approved
  tool-definition bindings;
- timestamps, short expirations, cryptographic nonces, and atomic replay claims;
- fail-closed handling for missing signatures, unknown or inactive keys,
  invalid bindings, stale messages, and unavailable or full replay stores; and
- content-free verification events with hashed references instead of payloads
  or raw identities.

## Sign and verify a request

```python
from trustrail import (
    MCPMessageSigner,
    MCPMessageType,
    MCPMessageVerificationContext,
    MCPMessageVerifier,
    MemoryMCPReplayStore,
)

# Generate only for this example. Load the private key from protected key
# management in production and provision signer.trusted_key to the receiver
# through an authenticated channel.
signer = MCPMessageSigner.generate(sender_id="agent-client")
verifier = MCPMessageVerifier(
    (signer.trusted_key,),
    replay_store=MemoryMCPReplayStore(max_entries=10_000),
)

envelope = signer.sign(
    {
        "jsonrpc": "2.0",
        "id": "call-123",
        "method": "tools/call",
        "params": {"name": "calendar.events.list", "arguments": {}},
    },
    message_type=MCPMessageType.REQUEST,
    recipient_id="calendar-server",
    user_id=authenticated_user_id,
    agent_id=authenticated_agent_id,
    session_id=active_session_id,
    tool_definition_digest=approved_definition.definition_digest,
)

# Build expectations from authenticated application state, never from the
# untrusted envelope itself.
context = MCPMessageVerificationContext(
    message_type=MCPMessageType.REQUEST,
    sender_id="agent-client",
    recipient_id="calendar-server",
    user_id=authenticated_user_id,
    agent_id=authenticated_agent_id,
    session_id=active_session_id,
    tool_definition_digest=approved_definition.definition_digest,
)
payload = verifier.require_verified(envelope, context)
await process_json_rpc(payload)
```

`require_verified()` returns a defensive copy only after every check succeeds.
It raises `MCPMessageVerificationError` otherwise. `verify()` provides a typed
`MCPMessageVerificationResult` when the application needs to inspect or route a
denial without processing the payload.

Use `MCPMessageEnvelope.model_dump_json()` for transport and
`MCPMessageEnvelope.model_validate_json()` at the receiving boundary.

## Require mutual signing

Create a signer for each peer and configure each receiver with the other peer's
trusted public-key binding. Sign server results with
`MCPMessageType.RESPONSE`, then verify them against a response context before
adding tool output to model context. Do not accept an unsigned response after a
signed request, and do not learn a server key from that response.

The `key_id` in an envelope is the SHA-256 fingerprint of its raw Ed25519 public
key. `MCPTrustedKey` additionally binds that key to one sender and can carry
activation, expiration, and revocation state. Provision trusted keys over an
authenticated administrative channel. A key supplied by the message sender is
not a trust anchor.

## Replay storage

`MemoryMCPReplayStore` atomically claims a digest of the signing key and nonce.
It removes expired claims, has a hard capacity, and fails closed when full. It is
suitable for one-process deployments, development, and tests.

Production deployments with multiple workers or regions must implement the
`MCPReplayStore` protocol using a shared store. Its `claim()` operation must be
atomic across all message consumers and retain the claim through the complete
freshness window. Treat a backend outage or capacity failure as a denial, never
as permission to skip replay checks.

## Policy and audit signals

`MCPMessageSigningPolicy` bounds the default and maximum TTL, maximum message
age, accepted clock skew, and canonical envelope size. Keep the window short and
synchronize peer clocks. The defaults accept a maximum five-minute message age,
a five-minute TTL, 30 seconds of clock skew, and a 1 MiB envelope.

Pass an `MCPMessageAuditSink` to the verifier to persist each outcome.
`MemoryMCPMessageAuditSink` is a bounded development implementation. Events
contain the action, outcome code, direction, timestamp, and hashed message, key,
sender, recipient, user, agent, session, tool-definition, and nonce references.
They intentionally exclude the JSON-RPC payload, raw identifiers, signatures,
and raw nonces.

## Canonicalization profile

trustrail's signing profile normalizes JSON object keys and string values to NFC,
rejects normalized-key collisions and non-finite numbers, sorts object keys,
uses compact separators, preserves Unicode as UTF-8, and covers every envelope
field except `signature`. This is a documented trustrail profile, not a claim of
full RFC 8785 conformance. Non-Python peers must reproduce the exact bytes and
should share cross-language conformance vectors before deployment.

## Assumptions, limitations, and residual risk

- Continue using TLS and normal MCP authentication and authorization. Message
  signing does not provide confidentiality or decide whether a tool call is
  permitted.
- Resolve user, agent, peer, recipient, session, and definition expectations
  from trusted local state. Copying them from the envelope defeats identity and
  audience binding.
- Protect private keys in a KMS, HSM, or OS credential store; authenticate key
  provisioning; rotate keys deliberately; and distribute revocation promptly.
- A valid signature proves possession of a configured key, not that the sender,
  tool implementation, payload semantics, host, or endpoint is safe.
- The in-memory replay store is not shared or durable. A restart loses claims,
  and separate workers can each accept the same nonce unless they use a shared
  atomic implementation.
- Capacity limits fail closed and can affect availability during a flood. Apply
  authenticated rate limits before expensive verification and monitor replay
  store utilization.
- Hashed audit references avoid direct content retention but low-entropy
  identifiers may still be guessable. Apply normal access control and retention
  limits to security logs.
- Signatures do not replace tool-definition scanning, pre-execution pin checks,
  runtime argument validation, least-privilege authorization, output sanitizing,
  service-side permissions, or human confirmation for sensitive effects.

See [MCP tool-definition integrity](mcp-tool-integrity.md) for discovery,
approval, and rug-pull protection,
[MCP server isolation](mcp-server-isolation.md) for cross-origin data-flow and
credential boundaries, and [Protect tool calls](../guides/protect-tools.md) for
the surrounding execution boundary.

When the host connects multiple servers, apply
[MCP server isolation](mcp-server-isolation.md) before dispatch; message
signatures authenticate an envelope but do not authorize cross-origin data flow.

The broader deployment guidance is in the
[OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html#7-message-level-integrity-and-replay-protection).
