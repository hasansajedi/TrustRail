# MCP tool-definition integrity

MCP tool metadata is untrusted model input and a control-plane artifact. A host
must inspect and pin the complete definition before showing it to a model, bind
user or operator approval to exactly those bytes, and verify the live definition
again immediately before every execution. This prevents a server from earning
approval with a benign definition and replacing it later (a rug pull).

`MCPToolDefinitionGuard` implements this as a two-phase workflow:

1. `discover()` scans all definitions together and creates an authenticated
   discovery snapshot.
2. The application presents the reviewed definitions through a trusted consent
   UI and calls `approve()` with the unchanged discovery snapshot.
3. `verify_execution()` re-scans and re-hashes the live definition immediately
   before dispatch. A safe change returns `REQUIRE_APPROVAL`; a poisoned change
   is blocked.

## Complete mediation example

```python
import os

from trustrail import MCPToolDefinition, MCPToolDefinitionGuard

definition = MCPToolDefinition(
    # Derive server_id from the authenticated connection configuration, not
    # from tool metadata controlled by the MCP server.
    server_id="calendar-production",
    name="calendar.events.list",
    description="List events in an explicitly bounded UTC range.",
    inputSchema={
        "type": "object",
        "properties": {
            "start": {
                "type": "string",
                "format": "date-time",
                "description": "Inclusive UTC range start.",
            },
            "end": {
                "type": "string",
                "format": "date-time",
                "description": "Exclusive UTC range end.",
            },
        },
        "required": ["start", "end"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "event_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Opaque identifiers for matching events.",
            }
        },
        "required": ["event_ids"],
        "additionalProperties": False,
    },
    annotations={"readOnlyHint": True, "idempotentHint": True},
)

key = os.environ["TRUSTRAIL_MCP_PIN_KEY"].encode()
definition_guard = MCPToolDefinitionGuard(key)

# Persist both snapshots in application-owned protected storage.
discovery = definition_guard.require_discovery([definition])

# Only trusted UI/backend code may set approved_by after showing the exact
# reviewed definition. Never accept this value from the model or MCP server.
approval = definition_guard.require_approval(
    [definition],
    discovery,
    approved_by="operator-42",
)

# Fetch the live definition again as close to dispatch as possible.
live_definition = await mcp_client.get_tool_definition(definition.name)
definition_guard.require_execution(live_definition, approval)
result = await mcp_client.call_tool(live_definition.name, arguments)
```

Use separate keys per environment and load them from a secrets manager or OS
credential store. The HMAC key must contain at least 32 bytes. Rotate it through
an application-controlled migration that invalidates old snapshots and requires
new discovery and consent. Do not log it or place it in model context.

## What is bound and scanned

The canonical SHA-256 digest covers the application-assigned `server_id`, tool
name, title, description, complete input schema, complete output/result schema,
and annotations. Canonicalization sorts object keys, uses stable JSON encoding,
normalizes strings to Unicode NFC, rejects non-finite numbers, and rejects keys
that collide after normalization. The signed snapshot also binds its lifecycle
phase, timestamp, reviewer identity, and a digest for every JSON-pointer leaf.

Discovery scans every nested key and string value for:

- model-directed or hidden instructions;
- invisible, private-use, surrogate, tag, and bidirectional Unicode channels;
- references to tools exposed by another definition;
- duplicate and common Unicode-confusable tool names across servers;
- open-ended object schemas, missing property types, missing descriptions, and
  configured size or nesting-limit violations.

Mutation findings contain only hashed identity references, field paths, change
kinds, and before and after digests. They intentionally exclude names,
descriptions, schemas, and other potentially malicious content. A valid but changed definition returns
`GuardAction.REQUIRE_APPROVAL`; repeat discovery and approval rather than
overriding that result.

## Legacy schema configuration

Strict closed schemas are the default. A staged migration can temporarily relax
schema-shape checks while retaining scanning and cryptographic pinning:

```python
from trustrail import MCPToolDefinitionGuard, MCPToolDefinitionPolicy

policy = MCPToolDefinitionPolicy(
    require_closed_object_schemas=False,
    require_schema_descriptions=False,
    max_tools=100,
    max_nodes_per_definition=2_000,
    max_total_string_chars=50_000,
    max_depth=24,
)
definition_guard = MCPToolDefinitionGuard(key, policy)
```

Keep `reject_cross_tool_references=True` for multi-server deployments. Relaxed
schema policies increase ambiguity and should have a removal date.

## Security assumptions and residual risk

- The host authenticates the MCP server connection and assigns `server_id`; a
  digest does not establish server identity or publisher trust.
- The signing key, snapshots, reviewer identity, and consent UI are outside model
  and MCP-server control. HMAC provides integrity only while that shared secret
  remains protected.
- Complete mediation is an application responsibility. Any executor path that
  omits `require_execution()` can bypass pinning.
- Pattern and homoglyph detection are defense in depth, not proof of benign
  semantics. Novel, multilingual, encoded, or context-dependent poisoning may
  evade static scanning, while legitimate cross-tool documentation may require
  redesign rather than an allow-by-default exception.
- Pinning detects changed metadata; it does not prove that an unchanged tool
  implementation is safe or unchanged. Continue to enforce least-privilege
  credentials, service-side authorization, input/output validation, sandboxing,
  egress controls, transaction limits, monitoring, and human confirmation for
  sensitive effects.
- Process-local verification cannot guarantee that a remote server executes the
  same implementation described by the definition. Use authenticated transports,
  [signed MCP requests and responses](mcp-message-integrity.md), attestations where
  available, and independent runtime controls.

For hosts exposing several servers, also enforce independent ownership,
credentials, principals, and explicit labeled data flows with
[MCP server isolation](mcp-server-isolation.md).

See the
[OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html#2-tool-description-schema-integrity)
for the broader deployment guidance.

For hosts that connect more than one server, also enforce
[MCP server isolation and explicit cross-origin data flows](mcp-server-isolation.md).
