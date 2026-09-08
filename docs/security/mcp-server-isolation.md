# MCP server isolation and cross-origin protection

An MCP host that connects several servers creates a shared decision surface. A
malicious or compromised server can try to impersonate another server's tool,
place cross-server instructions in metadata or results, induce a confused-deputy
call, or move credentials and sensitive data across trust boundaries.

`MCPServerIsolationGateway` treats every server as an independent trust domain.
It fails closed unless the application declares the server, its unique tool
namespace, its exact tools, its credential binding, its permitted principals,
and every allowed cross-server data-flow edge.

The gateway provides:

- a closed inventory of server identities, namespaces, tools, and per-server
  credential fingerprints;
- joint definition inspection for unknown servers, namespace violations,
  duplicate or Unicode-confusable tool identities, and cross-server references;
- authenticated user, agent, tenant, scope, destination, initiator, session, and
  credential bindings at dispatch;
- source- and tool-bound result envelopes with explicit data labels and an
  integrity digest;
- explicit source-tool to destination-tool edges that allow, redact, or require
  approval for specific labels;
- hooks for trusted redaction, out-of-band approval verification, and audit
  persistence; and
- bounded, content-free findings and audit events.

## Define independent trust domains

Build the policy from authenticated application configuration. Do not accept
server IDs, ownership, credentials, principals, scopes, labels, or data-flow
edges from a model or an MCP server.

```python
import os

from trustrail import (
    MCPDataFlowEdge,
    MCPServerIsolationGateway,
    MCPServerIsolationPolicy,
    MCPServerTrustDomain,
    mcp_credential_reference,
)

search = MCPServerTrustDomain(
    server_id="search-production",
    tool_namespace="search",
    allowed_tools=frozenset({"search.query"}),
    # The policy retains only a fingerprint. Keep the actual credential in a
    # secrets manager and resolve it only after authorization.
    credential_ref=mcp_credential_reference(os.environ["SEARCH_MCP_TOKEN"]),
    allowed_tenant_ids=frozenset({"tenant-a"}),
    allowed_agent_ids=frozenset({"research-agent"}),
    allowed_user_ids=frozenset({"operator-42"}),
    required_scopes=frozenset({"search.read"}),
)

archive = MCPServerTrustDomain(
    server_id="archive-production",
    tool_namespace="archive",
    allowed_tools=frozenset({"archive.store"}),
    credential_ref=mcp_credential_reference(os.environ["ARCHIVE_MCP_TOKEN"]),
    allowed_tenant_ids=frozenset({"tenant-a"}),
    allowed_agent_ids=frozenset({"research-agent"}),
    allowed_user_ids=frozenset({"operator-42"}),
    required_scopes=frozenset({"archive.write"}),
)

policy = MCPServerIsolationPolicy(
    domains=(search, archive),
    data_flow_edges=(
        MCPDataFlowEdge(
            source_server_id="search-production",
            destination_server_id="archive-production",
            source_tools=frozenset({"search.query"}),
            destination_tools=frozenset({"archive.store"}),
            allowed_labels=frozenset({"public"}),
            redact_labels=frozenset({"confidential"}),
            approval_labels=frozenset({"restricted"}),
        ),
    ),
    max_definitions=100,
    max_inputs_per_request=16,
    max_nodes_per_request=10_000,
    max_string_chars_per_request=250_000,
)
gateway = MCPServerIsolationGateway(policy)
```

Server IDs and credential references must be unique. Tools must be qualified by
their owner's namespace, and cross-server edges must name tools owned by their
declared endpoints. Credentials can never appear in an allowed, redacted, or
approval-gated label set.

## Inspect definitions before model exposure

Pass the complete exposed inventory to `require_definitions()` before adding any
definition to model context:

```python
definitions = gateway.require_definitions(discovered_definitions)
```

The gateway checks ownership and the inventory as a whole. It raises
`MCPIsolationError` on an unknown server, an unowned tool, cross-server
references, or exact and common Unicode-confusable shadowing. Combine this with
`MCPToolDefinitionGuard` to scan complete schemas, bind consent, and detect
post-approval mutations.

## Label results and authorize every dispatch

Create result envelopes in the trusted adapter immediately after verifying the
source response. Labels are application policy facts, not server assertions.

```python
from trustrail import (
    MCPGatewayPrincipal,
    MCPGatewayRequest,
    MCPToolResultEnvelope,
)

search_result = MCPToolResultEnvelope.create(
    result_id="result-123",
    source_server_id="search-production",
    source_tool="search.query",
    labels=frozenset({"public"}),
    content=verified_search_payload,
)

request = MCPGatewayRequest(
    request_id="request-456",
    destination_server_id="archive-production",
    destination_tool="archive.store",
    initiating_server_id="search-production",
    initiating_tool="search.query",
    principal=MCPGatewayPrincipal(
        user_id=authenticated_user_id,
        agent_id=authenticated_agent_id,
        tenant_id=authenticated_tenant_id,
        scopes=frozenset(authenticated_scopes),
    ),
    session_id=active_session_id,
    credential_ref=archive.credential_ref,
    arguments={"retention_days": 7},
    inputs=(search_result,),
)

permit = gateway.require(request)

# Select the real credential by permit.destination_server_id in trusted code.
# Never copy credential material from arguments, results, or model output.
await dispatch_with_destination_credential(permit)
```

`require()` returns an `AuthorizedMCPGatewayRequest` only after complete
mediation. Its arguments and input content are deliberately excluded from normal
Pydantic serialization. The gateway rejects results with changed source, tool,
labels, content, or digest; undeclared routes or labels; credential-labeled or
known raw credential data crossing domains; and result text that instructs the
destination server or tool.

Set `initiating_server_id` and `initiating_tool` whenever a server result or
instruction caused a new call, even if no result object is forwarded. This
preserves causal provenance and ensures the explicit edge is checked.

## Redaction and approval hooks

Pass an `MCPGatewayRedactor` when an edge has `redact_labels`. The hook receives a
defensive copy. Its returned envelope must retain the original result ID, source
server, and source tool; remove the targeted labels; contain only labels allowed
by the edge; change the integrity digest; and remain internally valid. A missing,
failing, unchanged, or substituting redactor is denied.

For `approval_labels`, first call `authorize()` without an approval. It returns
`GuardAction.REQUIRE_APPROVAL`. A trusted UI or backend can then issue an
`MCPDataFlowApproval` containing the exact `request_digest`, exact required label
set, authenticated approver, and short expiration. Configure an
`MCPGatewayApprovalVerifier` that authenticates that record and submit it on a
second call. Approval IDs are single-use within one gateway process.

`StaticMCPGatewayApprovalVerifier` and `MemoryMCPIsolationAuditSink` are bounded
development and test helpers. Production systems should verify approvals from
protected application state and persist audit events in access-controlled,
durable storage.

## Audit events

Each decision produces `MCPIsolationAuditEvent`. Events record the operation,
action, stable finding codes, labels, redaction count, approval reference, and
hashed request, source, destination, user, agent, and session references. They do
not include arguments, result content, raw identities, or credentials. Hashes of
low-entropy identifiers may still be guessable, so normal log access and
retention controls remain necessary.

## Assumptions, limitations, and residual risk

- The host must authenticate each connection and assign the server ID. Namespace
  and digest checks do not establish publisher identity or implementation trust.
- Every discovery, model-exposure, orchestration, and dispatch path must pass
  through the gateway. Optional initiator fields distinguish direct user calls;
  trusted orchestration code must not omit them for server-caused calls.
- Labels, principal context, tool ownership, approval records, and credential
  selection must come from trusted application state. A dishonest label can
  turn a prohibited flow into an apparently public one.
- Cross-server instruction and homoglyph detection are defense in depth, not a
  complete semantic or multilingual classifier. Application-specific review and
  red-team evaluation remain necessary.
- The integrity digest detects accidental or in-process mutation but is not a
  remote signature. Verify signed responses first and protect envelopes from
  untrusted construction.
- Approval replay state is process-local. Multi-worker deployments need a shared
  atomic approval claim in the trusted verifier or surrounding gateway before
  dispatch; fail closed if that state is unavailable.
- The gateway authorizes data flow; it does not provide network, process, memory,
  or model-context isolation. Use separate server processes or containers,
  per-server model contexts where needed, outbound network policy, quotas, and
  operating-system sandboxing.
- Keep TLS, per-server credentials, downstream service authorization, schema and
  response validation, tool-definition pinning, signed messages, rate limits,
  monitoring, incident response, and human confirmation for sensitive effects.

See [MCP tool-definition integrity](mcp-tool-integrity.md),
[MCP message integrity](mcp-message-integrity.md), and
[Protect tool calls](../guides/protect-tools.md) for the surrounding boundaries.
This control follows the
[OWASP MCP multi-server isolation guidance](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html#8-multi-server-isolation-cross-origin-protection).
