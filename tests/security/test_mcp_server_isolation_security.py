"""Bypass-oriented corpus for MCP cross-origin isolation controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from trustrail import (
    MCPDataFlowEdge,
    MCPGatewayPrincipal,
    MCPGatewayRequest,
    MCPIsolationCode,
    MCPServerIsolationGateway,
    MCPServerIsolationPolicy,
    MCPServerTrustDomain,
    MCPToolDefinition,
    MCPToolResultEnvelope,
    mcp_credential_reference,
)

CORPUS_PATH = Path(__file__).parent.parent / "security_corpus" / "mcp_server_isolation.json"
CASES: list[dict[str, str]] = json.loads(CORPUS_PATH.read_text())
NOW = datetime(2026, 9, 8, 19, tzinfo=UTC)
SEARCH_CREDENTIAL = "search-security-secret"
VAULT_CREDENTIAL = "vault-security-secret"


def _domain(server_id: str, namespace: str, tool: str, credential: str):
    return MCPServerTrustDomain(
        server_id=server_id,
        tool_namespace=namespace,
        allowed_tools=frozenset({tool}),
        credential_ref=mcp_credential_reference(credential),
        allowed_tenant_ids=frozenset({"tenant-a"}),
        allowed_agent_ids=frozenset({"security-agent"}),
        allowed_user_ids=frozenset({"user-1"}),
        required_scopes=frozenset({"vault.write"}),
    )


def _gateway():
    search = _domain("search-server", "search", "search.query", SEARCH_CREDENTIAL)
    vault = _domain("vault-server", "vault", "vault.store", VAULT_CREDENTIAL)
    return MCPServerIsolationGateway(
        MCPServerIsolationPolicy(
            domains=(search, vault),
            data_flow_edges=(
                MCPDataFlowEdge(
                    source_server_id="search-server",
                    destination_server_id="vault-server",
                    source_tools=frozenset({"search.query"}),
                    destination_tools=frozenset({"vault.store"}),
                    allowed_labels=frozenset({"public"}),
                ),
            ),
        )
    )


def _result(*, labels=frozenset({"public"}), content="public result", tool="search.query"):
    return MCPToolResultEnvelope.create(
        result_id="corpus-result",
        source_server_id="search-server",
        source_tool=tool,
        labels=labels,
        content=content,
    )


def _request(**updates):
    values = {
        "request_id": "corpus-request",
        "destination_server_id": "vault-server",
        "destination_tool": "vault.store",
        "principal": MCPGatewayPrincipal(
            user_id="user-1",
            agent_id="security-agent",
            tenant_id="tenant-a",
            scopes=frozenset({"vault.write"}),
        ),
        "session_id": "security-session",
        "credential_ref": mcp_credential_reference(VAULT_CREDENTIAL),
        "arguments": {"record": "public summary"},
        "inputs": (_result(),),
    }
    values.update(updates)
    return MCPGatewayRequest(**values)


def _definition(server_id: str, name: str, description: str):
    return MCPToolDefinition(
        server_id=server_id,
        name=name,
        description=description,
        inputSchema={"type": "object", "additionalProperties": False},
        outputSchema={"type": "object", "additionalProperties": False},
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_cross_origin_bypass_corpus_is_rejected_without_echoing_payload(case):
    gateway = _gateway()
    mutation = case["mutation"]
    value = case["value"]

    if mutation == "definition_description":
        decision = gateway.inspect_definitions(
            (_definition("search-server", "search.query", value),), now=NOW
        )
    elif mutation == "definition_shadow":
        decision = gateway.inspect_definitions(
            (
                _definition("search-server", "search.query", "Search public records."),
                _definition("vault-server", value, "Store an authorized record."),
            ),
            now=NOW,
        )
    else:
        request = _request()
        if mutation == "destination_server":
            request = _request(destination_server_id=value)
        elif mutation == "destination_tool":
            request = _request(destination_tool=value)
        elif mutation == "principal_user":
            request = _request(principal=request.principal.model_copy(update={"user_id": value}))
        elif mutation == "credential_ref":
            request = _request(credential_ref=mcp_credential_reference(SEARCH_CREDENTIAL))
        elif mutation == "raw_credential":
            request = _request(arguments={"authorization": value})
        elif mutation == "result_label":
            request = _request(inputs=(_result(labels=frozenset({value})),))
        elif mutation == "result_content":
            request = _request(inputs=(_result(content=value),))
        elif mutation == "source_tool":
            request = _request(inputs=(_result(tool=value),))
        decision = gateway.authorize(request, now=NOW)

    codes = {finding.code for finding in decision.findings}
    assert MCPIsolationCode(case["expected_code"]) in codes
    if mutation != "result_label":
        assert value not in decision.model_dump_json()
