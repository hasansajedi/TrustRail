"""Bypass-oriented corpus for MCP tool poisoning and Unicode smuggling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustrail import MCPToolDefinition, MCPToolDefinitionCode, MCPToolDefinitionGuard

CORPUS_PATH = Path(__file__).parent.parent / "security_corpus" / "mcp_tool_definitions.json"
CASES: list[dict[str, str]] = json.loads(CORPUS_PATH.read_text())
KEY = b"security-corpus-mcp-signing-key-at-least-32-bytes"


def _definition(case: dict[str, str]) -> MCPToolDefinition:
    description = "Search reviewed public records by identifier."
    parameter_description = "Public record identifier selected by the user."
    result_description = "Matching public record identifiers."
    annotations: dict[str, object] = {"readOnlyHint": True}
    surface = case["surface"]
    if surface == "description":
        description = case["payload"]
    elif surface == "parameter_description":
        parameter_description = case["payload"]
    elif surface == "result_description":
        result_description = case["payload"]
    elif surface == "annotation":
        annotations["audience"] = case["payload"]
    return MCPToolDefinition(
        server_id="public-records",
        name="records.search",
        description=description,
        inputSchema={
            "type": "object",
            "properties": {
                "record_id": {
                    "type": "string",
                    "description": parameter_description,
                }
            },
            "required": ["record_id"],
            "additionalProperties": False,
        },
        outputSchema={
            "type": "object",
            "properties": {
                "record_ids": {
                    "type": "array",
                    "description": result_description,
                }
            },
            "required": ["record_ids"],
            "additionalProperties": False,
        },
        annotations=annotations,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_poisoned_definition_corpus_is_rejected_without_echoing_payload(case: dict[str, str]):
    definition = _definition(case)

    result = MCPToolDefinitionGuard(KEY).discover((definition,))

    codes = {finding.code for finding in result.findings}
    assert MCPToolDefinitionCode(case["expected_code"]) in codes
    assert case["payload"] not in result.model_dump_json()
