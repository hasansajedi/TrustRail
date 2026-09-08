"""Fail-closed MCP server isolation and cross-origin gateway controls."""

from __future__ import annotations

import contextlib
import copy
import re
import threading
import unicodedata
from collections import deque
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Literal, Protocol

from pydantic import JsonValue

from trustrail.exceptions import MCPIsolationError
from trustrail.models.enums import GuardAction, Severity
from trustrail.models.mcp import MCPToolDefinition
from trustrail.models.mcp_isolation import (
    AuthorizedMCPGatewayRequest,
    MCPDataFlowApproval,
    MCPDataFlowEdge,
    MCPDataLabel,
    MCPGatewayPrincipal,
    MCPGatewayRequest,
    MCPIsolationAuditEvent,
    MCPIsolationCode,
    MCPIsolationFinding,
    MCPIsolationOperation,
    MCPIsolationResult,
    MCPServerIsolationPolicy,
    MCPServerTrustDomain,
    MCPToolResultEnvelope,
    _digest,
    utcnow,
)
from trustrail.models.mcp_messages import content_reference

_INSTRUCTION_RE = re.compile(
    r"\b(?:call|invoke|use|send|forward|pass|upload|exfiltrate|ignore|override|tell|ask)\b",
    re.IGNORECASE,
)
_CREDENTIAL_CANDIDATE_RE = re.compile(r"[A-Za-z0-9_./+=:-]{8,}")
_CONFUSABLES = str.maketrans(
    {
        "\u0430": "a",
        "\u0410": "a",
        "\u03b1": "a",
        "\u0391": "a",
        "\u0435": "e",
        "\u0415": "e",
        "\u03b5": "e",
        "\u0395": "e",
        "\u0456": "i",
        "\u0406": "i",
        "\u03b9": "i",
        "\u0399": "i",
        "\u043e": "o",
        "\u041e": "o",
        "\u03bf": "o",
        "\u039f": "o",
        "\u0440": "p",
        "\u0420": "p",
        "\u03c1": "p",
        "\u03a1": "p",
        "\u0441": "c",
        "\u0421": "c",
        "\u03f2": "c",
        "\u0445": "x",
        "\u0425": "x",
        "\u03c7": "x",
        "\u03a7": "x",
        "\u0443": "y",
        "\u04ae": "y",
    }
)


class MCPGatewayRedactor(Protocol):
    """Transform one labeled result before it crosses a server boundary."""

    def redact(
        self,
        result: MCPToolResultEnvelope,
        *,
        labels: frozenset[str],
        destination_server_id: str,
        destination_tool: str,
    ) -> MCPToolResultEnvelope:
        """Return a new integrity-bound result with sensitive labels removed."""
        ...


class MCPGatewayApprovalVerifier(Protocol):
    """Authenticate an out-of-band cross-server data-flow approval."""

    def verify_approval(self, approval: MCPDataFlowApproval) -> bool:
        """Return whether trusted application state issued this exact approval."""
        ...


class MCPIsolationAuditSink(Protocol):
    """Persist content-free MCP isolation events."""

    def emit(self, event: MCPIsolationAuditEvent) -> None:
        """Persist one event without inspecting arguments or tool-result content."""
        ...


class MemoryMCPIsolationAuditSink:
    """Thread-safe bounded audit sink for tests and development."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[MCPIsolationAuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, event: MCPIsolationAuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[MCPIsolationAuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class StaticMCPGatewayApprovalVerifier:
    """Exact-match approval verifier for tests and protected application state."""

    def __init__(self, approvals: Iterable[MCPDataFlowApproval]) -> None:
        self._approvals = tuple(approval.model_copy(deep=True) for approval in approvals)

    def verify_approval(self, approval: MCPDataFlowApproval) -> bool:
        return any(approval == expected for expected in self._approvals)


def _skeleton(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).translate(_CONFUSABLES).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() and not unicodedata.combining(character)
    )


def _walk(value: object, depth: int = 0) -> Iterator[tuple[object, int]]:
    yield value, depth
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, depth + 1
            yield from _walk(item, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _walk(item, depth + 1)


def _strings(value: object) -> Iterator[str]:
    for item, _ in _walk(value):
        if isinstance(item, str):
            yield item


def _credential_candidates(value: str) -> Iterator[str]:
    stripped = value.strip()
    if stripped:
        yield stripped
    if stripped.casefold().startswith("bearer "):
        yield stripped[7:].strip()
    yield from _CREDENTIAL_CANDIDATE_RE.findall(value)


class MCPServerIsolationGateway:
    """Completely mediate tool discovery and calls across MCP trust domains.

    All domain, principal, credential, and label facts must come from trusted
    application state. Tool arguments and result content are excluded from
    findings and audit events.
    """

    def __init__(
        self,
        policy: MCPServerIsolationPolicy,
        *,
        redactor: MCPGatewayRedactor | None = None,
        approval_verifier: MCPGatewayApprovalVerifier | None = None,
        audit_sink: MCPIsolationAuditSink | None = None,
    ) -> None:
        self._policy = policy.model_copy(deep=True)
        self._domains = {domain.server_id: domain for domain in self._policy.domains}
        self._redactor = redactor
        self._approval_verifier = approval_verifier
        self._audit_sink = audit_sink
        self._used_approval_ids: set[str] = set()
        self._lock = threading.Lock()

    @property
    def policy(self) -> MCPServerIsolationPolicy:
        return self._policy.model_copy(deep=True)

    def inspect_definitions(
        self,
        definitions: Iterable[MCPToolDefinition],
        *,
        now: datetime | None = None,
    ) -> MCPIsolationResult:
        """Reject cross-origin references, shadowing, and namespace violations."""
        items = tuple(definition.model_copy(deep=True) for definition in definitions)
        findings: list[MCPIsolationFinding] = []
        if not items or len(items) > self._policy.max_definitions:
            findings.append(
                self._finding(
                    MCPIsolationCode.RESOURCE_LIMIT_EXCEEDED,
                    "MCP definition inventory is empty or exceeds its configured limit",
                )
            )

        by_name: dict[str, list[MCPToolDefinition]] = {}
        by_skeleton: dict[str, list[MCPToolDefinition]] = {}
        for definition in items:
            by_name.setdefault(
                unicodedata.normalize("NFKC", definition.name).casefold(), []
            ).append(definition)
            by_skeleton.setdefault(_skeleton(definition.name), []).append(definition)
            domain = self._domains.get(definition.server_id)
            if domain is None:
                findings.append(
                    self._finding(
                        MCPIsolationCode.UNKNOWN_SERVER,
                        "MCP definition belongs to an undeclared server trust domain",
                        source=definition.server_id,
                    )
                )
                continue
            if not self._owns_tool(domain, definition.name):
                findings.append(
                    self._finding(
                        MCPIsolationCode.TOOL_ROUTE_MISMATCH,
                        "MCP definition attempts to use a tool name outside its owned namespace",
                        source=definition.server_id,
                    )
                )
            findings.extend(self._cross_definition_findings(definition))

        for group in (*by_name.values(), *by_skeleton.values()):
            servers = {definition.server_id for definition in group}
            if len(group) > 1 and len(servers) > 1:
                findings.append(
                    self._finding(
                        MCPIsolationCode.TOOL_SHADOWING,
                        "MCP definitions from separate trust domains have colliding tool "
                        "identities",
                        source="|".join(sorted(servers)),
                    )
                )

        findings = self._deduplicate(findings)
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK] = (
            GuardAction.BLOCK if findings else GuardAction.ALLOW
        )
        return self._result(
            action=action,
            operation=MCPIsolationOperation.DEFINITION_INSPECTION,
            findings=findings,
            now=now or utcnow(),
        )

    def require_definitions(
        self,
        definitions: Iterable[MCPToolDefinition],
        *,
        now: datetime | None = None,
    ) -> tuple[MCPToolDefinition, ...]:
        """Return defensive definitions or raise before model exposure."""
        items = tuple(definition.model_copy(deep=True) for definition in definitions)
        result = self.inspect_definitions(items, now=now)
        if not result.is_allowed:
            raise MCPIsolationError(result)
        return items

    def authorize(
        self,
        request: MCPGatewayRequest,
        *,
        approval: MCPDataFlowApproval | None = None,
        now: datetime | None = None,
    ) -> MCPIsolationResult:
        """Authorize one exact call and transform policy-redacted cross-server inputs."""
        current_time = now or utcnow()
        snapshot = request.model_copy(deep=True)
        findings = self._request_findings(snapshot)
        safe_inputs: list[MCPToolResultEnvelope] = []
        redacted_ids: list[str] = []
        approval_labels: set[str] = set()

        destination = self._domains.get(snapshot.destination_server_id)
        if destination is not None and not findings:
            for item in snapshot.inputs:
                transformed, item_findings, required_labels, redacted = self._mediate_input(
                    item,
                    snapshot,
                    destination,
                )
                findings.extend(item_findings)
                approval_labels.update(required_labels)
                if transformed is not None:
                    safe_inputs.append(transformed)
                if redacted:
                    redacted_ids.append(item.result_id)

        findings = self._deduplicate(findings)
        if findings:
            return self._result(
                action=GuardAction.BLOCK,
                operation=MCPIsolationOperation.INVOCATION,
                findings=findings,
                request=snapshot,
                redaction_count=len(redacted_ids),
                now=current_time,
            )

        if approval_labels:
            approval_finding = self._approval_finding(
                snapshot,
                approval,
                frozenset(approval_labels),
                current_time,
            )
            if approval_finding is not None:
                action: Literal[GuardAction.BLOCK, GuardAction.REQUIRE_APPROVAL] = (
                    GuardAction.REQUIRE_APPROVAL
                    if approval_finding.code == MCPIsolationCode.APPROVAL_REQUIRED
                    else GuardAction.BLOCK
                )
                return self._result(
                    action=action,
                    operation=MCPIsolationOperation.INVOCATION,
                    findings=[approval_finding],
                    request=snapshot,
                    redaction_count=len(redacted_ids),
                    approval=approval,
                    now=current_time,
                )

        authorization = AuthorizedMCPGatewayRequest(
            authorization_id=_digest(
                {
                    "approval_id": approval.approval_id if approval is not None else None,
                    "input_digests": [item.result_digest for item in safe_inputs],
                    "request_digest": snapshot.request_digest,
                }
            ),
            request_digest=snapshot.request_digest,
            destination_server_id=snapshot.destination_server_id,
            destination_tool=snapshot.destination_tool,
            principal=snapshot.principal,
            session_id=snapshot.session_id,
            arguments=copy.deepcopy(snapshot.arguments),
            inputs=tuple(item.model_copy(deep=True) for item in safe_inputs),
            redacted_result_ids=tuple(redacted_ids),
            approval_id=approval.approval_id if approval_labels and approval is not None else None,
        )
        return self._result(
            action=GuardAction.ALLOW,
            operation=MCPIsolationOperation.INVOCATION,
            findings=[],
            request=snapshot,
            authorization=authorization,
            redaction_count=len(redacted_ids),
            approval=approval if approval_labels else None,
            now=current_time,
        )

    def require(
        self,
        request: MCPGatewayRequest,
        *,
        approval: MCPDataFlowApproval | None = None,
        now: datetime | None = None,
    ) -> AuthorizedMCPGatewayRequest:
        """Return a safe dispatch permit or raise before invoking an MCP server."""
        result = self.authorize(request, approval=approval, now=now)
        if not result.is_authorized or result.authorization is None:
            raise MCPIsolationError(result)
        return result.authorization

    def _request_findings(self, request: MCPGatewayRequest) -> list[MCPIsolationFinding]:
        findings: list[MCPIsolationFinding] = []
        destination = self._domains.get(request.destination_server_id)
        if destination is None:
            return [
                self._finding(
                    MCPIsolationCode.UNKNOWN_SERVER,
                    "MCP call targets an undeclared server trust domain",
                    destination=request.destination_server_id,
                )
            ]
        if not self._owns_tool(destination, request.destination_tool):
            findings.append(
                self._finding(
                    MCPIsolationCode.TOOL_ROUTE_MISMATCH,
                    "MCP call targets a tool not owned by the destination server",
                    destination=request.destination_server_id,
                )
            )
        if not self._principal_allowed(destination, request.principal):
            findings.append(
                self._finding(
                    MCPIsolationCode.CONFUSED_DEPUTY,
                    "Authenticated principal context is not authorized for the destination server",
                    destination=request.destination_server_id,
                )
            )
        if request.credential_ref != destination.credential_ref:
            findings.append(
                self._finding(
                    MCPIsolationCode.CREDENTIAL_CROSSOVER,
                    "MCP call selected credentials bound to another trust domain",
                    destination=request.destination_server_id,
                )
            )
        if len(request.inputs) > self._policy.max_inputs_per_request:
            findings.append(
                self._finding(
                    MCPIsolationCode.RESOURCE_LIMIT_EXCEEDED,
                    "MCP call contains too many labeled inputs",
                    destination=request.destination_server_id,
                )
            )
        if self._resource_limit_exceeded(request):
            findings.append(
                self._finding(
                    MCPIsolationCode.RESOURCE_LIMIT_EXCEEDED,
                    "MCP call exceeds configured inspection limits",
                    destination=request.destination_server_id,
                )
            )
        if self._contains_credential(request):
            findings.append(
                self._finding(
                    MCPIsolationCode.CREDENTIAL_CROSSOVER,
                    "MCP arguments or results contain a server credential",
                    destination=request.destination_server_id,
                )
            )

        if request.initiating_server_id is not None:
            initiating = self._domains.get(request.initiating_server_id)
            if initiating is None:
                findings.append(
                    self._finding(
                        MCPIsolationCode.UNKNOWN_SERVER,
                        "MCP call claims an undeclared initiating server",
                        source=request.initiating_server_id,
                        destination=request.destination_server_id,
                    )
                )
            elif request.initiating_tool is not None and not self._owns_tool(
                initiating, request.initiating_tool
            ):
                findings.append(
                    self._finding(
                        MCPIsolationCode.TOOL_ROUTE_MISMATCH,
                        "Initiating tool is not owned by its claimed server",
                        source=request.initiating_server_id,
                        destination=request.destination_server_id,
                    )
                )
            elif request.initiating_server_id != request.destination_server_id and not any(
                self._edge_matches_initiator(edge, request) for edge in self._policy.data_flow_edges
            ):
                findings.append(
                    self._finding(
                        MCPIsolationCode.DATA_FLOW_DENIED,
                        "Initiating server has no explicit route to the destination tool",
                        source=request.initiating_server_id,
                        destination=request.destination_server_id,
                    )
                )
        return findings

    def _mediate_input(
        self,
        result: MCPToolResultEnvelope,
        request: MCPGatewayRequest,
        destination: MCPServerTrustDomain,
    ) -> tuple[
        MCPToolResultEnvelope | None,
        list[MCPIsolationFinding],
        frozenset[str],
        bool,
    ]:
        findings: list[MCPIsolationFinding] = []
        source = self._domains.get(result.source_server_id)
        if source is None:
            return (
                None,
                [
                    self._finding(
                        MCPIsolationCode.UNKNOWN_SERVER,
                        "Labeled input came from an undeclared MCP server",
                        source=result.source_server_id,
                        destination=destination.server_id,
                    )
                ],
                frozenset(),
                False,
            )
        if not result.has_valid_integrity:
            findings.append(
                self._finding(
                    MCPIsolationCode.RESULT_INTEGRITY_INVALID,
                    "MCP result source, labels, or content changed after labeling",
                    source=result.source_server_id,
                    destination=destination.server_id,
                )
            )
        if not self._owns_tool(source, result.source_tool):
            findings.append(
                self._finding(
                    MCPIsolationCode.TOOL_ROUTE_MISMATCH,
                    "MCP result names a tool not owned by its source server",
                    source=result.source_server_id,
                    destination=destination.server_id,
                )
            )
        if source.server_id == destination.server_id:
            return result.model_copy(deep=True), findings, frozenset(), False

        if MCPDataLabel.CREDENTIAL.value in result.labels:
            findings.append(
                self._finding(
                    MCPIsolationCode.CREDENTIAL_CROSSOVER,
                    "Credential-labeled data cannot cross MCP server trust domains",
                    source=result.source_server_id,
                    destination=destination.server_id,
                    labels=result.labels,
                )
            )

        edge = next(
            (
                candidate
                for candidate in self._policy.data_flow_edges
                if self._edge_matches_result(candidate, result, request)
            ),
            None,
        )
        if edge is None:
            findings.append(
                self._finding(
                    MCPIsolationCode.DATA_FLOW_DENIED,
                    "No explicit cross-server edge permits this labeled tool result",
                    source=result.source_server_id,
                    destination=destination.server_id,
                    labels=result.labels,
                )
            )
            return None, findings, frozenset(), False
        if self._contains_cross_server_instruction(result.content, destination, request):
            findings.append(
                self._finding(
                    MCPIsolationCode.CROSS_SERVER_INSTRUCTION,
                    "MCP result contains an instruction targeting another server's tool",
                    source=result.source_server_id,
                    destination=destination.server_id,
                )
            )

        known_labels = edge.allowed_labels | edge.redact_labels | edge.approval_labels
        undeclared = result.labels - known_labels
        if undeclared:
            findings.append(
                self._finding(
                    MCPIsolationCode.DATA_FLOW_DENIED,
                    "MCP result has labels not permitted by the cross-server edge",
                    source=result.source_server_id,
                    destination=destination.server_id,
                    labels=undeclared,
                )
            )
        redact = result.labels & edge.redact_labels
        require_approval = result.labels & edge.approval_labels
        if findings:
            return None, findings, frozenset(require_approval), False
        if not redact:
            return result.model_copy(deep=True), [], frozenset(require_approval), False
        if self._redactor is None:
            return (
                None,
                [
                    self._finding(
                        MCPIsolationCode.REDACTION_REQUIRED,
                        "Cross-server data requires a configured redaction hook",
                        source=result.source_server_id,
                        destination=destination.server_id,
                        labels=redact,
                    )
                ],
                frozenset(require_approval),
                False,
            )
        try:
            transformed = self._redactor.redact(
                result.model_copy(deep=True),
                labels=frozenset(redact),
                destination_server_id=destination.server_id,
                destination_tool=request.destination_tool,
            )
        except Exception:
            transformed = None
        if not self._valid_redaction(result, transformed, edge, redact):
            return (
                None,
                [
                    self._finding(
                        MCPIsolationCode.REDACTION_FAILED,
                        "MCP redaction hook failed or returned an unsafe result",
                        source=result.source_server_id,
                        destination=destination.server_id,
                        labels=redact,
                    )
                ],
                frozenset(require_approval),
                False,
            )
        return transformed, [], frozenset(require_approval), True

    def _approval_finding(
        self,
        request: MCPGatewayRequest,
        approval: MCPDataFlowApproval | None,
        labels: frozenset[str],
        now: datetime,
    ) -> MCPIsolationFinding | None:
        if approval is None:
            return self._finding(
                MCPIsolationCode.APPROVAL_REQUIRED,
                "Cross-server data labels require an out-of-band approval",
                destination=request.destination_server_id,
                labels=labels,
            )
        if (
            approval.request_digest != request.request_digest
            or approval.approved_labels != labels
            or approval.expires_at <= now
            or self._approval_verifier is None
        ):
            return self._finding(
                MCPIsolationCode.APPROVAL_INVALID,
                "Cross-server data-flow approval is invalid, expired, or incorrectly bound",
                destination=request.destination_server_id,
                labels=labels,
            )
        try:
            verified = self._approval_verifier.verify_approval(approval)
        except Exception:
            verified = False
        if not verified:
            return self._finding(
                MCPIsolationCode.APPROVAL_INVALID,
                "Cross-server data-flow approval could not be authenticated",
                destination=request.destination_server_id,
                labels=labels,
            )
        with self._lock:
            if approval.approval_id in self._used_approval_ids:
                return self._finding(
                    MCPIsolationCode.APPROVAL_REPLAYED,
                    "Cross-server data-flow approval has already been used",
                    destination=request.destination_server_id,
                    labels=labels,
                )
            self._used_approval_ids.add(approval.approval_id)
        return None

    def _cross_definition_findings(
        self,
        definition: MCPToolDefinition,
    ) -> list[MCPIsolationFinding]:
        other_markers: set[str] = set()
        for domain in self._policy.domains:
            if domain.server_id == definition.server_id:
                continue
            other_markers.add(unicodedata.normalize("NFKC", domain.server_id).casefold())
            other_markers.add(f"{domain.tool_namespace.casefold()}.")
            other_markers.update(tool.casefold() for tool in domain.allowed_tools)
        for text in _strings(definition.canonical_payload):
            normalized = unicodedata.normalize("NFKC", text).casefold()
            if any(marker in normalized for marker in other_markers):
                return [
                    self._finding(
                        MCPIsolationCode.CROSS_SERVER_INSTRUCTION,
                        "MCP definition references another server trust domain or its tools",
                        source=definition.server_id,
                    )
                ]
        return []

    def _contains_cross_server_instruction(
        self,
        content: JsonValue,
        destination: MCPServerTrustDomain,
        request: MCPGatewayRequest,
    ) -> bool:
        markers = (
            destination.server_id.casefold(),
            f"{destination.tool_namespace.casefold()}.",
            request.destination_tool.casefold(),
        )
        for text in _strings(content):
            normalized = unicodedata.normalize("NFKC", text).casefold()
            if _INSTRUCTION_RE.search(normalized) and any(
                marker in normalized for marker in markers
            ):
                return True
        return False

    def _contains_credential(self, request: MCPGatewayRequest) -> bool:
        configured = {domain.credential_ref for domain in self._policy.domains}
        values: list[object] = [request.arguments]
        values.extend(result.content for result in request.inputs)
        for value in values:
            for text in _strings(value):
                if any(
                    content_reference(candidate) in configured
                    for candidate in _credential_candidates(text)
                ):
                    return True
        return False

    def _resource_limit_exceeded(self, request: MCPGatewayRequest) -> bool:
        nodes = 0
        string_chars = 0
        values: list[object] = [request.arguments]
        values.extend(result.content for result in request.inputs)
        for value in values:
            for item, _ in _walk(value):
                nodes += 1
                if isinstance(item, str):
                    string_chars += len(item)
                if (
                    nodes > self._policy.max_nodes_per_request
                    or string_chars > self._policy.max_string_chars_per_request
                ):
                    return True
        return False

    @staticmethod
    def _principal_allowed(domain: MCPServerTrustDomain, principal: MCPGatewayPrincipal) -> bool:
        user_allowed = not domain.allowed_user_ids or principal.user_id in domain.allowed_user_ids
        return (
            principal.tenant_id in domain.allowed_tenant_ids
            and principal.agent_id in domain.allowed_agent_ids
            and user_allowed
            and domain.required_scopes.issubset(principal.scopes)
        )

    @staticmethod
    def _owns_tool(domain: MCPServerTrustDomain, tool: str) -> bool:
        return tool.startswith(f"{domain.tool_namespace}.") and tool in domain.allowed_tools

    @staticmethod
    def _edge_matches_initiator(edge: MCPDataFlowEdge, request: MCPGatewayRequest) -> bool:
        return (
            edge.source_server_id == request.initiating_server_id
            and edge.destination_server_id == request.destination_server_id
            and request.initiating_tool in edge.source_tools
            and request.destination_tool in edge.destination_tools
        )

    @staticmethod
    def _edge_matches_result(
        edge: MCPDataFlowEdge,
        result: MCPToolResultEnvelope,
        request: MCPGatewayRequest,
    ) -> bool:
        return (
            edge.source_server_id == result.source_server_id
            and edge.destination_server_id == request.destination_server_id
            and result.source_tool in edge.source_tools
            and request.destination_tool in edge.destination_tools
        )

    @staticmethod
    def _valid_redaction(
        original: MCPToolResultEnvelope,
        transformed: MCPToolResultEnvelope | None,
        edge: MCPDataFlowEdge,
        redacted_labels: set[str] | frozenset[str],
    ) -> bool:
        if transformed is None:
            return False
        return (
            transformed.has_valid_integrity
            and transformed.result_id == original.result_id
            and transformed.source_server_id == original.source_server_id
            and transformed.source_tool == original.source_tool
            and transformed.result_digest != original.result_digest
            and not (transformed.labels & redacted_labels)
            and transformed.labels.issubset(edge.allowed_labels)
            and MCPDataLabel.CREDENTIAL.value not in transformed.labels
        )

    def _result(
        self,
        *,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK, GuardAction.REQUIRE_APPROVAL],
        operation: MCPIsolationOperation,
        findings: list[MCPIsolationFinding],
        now: datetime,
        request: MCPGatewayRequest | None = None,
        authorization: AuthorizedMCPGatewayRequest | None = None,
        redaction_count: int = 0,
        approval: MCPDataFlowApproval | None = None,
    ) -> MCPIsolationResult:
        sources = sorted({item.source_server_id for item in request.inputs}) if request else []
        labels = (
            sorted({label for item in request.inputs for label in item.labels}) if request else []
        )
        event = MCPIsolationAuditEvent(
            occurred_at=now,
            operation=operation,
            action=action,
            finding_codes=tuple(finding.code for finding in findings)
            or (MCPIsolationCode.ALLOWED,),
            request_ref=(content_reference(request.request_digest) if request else None),
            source_refs=tuple(content_reference(source) for source in sources),
            destination_ref=(
                content_reference(request.destination_server_id) if request is not None else None
            ),
            user_ref=(
                content_reference(request.principal.user_id) if request is not None else None
            ),
            agent_ref=(
                content_reference(request.principal.agent_id) if request is not None else None
            ),
            session_ref=(content_reference(request.session_id) if request is not None else None),
            labels=tuple(labels),
            redaction_count=redaction_count,
            approval_ref=(
                content_reference(approval.approval_id) if approval is not None else None
            ),
        )
        if self._audit_sink is not None:
            with contextlib.suppress(Exception):
                self._audit_sink.emit(event)
        return MCPIsolationResult(
            action=action,
            findings=tuple(findings),
            authorization=authorization,
            audit_event=event,
        )

    @staticmethod
    def _finding(
        code: MCPIsolationCode,
        message: str,
        *,
        source: str | None = None,
        destination: str | None = None,
        labels: Iterable[str] = (),
    ) -> MCPIsolationFinding:
        return MCPIsolationFinding(
            code=code,
            severity=Severity.CRITICAL,
            message=message,
            source_ref=content_reference(source) if source is not None else None,
            destination_ref=(content_reference(destination) if destination is not None else None),
            labels=tuple(sorted(labels)),
        )

    @staticmethod
    def _deduplicate(findings: list[MCPIsolationFinding]) -> list[MCPIsolationFinding]:
        unique: list[MCPIsolationFinding] = []
        seen: set[tuple[MCPIsolationCode, str | None, str | None, tuple[str, ...]]] = set()
        for finding in findings:
            key = (finding.code, finding.source_ref, finding.destination_ref, finding.labels)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        return unique
