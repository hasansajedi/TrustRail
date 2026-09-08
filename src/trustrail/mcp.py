"""MCP tool-definition scanning, signing, approval, and execution pinning."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import datetime

from pydantic import JsonValue

from trustrail.exceptions import MCPToolDefinitionError
from trustrail.models.enums import GuardAction, Severity
from trustrail.models.mcp import (
    MCPDefinitionChange,
    MCPDefinitionChangeKind,
    MCPToolDefinition,
    MCPToolDefinitionBundle,
    MCPToolDefinitionCode,
    MCPToolDefinitionFinding,
    MCPToolDefinitionPhase,
    MCPToolDefinitionPolicy,
    MCPToolDefinitionResult,
    MCPToolFieldFingerprint,
    bundle_signing_payload,
    fingerprint_definition,
    utcnow,
)

_INSTRUCTION_PATTERNS = (
    re.compile(
        r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|other)\s+instructions?\b",
        re.I,
    ),
    re.compile(r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b", re.I),
    re.compile(r"\b(?:you|assistant|model)\s+must\b", re.I),
    re.compile(r"\b(?:do\s+not|never)\s+(?:tell|show|reveal|mention|disclose)\b", re.I),
    re.compile(r"\b(?:secretly|covertly|without\s+(?:the\s+)?user)\b", re.I),
    re.compile(r"<\s*/?\s*(?:instructions?|system|assistant|developer)\b", re.I),
)
_BIDI_CONTROL_CLASSES = {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_$#.-]{1,128}$")
_CONFUSABLES = str.maketrans(
    {
        # Common Cyrillic and Greek homoglyphs used in identifier spoofing.
        "\u0430": "a",
        "\u0251": "a",
        "\u0391": "a",
        "\u0410": "a",
        "\u0392": "b",
        "\u0412": "b",
        "\u03f2": "c",
        "\u0421": "c",
        "\u0441": "c",
        "\u0395": "e",
        "\u0415": "e",
        "\u0435": "e",
        "\u0397": "h",
        "\u041d": "h",
        "\u0399": "i",
        "\u0406": "i",
        "\u0456": "i",
        "\u039a": "k",
        "\u041a": "k",
        "\u039c": "m",
        "\u041c": "m",
        "\u039d": "n",
        "\u041e": "o",
        "\u03bf": "o",
        "\u043e": "o",
        "\u03a1": "p",
        "\u0420": "p",
        "\u0440": "p",
        "\u0405": "s",
        "\u0455": "s",
        "\u03a4": "t",
        "\u0422": "t",
        "\u03a7": "x",
        "\u0425": "x",
        "\u0445": "x",
        "\u03a5": "y",
        "\u04ae": "y",
        "\u0443": "y",
    }
)


def _walk(value: object, path: str = "", depth: int = 0) -> Iterator[tuple[str, object, int]]:
    yield path or "/", value, depth
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            child_path = f"{path}/{escaped}"
            yield f"{child_path}/#key", key, depth + 1
            yield from _walk(item, child_path, depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}/{index}", depth + 1)


def _identity_skeleton(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).translate(_CONFUSABLES).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() and not unicodedata.combining(character)
    )


def _safe_identity_reference(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value).encode()
    return f"sha256:{hashlib.sha256(normalized).hexdigest()}"


def _safe_field_path(value: str) -> str:
    safe_segments: list[str] = []
    for segment in value.split("/")[1:]:
        if _SAFE_PATH_SEGMENT_RE.fullmatch(segment):
            safe_segments.append(segment)
        else:
            digest = hashlib.sha256(segment.encode()).hexdigest()[:16]
            safe_segments.append(f"sha256-{digest}")
    return f"/{'/'.join(safe_segments)}"


def _contains_unicode_smuggling(value: str) -> bool:
    return any(
        unicodedata.category(character) in {"Cf", "Cs", "Co"}
        or (unicodedata.category(character) == "Cc" and character not in {"\t", "\n", "\r"})
        or unicodedata.bidirectional(character) in _BIDI_CONTROL_CLASSES
        or 0xFE00 <= ord(character) <= 0xFE0F
        or 0xE0000 <= ord(character) <= 0xE007F
        or 0xE0100 <= ord(character) <= 0xE01EF
        for character in value
    )


class MCPToolDefinitionGuard:
    """Fail-closed control plane for security-sensitive MCP tool metadata.

    The application owns the signing key and persists returned bundles in a
    protected store. Definition content is never retained in a bundle.
    """

    def __init__(
        self,
        signing_key: bytes,
        policy: MCPToolDefinitionPolicy | None = None,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("signing_key must contain at least 32 bytes")
        self._signing_key = bytes(signing_key)
        self._policy = (policy or MCPToolDefinitionPolicy()).model_copy(deep=True)

    @property
    def policy(self) -> MCPToolDefinitionPolicy:
        """Return a defensive copy of the definition policy."""
        return self._policy.model_copy(deep=True)

    def inspect(self, definitions: Iterable[MCPToolDefinition]) -> MCPToolDefinitionResult:
        """Scan complete definitions without creating a trust decision."""
        items = tuple(definitions)
        findings = self._inspection_findings(items)
        return MCPToolDefinitionResult(
            action=GuardAction.BLOCK if findings else GuardAction.ALLOW,
            findings=tuple(findings),
        )

    def discover(
        self,
        definitions: Iterable[MCPToolDefinition],
        *,
        now: datetime | None = None,
    ) -> MCPToolDefinitionResult:
        """Scan and cryptographically capture an exact discovery snapshot."""
        items = tuple(definitions)
        findings = self._inspection_findings(items)
        if findings:
            return MCPToolDefinitionResult(action=GuardAction.BLOCK, findings=tuple(findings))
        bundle = self._build_bundle(
            items,
            phase=MCPToolDefinitionPhase.DISCOVERY,
            approved_by=None,
            now=now,
        )
        return MCPToolDefinitionResult(action=GuardAction.ALLOW, bundle=bundle)

    def require_discovery(
        self,
        definitions: Iterable[MCPToolDefinition],
        *,
        now: datetime | None = None,
    ) -> MCPToolDefinitionBundle:
        """Return a discovery bundle or raise before definitions reach the model."""
        result = self.discover(definitions, now=now)
        if not result.is_allowed or result.bundle is None:
            raise MCPToolDefinitionError(result)
        return result.bundle

    def approve(
        self,
        definitions: Iterable[MCPToolDefinition],
        discovery: MCPToolDefinitionBundle,
        *,
        approved_by: str,
        now: datetime | None = None,
    ) -> MCPToolDefinitionResult:
        """Bind explicit approval to the exact signed discovery snapshot."""
        items = tuple(definitions)
        findings = self._inspection_findings(items)
        findings.extend(
            self._bundle_findings(
                items,
                discovery,
                expected_phase=MCPToolDefinitionPhase.DISCOVERY,
                mismatch_code=MCPToolDefinitionCode.DISCOVERY_MISMATCH,
            )
        )
        if findings:
            return MCPToolDefinitionResult(action=GuardAction.BLOCK, findings=tuple(findings))
        bundle = self._build_bundle(
            items,
            phase=MCPToolDefinitionPhase.APPROVAL,
            approved_by=approved_by,
            now=now,
        )
        return MCPToolDefinitionResult(action=GuardAction.ALLOW, bundle=bundle)

    def require_approval(
        self,
        definitions: Iterable[MCPToolDefinition],
        discovery: MCPToolDefinitionBundle,
        *,
        approved_by: str,
        now: datetime | None = None,
    ) -> MCPToolDefinitionBundle:
        """Return an approval bundle or raise on unsafe or changed definitions."""
        result = self.approve(definitions, discovery, approved_by=approved_by, now=now)
        if not result.is_allowed or result.bundle is None:
            raise MCPToolDefinitionError(result)
        return result.bundle

    def verify_execution(
        self,
        definition: MCPToolDefinition,
        approval: MCPToolDefinitionBundle,
    ) -> MCPToolDefinitionResult:
        """Re-scan and re-hash the live definition immediately before execution."""
        findings = self._inspection_findings((definition,))
        if findings:
            return MCPToolDefinitionResult(action=GuardAction.BLOCK, findings=tuple(findings))

        if approval.phase != MCPToolDefinitionPhase.APPROVAL or not self._signature_matches(
            approval
        ):
            finding = self._finding(
                MCPToolDefinitionCode.PIN_INVALID,
                Severity.CRITICAL,
                "MCP approval pin is invalid or is not an approval snapshot",
                identities=(definition.identity,),
            )
            return MCPToolDefinitionResult(action=GuardAction.BLOCK, findings=(finding,))

        fingerprint = next(
            (
                item
                for item in approval.fingerprints
                if item.server_id == definition.server_id and item.tool_name == definition.name
            ),
            None,
        )
        if fingerprint is None:
            finding = self._finding(
                MCPToolDefinitionCode.TOOL_NOT_APPROVED,
                Severity.CRITICAL,
                "MCP tool is absent from the approved definition snapshot",
                identities=(definition.identity,),
            )
            return MCPToolDefinitionResult(
                action=GuardAction.REQUIRE_APPROVAL,
                findings=(finding,),
            )
        if fingerprint.definition_digest != definition.definition_digest:
            finding = self._mutation_finding(
                fingerprint,
                fingerprint_definition(definition),
                code=MCPToolDefinitionCode.DEFINITION_CHANGED,
                message="MCP tool definition changed after approval",
            )
            return MCPToolDefinitionResult(
                action=GuardAction.REQUIRE_APPROVAL,
                findings=(finding,),
            )
        return MCPToolDefinitionResult(action=GuardAction.ALLOW, bundle=approval)

    def require_execution(
        self,
        definition: MCPToolDefinition,
        approval: MCPToolDefinitionBundle,
    ) -> MCPToolDefinition:
        """Return the verified definition or raise before invoking the tool."""
        result = self.verify_execution(definition, approval)
        if not result.is_allowed:
            raise MCPToolDefinitionError(result)
        return definition

    def _inspection_findings(
        self,
        definitions: tuple[MCPToolDefinition, ...],
    ) -> list[MCPToolDefinitionFinding]:
        if not definitions or len(definitions) > self._policy.max_tools:
            return [
                self._finding(
                    MCPToolDefinitionCode.DEFINITION_LIMIT_EXCEEDED,
                    Severity.HIGH,
                    "MCP discovery returned an invalid number of tool definitions",
                )
            ]

        findings = self._identity_findings(definitions)
        names = {
            unicodedata.normalize("NFKC", definition.name).casefold() for definition in definitions
        }
        for definition in definitions:
            findings.extend(self._content_findings(definition, names))
            findings.extend(self._schema_findings(definition))
        return findings

    def _identity_findings(
        self,
        definitions: tuple[MCPToolDefinition, ...],
    ) -> list[MCPToolDefinitionFinding]:
        by_name: dict[str, list[MCPToolDefinition]] = {}
        by_skeleton: dict[str, list[MCPToolDefinition]] = {}
        for definition in definitions:
            normalized_name = unicodedata.normalize("NFKC", definition.name).casefold()
            by_name.setdefault(normalized_name, []).append(definition)
            by_skeleton.setdefault(_identity_skeleton(definition.name), []).append(definition)

        findings: list[MCPToolDefinitionFinding] = []
        for same_name in by_name.values():
            if len(same_name) > 1:
                identities = tuple(sorted(item.identity for item in same_name))
                findings.append(
                    self._finding(
                        MCPToolDefinitionCode.DUPLICATE_IDENTITY,
                        Severity.CRITICAL,
                        "Duplicate MCP tool identity would create tool shadowing",
                        identities=identities,
                    )
                )
        for same_skeleton in by_skeleton.values():
            unique_names = {item.name.casefold() for item in same_skeleton}
            if len(same_skeleton) > 1 and len(unique_names) > 1:
                identities = tuple(sorted(item.identity for item in same_skeleton))
                findings.append(
                    self._finding(
                        MCPToolDefinitionCode.CONFUSABLE_IDENTITY,
                        Severity.CRITICAL,
                        "Confusable MCP tool identities would create ambiguous selection",
                        identities=identities,
                    )
                )
        return findings

    def _content_findings(
        self,
        definition: MCPToolDefinition,
        all_names: set[str],
    ) -> list[MCPToolDefinitionFinding]:
        nodes = 0
        string_chars = 0
        suspicious_paths: set[str] = set()
        unicode_paths: set[str] = set()
        reference_paths: set[str] = set()
        current_name = unicodedata.normalize("NFKC", definition.name).casefold()
        other_names = all_names - {current_name}

        for path, value, depth in _walk(definition.canonical_payload):
            nodes += 1
            if depth > self._policy.max_depth:
                return [
                    self._finding(
                        MCPToolDefinitionCode.DEFINITION_LIMIT_EXCEEDED,
                        Severity.HIGH,
                        "MCP tool definition exceeds the configured nesting depth",
                        identities=(definition.identity,),
                        field_paths=(path,),
                    )
                ]
            if not isinstance(value, str):
                continue
            string_chars += len(value)
            bounded = value[:16_384]
            normalized = unicodedata.normalize("NFKC", bounded)
            if any(pattern.search(normalized) for pattern in _INSTRUCTION_PATTERNS):
                suspicious_paths.add(path)
            if _contains_unicode_smuggling(value):
                unicode_paths.add(path)
            if self._policy.reject_cross_tool_references:
                normalized_casefolded = normalized.casefold()
                if any(name and name in normalized_casefolded for name in other_names):
                    reference_paths.add(path)

        findings: list[MCPToolDefinitionFinding] = []
        if nodes > self._policy.max_nodes_per_definition or (
            string_chars > self._policy.max_total_string_chars
        ):
            findings.append(
                self._finding(
                    MCPToolDefinitionCode.DEFINITION_LIMIT_EXCEEDED,
                    Severity.HIGH,
                    "MCP tool definition exceeds configured scan bounds",
                    identities=(definition.identity,),
                )
            )
        if suspicious_paths:
            findings.append(
                self._finding(
                    MCPToolDefinitionCode.HIDDEN_INSTRUCTION,
                    Severity.CRITICAL,
                    "MCP tool metadata contains a model-directed instruction",
                    identities=(definition.identity,),
                    field_paths=tuple(sorted(suspicious_paths)),
                )
            )
        if unicode_paths:
            findings.append(
                self._finding(
                    MCPToolDefinitionCode.UNICODE_SMUGGLING,
                    Severity.CRITICAL,
                    "MCP tool metadata contains invisible or directional Unicode controls",
                    identities=(definition.identity,),
                    field_paths=tuple(sorted(unicode_paths)),
                )
            )
        if reference_paths:
            findings.append(
                self._finding(
                    MCPToolDefinitionCode.CROSS_TOOL_REFERENCE,
                    Severity.HIGH,
                    "MCP metadata references another exposed tool",
                    identities=(definition.identity,),
                    field_paths=tuple(sorted(reference_paths)),
                )
            )
        return findings

    def _schema_findings(
        self,
        definition: MCPToolDefinition,
    ) -> list[MCPToolDefinitionFinding]:
        ambiguous_paths: set[str] = set()
        schemas: list[tuple[str, dict[str, JsonValue]]] = [
            ("/inputSchema", definition.input_schema)
        ]
        if definition.output_schema is not None:
            schemas.append(("/outputSchema", definition.output_schema))

        for root_path, schema in schemas:
            if schema.get("type") != "object":
                ambiguous_paths.add(f"{root_path}/type")
                continue
            self._inspect_schema_node(schema, root_path, ambiguous_paths)

        if not ambiguous_paths:
            return []
        return [
            self._finding(
                MCPToolDefinitionCode.AMBIGUOUS_SEMANTICS,
                Severity.HIGH,
                "MCP tool schema is open-ended or lacks explicit field semantics",
                identities=(definition.identity,),
                field_paths=tuple(sorted(ambiguous_paths)),
            )
        ]

    def _inspect_schema_node(
        self,
        schema: dict[str, JsonValue],
        path: str,
        ambiguous_paths: set[str],
    ) -> None:
        schema_type = schema.get("type")
        properties = schema.get("properties")
        if schema_type == "object" or properties is not None:
            if not isinstance(properties, dict):
                ambiguous_paths.add(f"{path}/properties")
            else:
                for property_name, property_schema in properties.items():
                    escaped = property_name.replace("~", "~0").replace("/", "~1")
                    property_path = f"{path}/properties/{escaped}"
                    if not isinstance(property_schema, dict) or not any(
                        keyword in property_schema
                        for keyword in ("type", "anyOf", "oneOf", "allOf", "$ref")
                    ):
                        ambiguous_paths.add(f"{property_path}/type")
                        continue
                    if self._policy.require_schema_descriptions:
                        description = property_schema.get("description")
                        if not isinstance(description, str) or not description.strip():
                            ambiguous_paths.add(f"{property_path}/description")
                    self._inspect_schema_node(property_schema, property_path, ambiguous_paths)
            if (
                self._policy.require_closed_object_schemas
                and schema.get("additionalProperties") is not False
            ):
                ambiguous_paths.add(f"{path}/additionalProperties")

        if schema_type == "array":
            items = schema.get("items")
            if not isinstance(items, dict):
                ambiguous_paths.add(f"{path}/items")
            else:
                self._inspect_schema_node(items, f"{path}/items", ambiguous_paths)

        for keyword in ("anyOf", "oneOf", "allOf"):
            alternatives = schema.get(keyword)
            if alternatives is None:
                continue
            if not isinstance(alternatives, list) or not alternatives:
                ambiguous_paths.add(f"{path}/{keyword}")
                continue
            for index, alternative in enumerate(alternatives):
                alternative_path = f"{path}/{keyword}/{index}"
                if not isinstance(alternative, dict):
                    ambiguous_paths.add(alternative_path)
                else:
                    self._inspect_schema_node(
                        alternative,
                        alternative_path,
                        ambiguous_paths,
                    )

        definitions = schema.get("$defs")
        if definitions is not None:
            if not isinstance(definitions, dict):
                ambiguous_paths.add(f"{path}/$defs")
            else:
                for definition_name, nested_schema in definitions.items():
                    escaped = definition_name.replace("~", "~0").replace("/", "~1")
                    definition_path = f"{path}/$defs/{escaped}"
                    if not isinstance(nested_schema, dict):
                        ambiguous_paths.add(definition_path)
                    else:
                        self._inspect_schema_node(
                            nested_schema,
                            definition_path,
                            ambiguous_paths,
                        )

    def _build_bundle(
        self,
        definitions: tuple[MCPToolDefinition, ...],
        *,
        phase: MCPToolDefinitionPhase,
        approved_by: str | None,
        now: datetime | None,
    ) -> MCPToolDefinitionBundle:
        unsigned = MCPToolDefinitionBundle(
            phase=phase,
            fingerprints=tuple(
                sorted(
                    (fingerprint_definition(definition) for definition in definitions),
                    key=lambda item: item.identity,
                )
            ),
            issued_at=now or utcnow(),
            approved_by=approved_by,
            signature="0" * 64,
        )
        return unsigned.model_copy(update={"signature": self._signature(unsigned)})

    def _signature(self, bundle: MCPToolDefinitionBundle) -> str:
        encoded = json.dumps(
            bundle_signing_payload(bundle),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hmac.new(self._signing_key, encoded, hashlib.sha256).hexdigest()

    def _signature_matches(self, bundle: MCPToolDefinitionBundle) -> bool:
        return hmac.compare_digest(bundle.signature, self._signature(bundle))

    def _bundle_findings(
        self,
        definitions: tuple[MCPToolDefinition, ...],
        bundle: MCPToolDefinitionBundle,
        *,
        expected_phase: MCPToolDefinitionPhase,
        mismatch_code: MCPToolDefinitionCode,
    ) -> list[MCPToolDefinitionFinding]:
        if bundle.phase != expected_phase or not self._signature_matches(bundle):
            return [
                self._finding(
                    MCPToolDefinitionCode.PIN_INVALID,
                    Severity.CRITICAL,
                    "MCP definition snapshot has an invalid phase or signature",
                )
            ]

        expected = {item.identity: item for item in bundle.fingerprints}
        current = {
            definition.identity: fingerprint_definition(definition) for definition in definitions
        }
        findings: list[MCPToolDefinitionFinding] = []
        for identity in sorted(set(expected).union(current)):
            previous = expected.get(identity)
            observed = current.get(identity)
            if previous is None or observed is None:
                findings.append(
                    self._finding(
                        mismatch_code,
                        Severity.CRITICAL,
                        "MCP tool set changed after the trusted snapshot",
                        identities=(identity,),
                    )
                )
            elif previous.definition_digest != observed.definition_digest:
                findings.append(
                    self._mutation_finding(
                        previous,
                        observed,
                        code=mismatch_code,
                        message="MCP tool definition changed after the trusted snapshot",
                    )
                )
        return findings

    def _mutation_finding(
        self,
        previous: MCPToolFieldFingerprint,
        current: MCPToolFieldFingerprint,
        *,
        code: MCPToolDefinitionCode,
        message: str,
    ) -> MCPToolDefinitionFinding:
        changes: list[MCPDefinitionChange] = []
        paths = sorted(set(previous.field_digests).union(current.field_digests))
        for path in paths:
            before = previous.field_digests.get(path)
            after = current.field_digests.get(path)
            if before == after:
                continue
            if before is None:
                kind = MCPDefinitionChangeKind.ADDED
            elif after is None:
                kind = MCPDefinitionChangeKind.REMOVED
            else:
                kind = MCPDefinitionChangeKind.CHANGED
            changes.append(
                MCPDefinitionChange(
                    path=_safe_field_path(path),
                    kind=kind,
                    previous_digest=before,
                    current_digest=after,
                )
            )
        return self._finding(
            code,
            Severity.CRITICAL,
            message,
            identities=(current.identity,),
            field_paths=tuple(change.path for change in changes),
            changes=tuple(changes),
        )

    @staticmethod
    def _finding(
        code: MCPToolDefinitionCode,
        severity: Severity,
        message: str,
        *,
        identities: tuple[str, ...] = (),
        field_paths: tuple[str, ...] = (),
        changes: tuple[MCPDefinitionChange, ...] = (),
    ) -> MCPToolDefinitionFinding:
        return MCPToolDefinitionFinding(
            code=code,
            severity=severity,
            message=message,
            identities=tuple(_safe_identity_reference(identity) for identity in identities),
            field_paths=tuple(_safe_field_path(path) for path in field_paths),
            changes=changes,
        )
