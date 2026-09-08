"""Typed contracts for GenAI data lifecycle and deletion enforcement."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trustrail.models.enums import GuardAction, Severity

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$"
_DIGEST_PATTERN = r"^[0-9a-f]{64}$"
_REFERENCE_PATTERN = r"^sha256:[0-9a-f]{64}$"


def utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(tz=UTC)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def _digest(value: Any) -> str:
    canonical = json.dumps(
        _canonicalize(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def lifecycle_reference(value: str | bytes) -> str:
    """Return a one-way reference for subjects, locations, and audit identities."""
    raw = value.encode() if isinstance(value, str) else value
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _require_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class DataArtifactKind(StrEnum):
    """GenAI data surfaces that carry lifecycle obligations."""

    PROMPT = "prompt"
    OUTPUT = "output"
    RETRIEVED_DOCUMENT = "retrieved_document"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    CACHE = "cache"
    TRACE = "trace"
    LOG = "log"
    MEMORY = "memory"
    TRAINING_DATASET = "training_dataset"
    FINE_TUNING_DATASET = "fine_tuning_dataset"
    MODEL_ARTIFACT = "model_artifact"
    DERIVED_ARTIFACT = "derived_artifact"


class DataClassification(StrEnum):
    """Ordered sensitivity classification used during derivation."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class TrainingConsent(StrEnum):
    """Consent state for training and fine-tuning use."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    NOT_APPLICABLE = "not_applicable"


class DataUseKind(StrEnum):
    """Purposes of processing checked at a lifecycle boundary."""

    INFERENCE = "inference"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    CACHE = "cache"
    LOG = "log"
    MEMORY = "memory"
    TRAINING = "training"
    EXPORT = "export"


class DataDeletionState(StrEnum):
    """Availability and deletion state of one artifact."""

    ACTIVE = "active"
    TOMBSTONED = "tombstoned"
    VERIFIED = "verified"
    FAILED = "failed"


class DataDeletionReason(StrEnum):
    """Reason recorded in an exact deletion plan."""

    RETENTION_EXPIRED = "retention_expired"
    SUBJECT_REQUEST = "subject_request"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    ADMINISTRATIVE = "administrative"


class DataDeletionCapability(StrEnum):
    """Deletion guarantee exposed by a configured storage connector."""

    HARD_DELETE = "hard_delete"
    CRYPTO_ERASURE = "crypto_erasure"
    TOMBSTONE_ONLY = "tombstone_only"
    UNKNOWN = "unknown"


class DataLifecycleOperation(StrEnum):
    """Lifecycle operation represented in an audit event."""

    REGISTER = "register"
    DERIVE = "derive"
    USE = "use"
    PLAN_DELETION = "plan_deletion"
    EXECUTE_DELETION = "execute_deletion"


class DataLifecycleCode(StrEnum):
    """Stable machine-readable lifecycle decision outcomes."""

    ALLOWED = "allowed"
    POLICY_INVALID = "policy_invalid"
    DUPLICATE_ARTIFACT = "duplicate_artifact"
    ARTIFACT_UNKNOWN = "artifact_unknown"
    RECORD_INTEGRITY_INVALID = "record_integrity_invalid"
    SOURCE_UNKNOWN = "source_unknown"
    SOURCE_CHANGED = "source_changed"
    SOURCE_NOT_ACTIVE = "source_not_active"
    TENANT_MISMATCH = "tenant_mismatch"
    PURPOSE_DENIED = "purpose_denied"
    RESIDENCY_DENIED = "residency_denied"
    RETENTION_EXPIRED = "retention_expired"
    RETENTION_EXTENSION = "retention_extension"
    TRAINING_CONSENT_DENIED = "training_consent_denied"
    CLASSIFICATION_DOWNGRADE = "classification_downgrade"
    PURPOSE_DOWNGRADE = "purpose_downgrade"
    RESIDENCY_DOWNGRADE = "residency_downgrade"
    RETENTION_DOWNGRADE = "retention_downgrade"
    CONSENT_DOWNGRADE = "consent_downgrade"
    LEGAL_HOLD_DROPPED = "legal_hold_dropped"
    SUBJECT_LINEAGE_DROPPED = "subject_lineage_dropped"
    DERIVATION_DEPTH_EXCEEDED = "derivation_depth_exceeded"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"
    DELETION_SCOPE_MISSING = "deletion_scope_missing"
    DELETION_TARGET_UNKNOWN = "deletion_target_unknown"
    DELETION_PLAN_INVALID = "deletion_plan_invalid"
    DELETION_PLAN_EXPIRED = "deletion_plan_expired"
    DELETION_PLAN_STALE = "deletion_plan_stale"
    LEGAL_HOLD_ACTIVE = "legal_hold_active"
    CONNECTOR_MISSING = "connector_missing"
    TOMBSTONE_FAILED = "tombstone_failed"
    DELETION_FAILED = "deletion_failed"
    VERIFICATION_FAILED = "verification_failed"
    EXTERNAL_GUARANTEE_LIMITED = "external_guarantee_limited"


class DataLifecycleMetadata(BaseModel):
    """Immutable obligations propagated to every derived artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: DataClassification
    allowed_purposes: frozenset[str] = Field(min_length=1, max_length=256)
    allowed_residencies: frozenset[str] = Field(min_length=1, max_length=256)
    retention_until: datetime
    training_consent: TrainingConsent = TrainingConsent.DENIED
    legal_hold_ids: frozenset[str] = Field(default_factory=frozenset, max_length=256)
    subject_refs: frozenset[str] = Field(default_factory=frozenset, max_length=10_000)

    @field_validator("allowed_purposes", "allowed_residencies", "legal_hold_ids")
    @classmethod
    def validate_identifiers(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_IDENTIFIER_PATTERN, value) is None for value in values):
            raise ValueError("lifecycle identifiers must use the supported identifier format")
        return values

    @field_validator("subject_refs")
    @classmethod
    def validate_subject_refs(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_REFERENCE_PATTERN, value) is None for value in values):
            raise ValueError("subject_refs must contain one-way sha256 references")
        return values

    @field_validator("retention_until")
    @classmethod
    def require_aware_retention(cls, value: datetime) -> datetime:
        return _require_aware(value, "retention_until")


class DataStorageLocation(BaseModel):
    """One external copy and its declared deletion guarantee."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    connector_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    external_id: str = Field(min_length=1, max_length=2_048, exclude=True, repr=False)
    external_id_ref: str = Field(pattern=_REFERENCE_PATTERN)
    residency: str = Field(pattern=_IDENTIFIER_PATTERN)
    deletion_capability: DataDeletionCapability
    backup_retention_until: datetime | None = None

    @classmethod
    def create(cls, **values: Any) -> DataStorageLocation:
        """Create a location with a one-way external identifier reference."""
        values = dict(values)
        external_id = values["external_id"]
        values.pop("external_id_ref", None)
        return cls(**values, external_id_ref=lifecycle_reference(external_id))

    @field_validator("backup_retention_until")
    @classmethod
    def require_aware_backup_deadline(cls, value: datetime | None) -> datetime | None:
        return value if value is None else _require_aware(value, "backup_retention_until")

    @model_validator(mode="after")
    def validate_external_reference(self) -> DataStorageLocation:
        if self.external_id_ref != lifecycle_reference(self.external_id):
            raise ValueError("external_id_ref does not match external_id")
        return self


class DataLifecycleSource(BaseModel):
    """Integrity-bound lineage edge to one source artifact version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    record_digest: str = Field(pattern=_DIGEST_PATTERN)


class DataLifecycleRecord(BaseModel):
    """Content-free lifecycle record stored beside one GenAI artifact copy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    version: int = Field(default=1, ge=1)
    artifact_kind: DataArtifactKind
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    lifecycle: DataLifecycleMetadata
    locations: tuple[DataStorageLocation, ...] = Field(min_length=1, max_length=1_000)
    sources: tuple[DataLifecycleSource, ...] = Field(default_factory=tuple, max_length=1_000)
    derivation_depth: int = Field(default=0, ge=0, le=1_000)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    created_at: datetime
    deletion_state: DataDeletionState = DataDeletionState.ACTIVE
    tombstone_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)
    record_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(cls, **values: Any) -> DataLifecycleRecord:
        """Create a record with an integrity digest over all lifecycle metadata."""
        values = dict(values)
        values.pop("record_digest", None)
        candidate = cls.model_construct(**values, record_digest="0" * 64)
        return cls(**values, record_digest=_digest(candidate.integrity_payload))

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")

    @model_validator(mode="after")
    def validate_record(self) -> DataLifecycleRecord:
        if len({(item.connector_id, item.external_id_ref) for item in self.locations}) != len(
            self.locations
        ):
            raise ValueError("storage locations must be unique")
        if any(item.residency not in self.lifecycle.allowed_residencies for item in self.locations):
            raise ValueError("storage location residency is not allowed by lifecycle metadata")
        if len({item.artifact_id for item in self.sources}) != len(self.sources):
            raise ValueError("source artifacts must be unique")
        if any(item.artifact_id == self.artifact_id for item in self.sources):
            raise ValueError("an artifact cannot derive from itself")
        if bool(self.sources) != (self.derivation_depth > 0):
            raise ValueError("derived records require sources and positive derivation_depth")
        if self.lifecycle.retention_until <= self.created_at:
            raise ValueError("retention_until must be later than created_at")
        if self.deletion_state == DataDeletionState.ACTIVE and self.tombstone_id is not None:
            raise ValueError("active records cannot carry a tombstone")
        if self.deletion_state != DataDeletionState.ACTIVE and self.tombstone_id is None:
            raise ValueError("non-active records require a tombstone")
        if not self.has_valid_integrity:
            raise ValueError("data lifecycle record integrity check failed")
        return self

    @property
    def integrity_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "content_digest": self.content_digest,
            "created_at": self.created_at,
            "deletion_state": self.deletion_state,
            "derivation_depth": self.derivation_depth,
            "lifecycle": self.lifecycle,
            "locations": self.locations,
            "sources": self.sources,
            "tenant_id": self.tenant_id,
            "tombstone_id": self.tombstone_id,
            "version": self.version,
        }

    @property
    def has_valid_integrity(self) -> bool:
        try:
            return self.record_digest == _digest(self.integrity_payload)
        except (TypeError, ValueError):
            return False

    def with_deletion_state(
        self,
        state: DataDeletionState,
        tombstone_id: str,
    ) -> DataLifecycleRecord:
        """Return the next integrity-bound deletion-state version."""
        return self.create(
            artifact_id=self.artifact_id,
            version=self.version + 1,
            artifact_kind=self.artifact_kind,
            tenant_id=self.tenant_id,
            lifecycle=self.lifecycle,
            locations=self.locations,
            sources=self.sources,
            derivation_depth=self.derivation_depth,
            content_digest=self.content_digest,
            created_at=self.created_at,
            deletion_state=state,
            tombstone_id=tombstone_id,
        )


class DataRetentionRule(BaseModel):
    """Maximum retention duration for one classification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: DataClassification
    max_retention_seconds: int = Field(ge=1, le=315_576_000)


def _default_retention_rules() -> tuple[DataRetentionRule, ...]:
    day = 86_400
    return (
        DataRetentionRule(
            classification=DataClassification.PUBLIC, max_retention_seconds=365 * day
        ),
        DataRetentionRule(
            classification=DataClassification.INTERNAL, max_retention_seconds=180 * day
        ),
        DataRetentionRule(
            classification=DataClassification.CONFIDENTIAL, max_retention_seconds=90 * day
        ),
        DataRetentionRule(
            classification=DataClassification.RESTRICTED, max_retention_seconds=30 * day
        ),
    )


class DataLifecyclePolicy(BaseModel):
    """Closed lifecycle policy for registration, use, derivation, and deletion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    allowed_purposes: frozenset[str] = Field(min_length=1, max_length=1_000)
    allowed_residencies: frozenset[str] = Field(min_length=1, max_length=1_000)
    retention_rules: tuple[DataRetentionRule, ...] = Field(
        default_factory=_default_retention_rules,
        min_length=1,
        max_length=4,
    )
    training_allowed_classifications: frozenset[DataClassification] = Field(
        default_factory=lambda: frozenset({DataClassification.PUBLIC, DataClassification.INTERNAL})
    )
    deletion_required_scope: str = Field(default="data.delete", pattern=_IDENTIFIER_PATTERN)
    deletion_plan_ttl_seconds: int = Field(default=300, ge=1, le=86_400)
    max_artifacts: int = Field(default=100_000, ge=1, le=10_000_000)
    max_sources_per_derivation: int = Field(default=1_000, ge=1, le=10_000)
    max_derivation_depth: int = Field(default=64, ge=1, le=1_000)
    max_deletion_actions: int = Field(default=10_000, ge=1, le=1_000_000)

    @field_validator("allowed_purposes", "allowed_residencies")
    @classmethod
    def validate_identifiers(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_IDENTIFIER_PATTERN, value) is None for value in values):
            raise ValueError("policy values must use the supported identifier format")
        return values

    @model_validator(mode="after")
    def validate_retention_rules(self) -> DataLifecyclePolicy:
        classes = [rule.classification for rule in self.retention_rules]
        if len(classes) != len(set(classes)):
            raise ValueError("retention rules must define each classification once")
        if set(classes) != set(DataClassification):
            raise ValueError("retention rules must cover every classification")
        return self


class DataLifecyclePrincipal(BaseModel):
    """Authenticated identity supplied by trusted application state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    scopes: frozenset[str] = Field(default_factory=frozenset, max_length=1_000)


class DataDerivationRequest(BaseModel):
    """Proposed lifecycle metadata for a new artifact derived from known sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_kind: DataArtifactKind
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    sources: tuple[DataLifecycleSource, ...] = Field(min_length=1, max_length=1_000)
    proposed_lifecycle: DataLifecycleMetadata
    locations: tuple[DataStorageLocation, ...] = Field(min_length=1, max_length=1_000)
    content_digest: str = Field(pattern=_DIGEST_PATTERN)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")

    @property
    def request_digest(self) -> str:
        return _digest(self)


class DataUseRequest(BaseModel):
    """Exact intended use of one lifecycle-governed artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    expected_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    principal: DataLifecyclePrincipal
    purpose_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_residency: str = Field(pattern=_IDENTIFIER_PATTERN)
    use_kind: DataUseKind
    requested_retention_until: datetime | None = None

    @field_validator("requested_retention_until")
    @classmethod
    def require_aware_requested_retention(cls, value: datetime | None) -> datetime | None:
        return value if value is None else _require_aware(value, "requested_retention_until")

    @property
    def request_digest(self) -> str:
        return _digest(self)


class AuthorizedDataUse(BaseModel):
    """Content-free permit for one exact lifecycle-compliant use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    authorization_id: str = Field(pattern=_DIGEST_PATTERN)
    request_digest: str = Field(pattern=_DIGEST_PATTERN)
    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    record_digest: str = Field(pattern=_DIGEST_PATTERN)
    use_kind: DataUseKind
    purpose_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    destination_residency: str = Field(pattern=_IDENTIFIER_PATTERN)


class DataDeletionRequest(BaseModel):
    """Authenticated selection criteria for a cascading deletion plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    tenant_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    principal: DataLifecyclePrincipal
    reason: DataDeletionReason
    artifact_ids: frozenset[str] = Field(default_factory=frozenset, max_length=10_000)
    subject_refs: frozenset[str] = Field(default_factory=frozenset, max_length=10_000)
    cascade_derived: bool = True

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_IDENTIFIER_PATTERN, value) is None for value in values):
            raise ValueError("artifact_ids contain an unsupported identifier")
        return values

    @field_validator("subject_refs")
    @classmethod
    def validate_subject_refs(cls, values: frozenset[str]) -> frozenset[str]:
        if any(re.fullmatch(_REFERENCE_PATTERN, value) is None for value in values):
            raise ValueError("subject_refs must contain one-way sha256 references")
        return values

    @model_validator(mode="after")
    def require_target(self) -> DataDeletionRequest:
        if not self.artifact_ids and not self.subject_refs:
            raise ValueError("a deletion request requires artifact_ids or subject_refs")
        if self.principal.tenant_id != self.tenant_id:
            raise ValueError("deletion principal tenant must match the request tenant")
        return self

    @property
    def request_digest(self) -> str:
        return _digest(self)


class DataDeletionAction(BaseModel):
    """One storage-specific action in an integrity-bound deletion plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    expected_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    connector_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    external_id: str = Field(min_length=1, max_length=2_048, exclude=True, repr=False)
    external_id_ref: str = Field(pattern=_REFERENCE_PATTERN)
    deletion_capability: DataDeletionCapability
    backup_retention_until: datetime | None = None

    @field_validator("backup_retention_until")
    @classmethod
    def require_aware_backup_deadline(cls, value: datetime | None) -> datetime | None:
        return value if value is None else _require_aware(value, "backup_retention_until")

    @model_validator(mode="after")
    def validate_external_reference(self) -> DataDeletionAction:
        if self.external_id_ref != lifecycle_reference(self.external_id):
            raise ValueError("deletion action external reference does not match")
        return self


class DataDeletionPlan(BaseModel):
    """Short-lived plan bound to exact record versions and storage locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    request: DataDeletionRequest
    actions: tuple[DataDeletionAction, ...] = Field(default_factory=tuple, max_length=1_000_000)
    held_artifact_ids: tuple[str, ...] = ()
    selected_record_digests: tuple[str, ...]
    created_at: datetime
    expires_at: datetime
    plan_digest: str = Field(pattern=_DIGEST_PATTERN)

    @classmethod
    def create(cls, **values: Any) -> DataDeletionPlan:
        """Create a deletion plan with an integrity digest."""
        values = dict(values)
        values.pop("plan_digest", None)
        candidate = cls.model_construct(**values, plan_digest="0" * 64)
        return cls(**values, plan_digest=_digest(candidate.integrity_payload))

    @field_validator("created_at", "expires_at")
    @classmethod
    def require_aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _require_aware(value, info.field_name)

    @model_validator(mode="after")
    def validate_plan(self) -> DataDeletionPlan:
        if self.expires_at <= self.created_at:
            raise ValueError("deletion plan expiry must follow creation")
        if tuple(sorted(set(self.held_artifact_ids))) != self.held_artifact_ids:
            raise ValueError("held artifact IDs must be unique and sorted")
        if tuple(sorted(set(self.selected_record_digests))) != self.selected_record_digests:
            raise ValueError("selected record digests must be unique and sorted")
        if not self.has_valid_integrity:
            raise ValueError("deletion plan integrity check failed")
        return self

    @property
    def integrity_payload(self) -> dict[str, Any]:
        return {
            "actions": self.actions,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "held_artifact_ids": self.held_artifact_ids,
            "plan_id": self.plan_id,
            "request": self.request,
            "selected_record_digests": self.selected_record_digests,
        }

    @property
    def has_valid_integrity(self) -> bool:
        try:
            return self.plan_digest == _digest(self.integrity_payload)
        except (TypeError, ValueError):
            return False


class DataDeletionTombstone(BaseModel):
    """Immediate content-free denial marker created before external deletion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tombstone_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    prior_record_digest: str = Field(pattern=_DIGEST_PATTERN)
    plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "created_at")


class DataDeletionReceipt(BaseModel):
    """Content-free connector acknowledgement for a deletion attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    connector_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    external_id_ref: str = Field(pattern=_REFERENCE_PATTERN)
    tombstone_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    receipt_ref: str = Field(pattern=_REFERENCE_PATTERN)
    deleted: bool
    occurred_at: datetime

    @field_validator("occurred_at")
    @classmethod
    def require_aware_occurred_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "occurred_at")


class DataDeletionVerification(BaseModel):
    """Trusted connector evidence that one external copy is no longer available."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    connector_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    external_id_ref: str = Field(pattern=_REFERENCE_PATTERN)
    tombstone_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    plan_digest: str = Field(pattern=_DIGEST_PATTERN)
    receipt_ref: str = Field(pattern=_REFERENCE_PATTERN)
    evidence_ref: str = Field(pattern=_REFERENCE_PATTERN)
    verified: bool
    verified_at: datetime

    @field_validator("verified_at")
    @classmethod
    def require_aware_verified_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "verified_at")


class DataLifecycleFinding(BaseModel):
    """Content-free explanation of a lifecycle decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: DataLifecycleCode
    severity: Severity
    message: str = Field(min_length=1, max_length=500)
    artifact_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    related_refs: tuple[str, ...] = ()


class DataLifecycleAuditEvent(BaseModel):
    """Metadata-only lifecycle evidence safe for a restricted audit sink."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    occurred_at: datetime
    operation: DataLifecycleOperation
    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    finding_codes: tuple[DataLifecycleCode, ...]
    request_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    artifact_refs: tuple[str, ...] = ()
    principal_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    tenant_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    purpose_ref: str | None = Field(default=None, pattern=_REFERENCE_PATTERN)
    action_count: int = Field(default=0, ge=0)
    held_count: int = Field(default=0, ge=0)

    @field_validator("occurred_at")
    @classmethod
    def require_aware_occurred_at(cls, value: datetime) -> datetime:
        return _require_aware(value, "occurred_at")


class DataLifecycleDecision(BaseModel):
    """Result of source registration, derivation, or use authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    findings: tuple[DataLifecycleFinding, ...] = ()
    record: DataLifecycleRecord | None = None
    authorization: AuthorizedDataUse | None = None
    audit_event: DataLifecycleAuditEvent

    @property
    def is_allowed(self) -> bool:
        return self.action == GuardAction.ALLOW

    @property
    def is_blocked(self) -> bool:
        return self.action == GuardAction.BLOCK


class DataDeletionPlanResult(BaseModel):
    """Result of resolving a deletion request across lineage and locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    findings: tuple[DataLifecycleFinding, ...] = ()
    plan: DataDeletionPlan | None = None
    audit_event: DataLifecycleAuditEvent

    @property
    def is_planned(self) -> bool:
        return self.action == GuardAction.ALLOW and self.plan is not None


class DataDeletionResult(BaseModel):
    """Deletion execution outcome with tombstones and connector evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal[GuardAction.ALLOW, GuardAction.BLOCK]
    findings: tuple[DataLifecycleFinding, ...] = ()
    tombstones: tuple[DataDeletionTombstone, ...] = ()
    receipts: tuple[DataDeletionReceipt, ...] = ()
    verifications: tuple[DataDeletionVerification, ...] = ()
    audit_event: DataLifecycleAuditEvent

    @property
    def is_complete(self) -> bool:
        return self.action == GuardAction.ALLOW and not self.findings
