"""Fail-closed lifecycle propagation, use authorization, and deletion orchestration."""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Literal, Protocol

from trustrail.exceptions import DataLifecycleError
from trustrail.models.data_lifecycle import (
    AuthorizedDataUse,
    DataClassification,
    DataDeletionAction,
    DataDeletionCapability,
    DataDeletionPlan,
    DataDeletionPlanResult,
    DataDeletionReason,
    DataDeletionReceipt,
    DataDeletionRequest,
    DataDeletionResult,
    DataDeletionState,
    DataDeletionTombstone,
    DataDeletionVerification,
    DataDerivationRequest,
    DataLifecycleAuditEvent,
    DataLifecycleCode,
    DataLifecycleDecision,
    DataLifecycleFinding,
    DataLifecycleMetadata,
    DataLifecycleOperation,
    DataLifecyclePolicy,
    DataLifecycleRecord,
    DataStorageLocation,
    DataUseKind,
    DataUseRequest,
    TrainingConsent,
    _digest,
    lifecycle_reference,
    utcnow,
)
from trustrail.models.enums import GuardAction, Severity

_CLASSIFICATION_RANK = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
}
_CONSENT_RANK = {
    TrainingConsent.GRANTED: 0,
    TrainingConsent.DENIED: 1,
    TrainingConsent.NOT_APPLICABLE: 1,
    TrainingConsent.WITHDRAWN: 2,
}
_STRONG_DELETION = {
    DataDeletionCapability.HARD_DELETE,
    DataDeletionCapability.CRYPTO_ERASURE,
}


class DataDeletionConnector(Protocol):
    """Delete and independently verify copies owned by one storage system."""

    @property
    def connector_id(self) -> str:
        """Return the application-configured connector identity."""
        ...

    def tombstone(
        self,
        action: DataDeletionAction,
        tombstone: DataDeletionTombstone,
    ) -> None:
        """Make the external copy unavailable before physical deletion."""
        ...

    def delete(
        self,
        action: DataDeletionAction,
        tombstone: DataDeletionTombstone,
    ) -> DataDeletionReceipt:
        """Attempt the storage-specific deletion operation."""
        ...

    def verify_deletion(
        self,
        action: DataDeletionAction,
        tombstone: DataDeletionTombstone,
        receipt: DataDeletionReceipt,
    ) -> DataDeletionVerification:
        """Verify through an authoritative read-back or provider mechanism."""
        ...


class DataLifecycleAuditSink(Protocol):
    """Persist content-free lifecycle decisions."""

    def emit(self, event: DataLifecycleAuditEvent) -> None:
        """Persist one lifecycle event without artifact content."""
        ...


class MemoryDataLifecycleAuditSink:
    """Thread-safe bounded audit sink for tests and development."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be at least 1")
        self._events: deque[DataLifecycleAuditEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

    def emit(self, event: DataLifecycleAuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> list[DataLifecycleAuditEvent]:
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


class DataLifecycleManager:
    """Completely mediate lifecycle registration, use, derivation, and deletion.

    Content remains in its owning store. The manager keeps only integrity-bound
    metadata and external storage identifiers needed by trusted connectors.
    """

    def __init__(
        self,
        policy: DataLifecyclePolicy,
        *,
        connectors: Iterable[DataDeletionConnector] = (),
        audit_sink: DataLifecycleAuditSink | None = None,
    ) -> None:
        self._policy = policy.model_copy(deep=True)
        connector_items = tuple(connectors)
        connector_ids = [connector.connector_id for connector in connector_items]
        if len(connector_ids) != len(set(connector_ids)):
            raise ValueError("deletion connector IDs must be unique")
        self._connectors = {connector.connector_id: connector for connector in connector_items}
        self._audit_sink = audit_sink
        self._records: dict[str, DataLifecycleRecord] = {}
        self._lock = threading.RLock()

    @property
    def policy(self) -> DataLifecyclePolicy:
        return self._policy.model_copy(deep=True)

    @property
    def records(self) -> tuple[DataLifecycleRecord, ...]:
        with self._lock:
            return tuple(self._records[key].model_copy(deep=True) for key in sorted(self._records))

    def get_record(self, artifact_id: str) -> DataLifecycleRecord | None:
        """Return a defensive lifecycle record copy when it exists."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.model_copy(deep=True) if record is not None else None

    def register(
        self,
        record: DataLifecycleRecord,
        *,
        now: datetime | None = None,
    ) -> DataLifecycleDecision:
        """Register an original artifact after policy and integrity validation."""
        current_time = now or utcnow()
        candidate = record.model_copy(deep=True)
        with self._lock:
            findings = self._record_findings(candidate)
            if candidate.sources:
                findings.append(
                    self._finding(
                        DataLifecycleCode.SOURCE_UNKNOWN,
                        "Source registration accepts only original artifacts",
                        candidate.artifact_id,
                    )
                )
            if candidate.artifact_id in self._records:
                findings.append(
                    self._finding(
                        DataLifecycleCode.DUPLICATE_ARTIFACT,
                        "Artifact identity is already registered",
                        candidate.artifact_id,
                    )
                )
            if len(self._records) >= self._policy.max_artifacts:
                findings.append(
                    self._finding(
                        DataLifecycleCode.RESOURCE_LIMIT_EXCEEDED,
                        "Lifecycle catalog reached its configured artifact limit",
                        candidate.artifact_id,
                    )
                )
            if findings:
                return self._decision(
                    GuardAction.BLOCK,
                    DataLifecycleOperation.REGISTER,
                    findings,
                    current_time,
                    artifact_ids=(candidate.artifact_id,),
                )
            self._records[candidate.artifact_id] = candidate.model_copy(deep=True)
        return self._decision(
            GuardAction.ALLOW,
            DataLifecycleOperation.REGISTER,
            [],
            current_time,
            artifact_ids=(candidate.artifact_id,),
            record=candidate,
        )

    def require_registration(
        self,
        record: DataLifecycleRecord,
        *,
        now: datetime | None = None,
    ) -> DataLifecycleRecord:
        """Register and return a defensive record or raise on denial."""
        result = self.register(record, now=now)
        if not result.is_allowed or result.record is None:
            raise DataLifecycleError(result)
        return result.record

    def derive(
        self,
        request: DataDerivationRequest,
        *,
        now: datetime | None = None,
    ) -> DataLifecycleDecision:
        """Register a derived artifact only when it preserves every restriction."""
        current_time = now or utcnow()
        snapshot = request.model_copy(deep=True)
        with self._lock:
            findings: list[DataLifecycleFinding] = []
            sources: list[DataLifecycleRecord] = []
            if snapshot.artifact_id in self._records:
                findings.append(
                    self._finding(
                        DataLifecycleCode.DUPLICATE_ARTIFACT,
                        "Derived artifact identity is already registered",
                        snapshot.artifact_id,
                    )
                )
            if len(self._records) >= self._policy.max_artifacts:
                findings.append(
                    self._finding(
                        DataLifecycleCode.RESOURCE_LIMIT_EXCEEDED,
                        "Lifecycle catalog reached its configured artifact limit",
                        snapshot.artifact_id,
                    )
                )
            if len(snapshot.sources) > self._policy.max_sources_per_derivation:
                findings.append(
                    self._finding(
                        DataLifecycleCode.RESOURCE_LIMIT_EXCEEDED,
                        "Derivation contains too many source artifacts",
                        snapshot.artifact_id,
                    )
                )
            for source_ref in snapshot.sources:
                source = self._records.get(source_ref.artifact_id)
                if source is None:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.SOURCE_UNKNOWN,
                            "Derivation references an unknown source artifact",
                            source_ref.artifact_id,
                        )
                    )
                    continue
                if source.record_digest != source_ref.record_digest:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.SOURCE_CHANGED,
                            "Derivation source version no longer matches",
                            source.artifact_id,
                        )
                    )
                if not source.has_valid_integrity:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.RECORD_INTEGRITY_INVALID,
                            "Derivation source failed lifecycle integrity validation",
                            source.artifact_id,
                        )
                    )
                if source.deletion_state != DataDeletionState.ACTIVE:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.SOURCE_NOT_ACTIVE,
                            "Tombstoned or deleted data cannot produce a derived artifact",
                            source.artifact_id,
                        )
                    )
                if source.tenant_id != snapshot.tenant_id:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.TENANT_MISMATCH,
                            "Derived artifacts cannot cross tenant boundaries",
                            source.artifact_id,
                        )
                    )
                sources.append(source)

            if sources:
                findings.extend(
                    self._propagation_findings(
                        snapshot.proposed_lifecycle,
                        sources,
                        snapshot.artifact_id,
                    )
                )
                depth = max(source.derivation_depth for source in sources) + 1
            else:
                depth = 1
            if depth > self._policy.max_derivation_depth:
                findings.append(
                    self._finding(
                        DataLifecycleCode.DERIVATION_DEPTH_EXCEEDED,
                        "Derived artifact exceeds the configured lineage depth",
                        snapshot.artifact_id,
                    )
                )

            try:
                candidate = DataLifecycleRecord.create(
                    artifact_id=snapshot.artifact_id,
                    artifact_kind=snapshot.artifact_kind,
                    tenant_id=snapshot.tenant_id,
                    lifecycle=snapshot.proposed_lifecycle,
                    locations=snapshot.locations,
                    sources=snapshot.sources,
                    derivation_depth=depth,
                    content_digest=snapshot.content_digest,
                    created_at=snapshot.created_at,
                )
            except (TypeError, ValueError):
                candidate = None
                findings.append(
                    self._finding(
                        DataLifecycleCode.POLICY_INVALID,
                        "Derived lifecycle record is structurally invalid",
                        snapshot.artifact_id,
                    )
                )
            if candidate is not None:
                findings.extend(self._record_findings(candidate))
            findings = self._deduplicate(findings)
            if findings or candidate is None:
                return self._decision(
                    GuardAction.BLOCK,
                    DataLifecycleOperation.DERIVE,
                    findings,
                    current_time,
                    request_digest=snapshot.request_digest,
                    artifact_ids=(snapshot.artifact_id, *(item.artifact_id for item in sources)),
                    tenant_id=snapshot.tenant_id,
                )
            self._records[candidate.artifact_id] = candidate.model_copy(deep=True)

        return self._decision(
            GuardAction.ALLOW,
            DataLifecycleOperation.DERIVE,
            [],
            current_time,
            request_digest=snapshot.request_digest,
            artifact_ids=(candidate.artifact_id, *(item.artifact_id for item in sources)),
            tenant_id=snapshot.tenant_id,
            record=candidate,
        )

    def require_derivation(
        self,
        request: DataDerivationRequest,
        *,
        now: datetime | None = None,
    ) -> DataLifecycleRecord:
        """Register and return a derived record or raise on a policy violation."""
        result = self.derive(request, now=now)
        if not result.is_allowed or result.record is None:
            raise DataLifecycleError(result)
        return result.record

    def authorize_use(
        self,
        request: DataUseRequest,
        *,
        now: datetime | None = None,
    ) -> DataLifecycleDecision:
        """Authorize one exact purpose, residency, retention, and processing use."""
        current_time = now or utcnow()
        snapshot = request.model_copy(deep=True)
        with self._lock:
            record = self._records.get(snapshot.artifact_id)
            findings: list[DataLifecycleFinding] = []
            if record is None:
                findings.append(
                    self._finding(
                        DataLifecycleCode.ARTIFACT_UNKNOWN,
                        "Lifecycle-governed artifact is unknown",
                        snapshot.artifact_id,
                    )
                )
            else:
                if record.record_digest != snapshot.expected_record_digest:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.SOURCE_CHANGED,
                            "Requested lifecycle record version no longer matches",
                            record.artifact_id,
                        )
                    )
                if not record.has_valid_integrity:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.RECORD_INTEGRITY_INVALID,
                            "Lifecycle record failed integrity validation",
                            record.artifact_id,
                        )
                    )
                if record.deletion_state != DataDeletionState.ACTIVE:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.SOURCE_NOT_ACTIVE,
                            "Tombstoned or deleted artifacts cannot be used",
                            record.artifact_id,
                        )
                    )
                if record.tenant_id != snapshot.principal.tenant_id:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.TENANT_MISMATCH,
                            "Artifact and authenticated principal tenants differ",
                            record.artifact_id,
                        )
                    )
                if (
                    snapshot.purpose_id not in record.lifecycle.allowed_purposes
                    or snapshot.purpose_id not in self._policy.allowed_purposes
                ):
                    findings.append(
                        self._finding(
                            DataLifecycleCode.PURPOSE_DENIED,
                            "Requested processing purpose is not allowed",
                            record.artifact_id,
                        )
                    )
                if (
                    snapshot.destination_residency not in record.lifecycle.allowed_residencies
                    or snapshot.destination_residency not in self._policy.allowed_residencies
                ):
                    findings.append(
                        self._finding(
                            DataLifecycleCode.RESIDENCY_DENIED,
                            "Destination residency is not allowed",
                            record.artifact_id,
                        )
                    )
                if current_time >= record.lifecycle.retention_until:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.RETENTION_EXPIRED,
                            "Artifact retention deadline has passed",
                            record.artifact_id,
                        )
                    )
                if (
                    snapshot.requested_retention_until is not None
                    and snapshot.requested_retention_until > record.lifecycle.retention_until
                ):
                    findings.append(
                        self._finding(
                            DataLifecycleCode.RETENTION_EXTENSION,
                            "Requested operation would extend artifact retention",
                            record.artifact_id,
                        )
                    )
                if snapshot.use_kind == DataUseKind.TRAINING and (
                    record.lifecycle.training_consent != TrainingConsent.GRANTED
                    or record.lifecycle.classification
                    not in self._policy.training_allowed_classifications
                ):
                    findings.append(
                        self._finding(
                            DataLifecycleCode.TRAINING_CONSENT_DENIED,
                            "Artifact is not eligible for training use",
                            record.artifact_id,
                        )
                    )

            findings = self._deduplicate(findings)
            if findings or record is None:
                return self._decision(
                    GuardAction.BLOCK,
                    DataLifecycleOperation.USE,
                    findings,
                    current_time,
                    request_digest=snapshot.request_digest,
                    artifact_ids=(snapshot.artifact_id,),
                    principal_id=snapshot.principal.principal_id,
                    tenant_id=snapshot.principal.tenant_id,
                    purpose_id=snapshot.purpose_id,
                )
            authorization = AuthorizedDataUse(
                authorization_id=_digest(
                    {
                        "record_digest": record.record_digest,
                        "request_digest": snapshot.request_digest,
                    }
                ),
                request_digest=snapshot.request_digest,
                artifact_id=record.artifact_id,
                record_digest=record.record_digest,
                use_kind=snapshot.use_kind,
                purpose_id=snapshot.purpose_id,
                destination_residency=snapshot.destination_residency,
            )
        return self._decision(
            GuardAction.ALLOW,
            DataLifecycleOperation.USE,
            [],
            current_time,
            request_digest=snapshot.request_digest,
            artifact_ids=(snapshot.artifact_id,),
            principal_id=snapshot.principal.principal_id,
            tenant_id=snapshot.principal.tenant_id,
            purpose_id=snapshot.purpose_id,
            authorization=authorization,
        )

    def require_use(
        self,
        request: DataUseRequest,
        *,
        now: datetime | None = None,
    ) -> AuthorizedDataUse:
        """Return an exact use permit or raise before reading the artifact."""
        result = self.authorize_use(request, now=now)
        if not result.is_allowed or result.authorization is None:
            raise DataLifecycleError(result)
        return result.authorization

    def plan_deletion(
        self,
        request: DataDeletionRequest,
        *,
        now: datetime | None = None,
    ) -> DataDeletionPlanResult:
        """Resolve explicit targets, subject copies, descendants, holds, and locations."""
        current_time = now or utcnow()
        snapshot = request.model_copy(deep=True)
        findings: list[DataLifecycleFinding] = []
        with self._lock:
            if self._policy.deletion_required_scope not in snapshot.principal.scopes:
                findings.append(
                    self._finding(
                        DataLifecycleCode.DELETION_SCOPE_MISSING,
                        "Authenticated principal lacks the deletion scope",
                    )
                )
            selected: set[str] = set()
            for artifact_id in snapshot.artifact_ids:
                record = self._records.get(artifact_id)
                if record is None or record.tenant_id != snapshot.tenant_id:
                    findings.append(
                        self._finding(
                            DataLifecycleCode.DELETION_TARGET_UNKNOWN,
                            "Deletion target is unknown in the requested tenant",
                            artifact_id,
                        )
                    )
                else:
                    selected.add(artifact_id)
            for record in self._records.values():
                if (
                    record.tenant_id == snapshot.tenant_id
                    and record.lifecycle.subject_refs & snapshot.subject_refs
                ):
                    selected.add(record.artifact_id)
            if snapshot.cascade_derived:
                changed = True
                while changed:
                    changed = False
                    for record in self._records.values():
                        if record.tenant_id != snapshot.tenant_id:
                            continue
                        if record.artifact_id in selected:
                            continue
                        if any(source.artifact_id in selected for source in record.sources):
                            selected.add(record.artifact_id)
                            changed = True
            if not selected:
                findings.append(
                    self._finding(
                        DataLifecycleCode.DELETION_TARGET_UNKNOWN,
                        "Deletion criteria did not resolve to an artifact",
                    )
                )
            records = [self._records[artifact_id] for artifact_id in sorted(selected)]
            if snapshot.reason == DataDeletionReason.RETENTION_EXPIRED and any(
                current_time < record.lifecycle.retention_until for record in records
            ):
                findings.append(
                    self._finding(
                        DataLifecycleCode.POLICY_INVALID,
                        "Retention-expiry deletion includes an unexpired artifact",
                    )
                )
            blocking_codes = {
                DataLifecycleCode.DELETION_SCOPE_MISSING,
                DataLifecycleCode.DELETION_TARGET_UNKNOWN,
                DataLifecycleCode.POLICY_INVALID,
            }
            if any(finding.code in blocking_codes for finding in findings):
                return self._plan_result(
                    GuardAction.BLOCK,
                    findings,
                    current_time,
                    snapshot,
                    artifact_ids=selected,
                )

            actions: list[DataDeletionAction] = []
            held_ids: list[str] = []
            pending_records = [
                record for record in records if record.deletion_state != DataDeletionState.VERIFIED
            ]
            for record in pending_records:
                if record.lifecycle.legal_hold_ids:
                    held_ids.append(record.artifact_id)
                    findings.append(
                        self._finding(
                            DataLifecycleCode.LEGAL_HOLD_ACTIVE,
                            "Legal hold prevents deletion of the selected artifact",
                            record.artifact_id,
                        )
                    )
                    continue
                for location in record.locations:
                    action = self._deletion_action(record, location)
                    actions.append(action)
                    if self._limited_guarantee(action, current_time):
                        findings.append(
                            self._finding(
                                DataLifecycleCode.EXTERNAL_GUARANTEE_LIMITED,
                                "External store cannot currently prove complete deletion",
                                record.artifact_id,
                                related=(location.external_id_ref,),
                            )
                        )
            if len(actions) > self._policy.max_deletion_actions:
                findings.append(
                    self._finding(
                        DataLifecycleCode.RESOURCE_LIMIT_EXCEEDED,
                        "Deletion plan exceeds its configured action limit",
                    )
                )
                return self._plan_result(
                    GuardAction.BLOCK,
                    findings,
                    current_time,
                    snapshot,
                    artifact_ids=selected,
                )
            plan = DataDeletionPlan.create(
                plan_id=snapshot.request_id,
                request=snapshot,
                actions=tuple(actions),
                held_artifact_ids=tuple(sorted(held_ids)),
                selected_record_digests=tuple(
                    sorted(record.record_digest for record in pending_records)
                ),
                created_at=current_time,
                expires_at=current_time + timedelta(seconds=self._policy.deletion_plan_ttl_seconds),
            )
        return self._plan_result(
            GuardAction.ALLOW,
            self._deduplicate(findings),
            current_time,
            snapshot,
            artifact_ids=selected,
            plan=plan,
        )

    def require_deletion_plan(
        self,
        request: DataDeletionRequest,
        *,
        now: datetime | None = None,
    ) -> DataDeletionPlan:
        """Return an exact deletion plan or raise when planning is denied."""
        result = self.plan_deletion(request, now=now)
        if not result.is_planned or result.plan is None:
            raise DataLifecycleError(result)
        return result.plan

    def execute_deletion(
        self,
        plan: DataDeletionPlan,
        *,
        now: datetime | None = None,
    ) -> DataDeletionResult:
        """Tombstone locally and externally, delete, verify, then finalize state."""
        current_time = now or utcnow()
        snapshot = plan.model_copy(deep=True)
        findings: list[DataLifecycleFinding] = []
        if not snapshot.has_valid_integrity:
            findings.append(
                self._finding(
                    DataLifecycleCode.DELETION_PLAN_INVALID,
                    "Deletion plan failed integrity validation",
                )
            )
        if current_time >= snapshot.expires_at:
            findings.append(
                self._finding(
                    DataLifecycleCode.DELETION_PLAN_EXPIRED,
                    "Deletion plan has expired",
                )
            )
        if self._policy.deletion_required_scope not in snapshot.request.principal.scopes:
            findings.append(
                self._finding(
                    DataLifecycleCode.DELETION_SCOPE_MISSING,
                    "Deletion plan no longer carries the required scope",
                )
            )

        actions_by_artifact: dict[str, list[DataDeletionAction]] = {}
        for action in snapshot.actions:
            actions_by_artifact.setdefault(action.artifact_id, []).append(action)
        tombstones: dict[str, DataDeletionTombstone] = {}
        with self._lock:
            selected_current = {
                record.record_digest
                for record in self._records.values()
                if record.record_digest in snapshot.selected_record_digests
            }
            if selected_current != set(snapshot.selected_record_digests):
                findings.append(
                    self._finding(
                        DataLifecycleCode.DELETION_PLAN_STALE,
                        "One or more selected lifecycle records changed after planning",
                    )
                )
            for action in snapshot.actions:
                record = self._records.get(action.artifact_id)
                if (
                    record is None
                    or record.record_digest != action.expected_record_digest
                    or record.deletion_state == DataDeletionState.VERIFIED
                    or record.lifecycle.legal_hold_ids
                ):
                    findings.append(
                        self._finding(
                            DataLifecycleCode.DELETION_PLAN_STALE,
                            "Deletion action no longer matches an outstanding, unheld record",
                            action.artifact_id,
                        )
                    )
            if not findings:
                for artifact_id in actions_by_artifact:
                    record = self._records[artifact_id]
                    tombstone = DataDeletionTombstone(
                        tombstone_id=(
                            f"tombstone-{_digest((snapshot.plan_digest, artifact_id))[:32]}"
                        ),
                        artifact_id=artifact_id,
                        prior_record_digest=record.record_digest,
                        plan_digest=snapshot.plan_digest,
                        created_at=current_time,
                    )
                    tombstones[artifact_id] = tombstone
                    self._records[artifact_id] = record.with_deletion_state(
                        DataDeletionState.TOMBSTONED,
                        tombstone.tombstone_id,
                    )
        if findings:
            return self._deletion_result(
                GuardAction.BLOCK,
                self._deduplicate(findings),
                current_time,
                snapshot,
            )

        receipts: list[DataDeletionReceipt] = []
        verifications: list[DataDeletionVerification] = []
        failed_artifacts: set[str] = set()
        limited_artifacts: set[str] = set()
        for action in snapshot.actions:
            connector = self._connectors.get(action.connector_id)
            tombstone = tombstones[action.artifact_id]
            if connector is None:
                findings.append(
                    self._finding(
                        DataLifecycleCode.CONNECTOR_MISSING,
                        "No deletion connector is configured for an external copy",
                        action.artifact_id,
                        related=(action.external_id_ref,),
                    )
                )
                failed_artifacts.add(action.artifact_id)
                continue
            try:
                connector.tombstone(action, tombstone)
            except Exception:
                findings.append(
                    self._finding(
                        DataLifecycleCode.TOMBSTONE_FAILED,
                        "External connector failed to apply the tombstone",
                        action.artifact_id,
                        related=(action.external_id_ref,),
                    )
                )
                failed_artifacts.add(action.artifact_id)
                continue
            try:
                receipt = connector.delete(action, tombstone)
            except Exception:
                receipt = None
            if receipt is None or not self._valid_receipt(action, tombstone, receipt):
                findings.append(
                    self._finding(
                        DataLifecycleCode.DELETION_FAILED,
                        "External connector did not acknowledge deletion",
                        action.artifact_id,
                        related=(action.external_id_ref,),
                    )
                )
                failed_artifacts.add(action.artifact_id)
                continue
            receipts.append(receipt)
            try:
                verification = connector.verify_deletion(action, tombstone, receipt)
            except Exception:
                verification = None
            if verification is None or not self._valid_verification(
                action,
                tombstone,
                receipt,
                verification,
            ):
                findings.append(
                    self._finding(
                        DataLifecycleCode.VERIFICATION_FAILED,
                        "External deletion could not be independently verified",
                        action.artifact_id,
                        related=(action.external_id_ref,),
                    )
                )
                failed_artifacts.add(action.artifact_id)
                continue
            verifications.append(verification)
            if self._limited_guarantee(action, current_time):
                limited_artifacts.add(action.artifact_id)
                findings.append(
                    self._finding(
                        DataLifecycleCode.EXTERNAL_GUARANTEE_LIMITED,
                        "External retention or connector capability limits deletion assurance",
                        action.artifact_id,
                        related=(action.external_id_ref,),
                    )
                )

        for artifact_id in snapshot.held_artifact_ids:
            findings.append(
                self._finding(
                    DataLifecycleCode.LEGAL_HOLD_ACTIVE,
                    "Legal hold keeps the selected artifact active",
                    artifact_id,
                )
            )
        with self._lock:
            for artifact_id in actions_by_artifact:
                record = self._records[artifact_id]
                if artifact_id in failed_artifacts:
                    self._records[artifact_id] = record.with_deletion_state(
                        DataDeletionState.FAILED,
                        record.tombstone_id or tombstones[artifact_id].tombstone_id,
                    )
                elif artifact_id not in limited_artifacts:
                    self._records[artifact_id] = record.with_deletion_state(
                        DataDeletionState.VERIFIED,
                        record.tombstone_id or tombstones[artifact_id].tombstone_id,
                    )

        findings = self._deduplicate(findings)
        outcome_action: Literal[GuardAction.ALLOW, GuardAction.BLOCK] = (
            GuardAction.BLOCK if findings else GuardAction.ALLOW
        )
        return self._deletion_result(
            outcome_action,
            findings,
            current_time,
            snapshot,
            tombstones=tuple(tombstones.values()),
            receipts=tuple(receipts),
            verifications=tuple(verifications),
        )

    def _record_findings(self, record: DataLifecycleRecord) -> list[DataLifecycleFinding]:
        findings: list[DataLifecycleFinding] = []
        if not record.has_valid_integrity:
            findings.append(
                self._finding(
                    DataLifecycleCode.RECORD_INTEGRITY_INVALID,
                    "Lifecycle record failed integrity validation",
                    record.artifact_id,
                )
            )
        if record.deletion_state != DataDeletionState.ACTIVE:
            findings.append(
                self._finding(
                    DataLifecycleCode.SOURCE_NOT_ACTIVE,
                    "Only active artifacts can enter the lifecycle catalog",
                    record.artifact_id,
                )
            )
        if not record.lifecycle.allowed_purposes.issubset(self._policy.allowed_purposes):
            findings.append(
                self._finding(
                    DataLifecycleCode.PURPOSE_DENIED,
                    "Artifact declares a purpose outside lifecycle policy",
                    record.artifact_id,
                )
            )
        if not record.lifecycle.allowed_residencies.issubset(self._policy.allowed_residencies):
            findings.append(
                self._finding(
                    DataLifecycleCode.RESIDENCY_DENIED,
                    "Artifact declares a residency outside lifecycle policy",
                    record.artifact_id,
                )
            )
        retention_rule = next(
            rule
            for rule in self._policy.retention_rules
            if rule.classification == record.lifecycle.classification
        )
        if (
            record.lifecycle.retention_until - record.created_at
        ).total_seconds() > retention_rule.max_retention_seconds:
            findings.append(
                self._finding(
                    DataLifecycleCode.RETENTION_EXTENSION,
                    "Artifact retention exceeds its classification maximum",
                    record.artifact_id,
                )
            )
        if (
            record.lifecycle.training_consent == TrainingConsent.GRANTED
            and record.lifecycle.classification not in self._policy.training_allowed_classifications
        ):
            findings.append(
                self._finding(
                    DataLifecycleCode.TRAINING_CONSENT_DENIED,
                    "Classification policy forbids training even with consent",
                    record.artifact_id,
                )
            )
        return findings

    def _propagation_findings(
        self,
        proposed: DataLifecycleMetadata,
        sources: list[DataLifecycleRecord],
        artifact_id: str,
    ) -> list[DataLifecycleFinding]:
        findings: list[DataLifecycleFinding] = []
        maximum_classification = max(
            (source.lifecycle.classification for source in sources),
            key=lambda classification: _CLASSIFICATION_RANK[classification],
        )
        if (
            _CLASSIFICATION_RANK[proposed.classification]
            < _CLASSIFICATION_RANK[maximum_classification]
        ):
            findings.append(
                self._finding(
                    DataLifecycleCode.CLASSIFICATION_DOWNGRADE,
                    "Derived artifact lowers a source classification",
                    artifact_id,
                )
            )
        allowed_purposes = set(sources[0].lifecycle.allowed_purposes)
        allowed_residencies = set(sources[0].lifecycle.allowed_residencies)
        for source in sources[1:]:
            allowed_purposes.intersection_update(source.lifecycle.allowed_purposes)
            allowed_residencies.intersection_update(source.lifecycle.allowed_residencies)
        if not proposed.allowed_purposes.issubset(allowed_purposes):
            findings.append(
                self._finding(
                    DataLifecycleCode.PURPOSE_DOWNGRADE,
                    "Derived artifact broadens the common source purposes",
                    artifact_id,
                )
            )
        if not proposed.allowed_residencies.issubset(allowed_residencies):
            findings.append(
                self._finding(
                    DataLifecycleCode.RESIDENCY_DOWNGRADE,
                    "Derived artifact broadens the common source residencies",
                    artifact_id,
                )
            )
        earliest_retention = min(source.lifecycle.retention_until for source in sources)
        if proposed.retention_until > earliest_retention:
            findings.append(
                self._finding(
                    DataLifecycleCode.RETENTION_DOWNGRADE,
                    "Derived artifact extends a source retention deadline",
                    artifact_id,
                )
            )
        required_consent_rank = max(
            _CONSENT_RANK[source.lifecycle.training_consent] for source in sources
        )
        if _CONSENT_RANK[proposed.training_consent] < required_consent_rank:
            findings.append(
                self._finding(
                    DataLifecycleCode.CONSENT_DOWNGRADE,
                    "Derived artifact weakens a source training-consent restriction",
                    artifact_id,
                )
            )
        required_holds = set().union(*(source.lifecycle.legal_hold_ids for source in sources))
        if not required_holds.issubset(proposed.legal_hold_ids):
            findings.append(
                self._finding(
                    DataLifecycleCode.LEGAL_HOLD_DROPPED,
                    "Derived artifact drops a source legal hold",
                    artifact_id,
                )
            )
        required_subjects = set().union(*(source.lifecycle.subject_refs for source in sources))
        if not required_subjects.issubset(proposed.subject_refs):
            findings.append(
                self._finding(
                    DataLifecycleCode.SUBJECT_LINEAGE_DROPPED,
                    "Derived artifact drops data-subject deletion lineage",
                    artifact_id,
                )
            )
        return findings

    @staticmethod
    def _deletion_action(
        record: DataLifecycleRecord,
        location: DataStorageLocation,
    ) -> DataDeletionAction:
        return DataDeletionAction(
            artifact_id=record.artifact_id,
            expected_record_digest=record.record_digest,
            connector_id=location.connector_id,
            external_id=location.external_id,
            external_id_ref=location.external_id_ref,
            deletion_capability=location.deletion_capability,
            backup_retention_until=location.backup_retention_until,
        )

    @staticmethod
    def _limited_guarantee(action: DataDeletionAction, now: datetime) -> bool:
        return action.deletion_capability not in _STRONG_DELETION or (
            action.backup_retention_until is not None and action.backup_retention_until > now
        )

    @staticmethod
    def _valid_receipt(
        action: DataDeletionAction,
        tombstone: DataDeletionTombstone,
        receipt: object,
    ) -> bool:
        return (
            isinstance(receipt, DataDeletionReceipt)
            and receipt.deleted
            and receipt.artifact_id == action.artifact_id
            and receipt.connector_id == action.connector_id
            and receipt.external_id_ref == action.external_id_ref
            and receipt.tombstone_id == tombstone.tombstone_id
            and receipt.plan_digest == tombstone.plan_digest
            and receipt.occurred_at >= tombstone.created_at
        )

    @staticmethod
    def _valid_verification(
        action: DataDeletionAction,
        tombstone: DataDeletionTombstone,
        receipt: DataDeletionReceipt,
        verification: object,
    ) -> bool:
        return (
            isinstance(verification, DataDeletionVerification)
            and verification.verified
            and verification.artifact_id == action.artifact_id
            and verification.connector_id == action.connector_id
            and verification.external_id_ref == action.external_id_ref
            and verification.tombstone_id == tombstone.tombstone_id
            and verification.plan_digest == tombstone.plan_digest
            and verification.receipt_ref == receipt.receipt_ref
            and verification.verified_at >= receipt.occurred_at
        )

    def _decision(
        self,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK],
        operation: DataLifecycleOperation,
        findings: list[DataLifecycleFinding],
        now: datetime,
        *,
        request_digest: str | None = None,
        artifact_ids: Iterable[str] = (),
        principal_id: str | None = None,
        tenant_id: str | None = None,
        purpose_id: str | None = None,
        record: DataLifecycleRecord | None = None,
        authorization: AuthorizedDataUse | None = None,
    ) -> DataLifecycleDecision:
        event = self._audit_event(
            action,
            operation,
            findings,
            now,
            request_digest=request_digest,
            artifact_ids=artifact_ids,
            principal_id=principal_id,
            tenant_id=tenant_id,
            purpose_id=purpose_id,
        )
        return DataLifecycleDecision(
            action=action,
            findings=tuple(findings),
            record=record.model_copy(deep=True) if record is not None else None,
            authorization=authorization,
            audit_event=event,
        )

    def _plan_result(
        self,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK],
        findings: list[DataLifecycleFinding],
        now: datetime,
        request: DataDeletionRequest,
        *,
        artifact_ids: Iterable[str],
        plan: DataDeletionPlan | None = None,
    ) -> DataDeletionPlanResult:
        event = self._audit_event(
            action,
            DataLifecycleOperation.PLAN_DELETION,
            findings,
            now,
            request_digest=request.request_digest,
            artifact_ids=artifact_ids,
            principal_id=request.principal.principal_id,
            tenant_id=request.tenant_id,
            action_count=len(plan.actions) if plan is not None else 0,
            held_count=len(plan.held_artifact_ids) if plan is not None else 0,
        )
        return DataDeletionPlanResult(
            action=action,
            findings=tuple(findings),
            plan=plan.model_copy(deep=True) if plan is not None else None,
            audit_event=event,
        )

    def _deletion_result(
        self,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK],
        findings: list[DataLifecycleFinding],
        now: datetime,
        plan: DataDeletionPlan,
        *,
        tombstones: tuple[DataDeletionTombstone, ...] = (),
        receipts: tuple[DataDeletionReceipt, ...] = (),
        verifications: tuple[DataDeletionVerification, ...] = (),
    ) -> DataDeletionResult:
        artifact_ids = {action.artifact_id for action in plan.actions}
        artifact_ids.update(plan.held_artifact_ids)
        event = self._audit_event(
            action,
            DataLifecycleOperation.EXECUTE_DELETION,
            findings,
            now,
            request_digest=plan.request.request_digest,
            artifact_ids=artifact_ids,
            principal_id=plan.request.principal.principal_id,
            tenant_id=plan.request.tenant_id,
            action_count=len(plan.actions),
            held_count=len(plan.held_artifact_ids),
        )
        return DataDeletionResult(
            action=action,
            findings=tuple(findings),
            tombstones=tombstones,
            receipts=receipts,
            verifications=verifications,
            audit_event=event,
        )

    def _audit_event(
        self,
        action: Literal[GuardAction.ALLOW, GuardAction.BLOCK],
        operation: DataLifecycleOperation,
        findings: list[DataLifecycleFinding],
        now: datetime,
        *,
        request_digest: str | None = None,
        artifact_ids: Iterable[str] = (),
        principal_id: str | None = None,
        tenant_id: str | None = None,
        purpose_id: str | None = None,
        action_count: int = 0,
        held_count: int = 0,
    ) -> DataLifecycleAuditEvent:
        event = DataLifecycleAuditEvent(
            occurred_at=now,
            operation=operation,
            action=action,
            finding_codes=tuple(finding.code for finding in findings)
            or (DataLifecycleCode.ALLOWED,),
            request_ref=(
                lifecycle_reference(request_digest) if request_digest is not None else None
            ),
            artifact_refs=tuple(
                lifecycle_reference(artifact_id) for artifact_id in sorted(set(artifact_ids))
            ),
            principal_ref=(lifecycle_reference(principal_id) if principal_id is not None else None),
            tenant_ref=lifecycle_reference(tenant_id) if tenant_id is not None else None,
            purpose_ref=(lifecycle_reference(purpose_id) if purpose_id is not None else None),
            action_count=action_count,
            held_count=held_count,
        )
        if self._audit_sink is not None:
            with contextlib.suppress(Exception):
                self._audit_sink.emit(event)
        return event

    @staticmethod
    def _finding(
        code: DataLifecycleCode,
        message: str,
        artifact_id: str | None = None,
        *,
        related: Iterable[str] = (),
    ) -> DataLifecycleFinding:
        return DataLifecycleFinding(
            code=code,
            severity=Severity.HIGH,
            message=message,
            artifact_ref=(lifecycle_reference(artifact_id) if artifact_id is not None else None),
            related_refs=tuple(sorted(set(related))),
        )

    @staticmethod
    def _deduplicate(findings: list[DataLifecycleFinding]) -> list[DataLifecycleFinding]:
        unique: list[DataLifecycleFinding] = []
        seen: set[tuple[DataLifecycleCode, str | None, tuple[str, ...]]] = set()
        for finding in findings:
            key = (finding.code, finding.artifact_ref, finding.related_refs)
            if key not in seen:
                seen.add(key)
                unique.append(finding)
        return unique
