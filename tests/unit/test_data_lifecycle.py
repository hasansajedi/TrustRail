"""Unit coverage for GenAI data lifecycle and deletion enforcement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from trustrail import (
    DataArtifactKind,
    DataClassification,
    DataDeletionCapability,
    DataDeletionReason,
    DataDeletionReceipt,
    DataDeletionRequest,
    DataDeletionState,
    DataDeletionVerification,
    DataDerivationRequest,
    DataLifecycleCode,
    DataLifecycleError,
    DataLifecycleManager,
    DataLifecycleMetadata,
    DataLifecyclePolicy,
    DataLifecyclePrincipal,
    DataLifecycleRecord,
    DataLifecycleSource,
    DataStorageLocation,
    DataUseKind,
    DataUseRequest,
    MemoryDataLifecycleAuditSink,
    TrainingConsent,
    lifecycle_reference,
)

NOW = datetime(2026, 9, 8, 20, tzinfo=UTC)
SUBJECT = lifecycle_reference("subject-42")


def _policy(**updates) -> DataLifecyclePolicy:
    values = {
        "allowed_purposes": frozenset({"support", "analytics", "fraud-detection"}),
        "allowed_residencies": frozenset({"eu", "de"}),
    }
    values.update(updates)
    return DataLifecyclePolicy(**values)


def _metadata(**updates) -> DataLifecycleMetadata:
    values = {
        "classification": DataClassification.CONFIDENTIAL,
        "allowed_purposes": frozenset({"support", "analytics"}),
        "allowed_residencies": frozenset({"eu", "de"}),
        "retention_until": NOW + timedelta(days=30),
        "training_consent": TrainingConsent.DENIED,
        "subject_refs": frozenset({SUBJECT}),
    }
    values.update(updates)
    return DataLifecycleMetadata(**values)


def _location(
    external_id: str = "objects/root-1",
    *,
    connector_id: str = "primary",
    residency: str = "eu",
    capability: DataDeletionCapability = DataDeletionCapability.HARD_DELETE,
    backup_retention_until: datetime | None = None,
) -> DataStorageLocation:
    return DataStorageLocation.create(
        connector_id=connector_id,
        external_id=external_id,
        residency=residency,
        deletion_capability=capability,
        backup_retention_until=backup_retention_until,
    )


def _record(
    artifact_id: str = "prompt-1",
    *,
    kind: DataArtifactKind = DataArtifactKind.PROMPT,
    metadata: DataLifecycleMetadata | None = None,
    location: DataStorageLocation | None = None,
    tenant_id: str = "tenant-a",
) -> DataLifecycleRecord:
    return DataLifecycleRecord.create(
        artifact_id=artifact_id,
        artifact_kind=kind,
        tenant_id=tenant_id,
        lifecycle=metadata or _metadata(),
        locations=(location or _location(),),
        content_digest="a" * 64,
        created_at=NOW,
    )


def _principal(*, tenant_id: str = "tenant-a", scopes=frozenset({"data.delete"})):
    return DataLifecyclePrincipal(
        principal_id="operator-1",
        tenant_id=tenant_id,
        scopes=scopes,
    )


def _use(record: DataLifecycleRecord, **updates) -> DataUseRequest:
    values = {
        "request_id": "use-1",
        "artifact_id": record.artifact_id,
        "expected_record_digest": record.record_digest,
        "principal": _principal(),
        "purpose_id": "support",
        "destination_residency": "eu",
        "use_kind": DataUseKind.INFERENCE,
    }
    values.update(updates)
    return DataUseRequest(**values)


def _deletion(**updates) -> DataDeletionRequest:
    values = {
        "request_id": "delete-1",
        "tenant_id": "tenant-a",
        "principal": _principal(),
        "reason": DataDeletionReason.SUBJECT_REQUEST,
        "subject_refs": frozenset({SUBJECT}),
    }
    values.update(updates)
    return DataDeletionRequest(**values)


def _derived_request(
    sources: tuple[DataLifecycleRecord, ...],
    *,
    artifact_id: str = "output-1",
    metadata: DataLifecycleMetadata | None = None,
    location: DataStorageLocation | None = None,
) -> DataDerivationRequest:
    return DataDerivationRequest(
        request_id=f"derive-{artifact_id}",
        artifact_id=artifact_id,
        artifact_kind=DataArtifactKind.OUTPUT,
        tenant_id="tenant-a",
        sources=tuple(
            DataLifecycleSource(
                artifact_id=source.artifact_id,
                record_digest=source.record_digest,
            )
            for source in sources
        ),
        proposed_lifecycle=metadata or _metadata(),
        locations=(location or _location(f"objects/{artifact_id}"),),
        content_digest="b" * 64,
        created_at=NOW + timedelta(minutes=1),
    )


class _Connector:
    connector_id = "primary"

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[str] = []

    def tombstone(self, action, tombstone) -> None:
        self.calls.append(f"tombstone:{action.artifact_id}")
        if self.mode == "tombstone":
            raise RuntimeError("tombstone unavailable")

    def delete(self, action, tombstone):
        self.calls.append(f"delete:{action.artifact_id}")
        if self.mode == "delete":
            raise RuntimeError("delete unavailable")
        artifact_id = "substituted" if self.mode == "receipt-binding" else action.artifact_id
        return DataDeletionReceipt(
            artifact_id=artifact_id,
            connector_id=action.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=lifecycle_reference(f"receipt:{action.external_id_ref}"),
            deleted=True,
            occurred_at=tombstone.created_at,
        )

    def verify_deletion(self, action, tombstone, receipt):
        self.calls.append(f"verify:{action.artifact_id}")
        verified = self.mode != "verify"
        return DataDeletionVerification(
            artifact_id=action.artifact_id,
            connector_id=action.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=receipt.receipt_ref,
            evidence_ref=lifecycle_reference(f"evidence:{action.external_id_ref}"),
            verified=verified,
            verified_at=receipt.occurred_at,
        )


def test_metadata_locations_and_records_require_integrity_and_aware_time():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _metadata(retention_until=datetime(2026, 9, 9))
    with pytest.raises(ValidationError, match="one-way"):
        _metadata(subject_refs=frozenset({"raw-subject-id"}))

    location = _location()
    location_values = location.model_dump()
    location_values["external_id_ref"] = lifecycle_reference("substituted")
    with pytest.raises(ValidationError, match="external_id_ref"):
        DataStorageLocation(
            **location_values,
            external_id=location.external_id,
        )

    record = _record()
    assert record.has_valid_integrity
    assert record.model_copy(deep=True).has_valid_integrity
    assert not record.model_copy(update={"tenant_id": "other"}).has_valid_integrity
    assert "objects/root-1" not in record.model_dump_json()


def test_policy_requires_complete_unique_retention_rules():
    policy = _policy()

    with pytest.raises(ValidationError, match="cover every classification"):
        _policy(retention_rules=policy.retention_rules[:-1])
    with pytest.raises(ValidationError, match="define each classification once"):
        _policy(
            retention_rules=(
                *policy.retention_rules[:-1],
                policy.retention_rules[0],
            )
        )


def test_registration_enforces_policy_and_catalog_identity():
    manager = DataLifecycleManager(_policy())
    record = _record()

    registered = manager.require_registration(record, now=NOW)
    duplicate = manager.register(record, now=NOW)
    disallowed = manager.register(
        _record(
            "bad-purpose",
            metadata=_metadata(allowed_purposes=frozenset({"advertising"})),
            location=_location("objects/bad-purpose"),
        ),
        now=NOW,
    )

    assert registered == record
    assert duplicate.findings[0].code == DataLifecycleCode.DUPLICATE_ARTIFACT
    assert disallowed.findings[0].code == DataLifecycleCode.PURPOSE_DENIED
    assert manager.get_record("prompt-1") == record
    assert manager.get_record("missing") is None


def test_registration_rejects_a_pre_tombstoned_artifact():
    manager = DataLifecycleManager(_policy())
    record = _record().with_deletion_state(DataDeletionState.TOMBSTONED, "tombstone-old")

    decision = manager.register(record, now=NOW)

    assert decision.is_blocked
    assert DataLifecycleCode.SOURCE_NOT_ACTIVE in {finding.code for finding in decision.findings}


def test_exact_compliant_use_is_authorized():
    manager = DataLifecycleManager(_policy())
    record = manager.require_registration(_record(), now=NOW)

    authorization = manager.require_use(_use(record), now=NOW + timedelta(minutes=1))

    assert authorization.artifact_id == "prompt-1"
    assert authorization.record_digest == record.record_digest
    assert authorization.use_kind == DataUseKind.INFERENCE


@pytest.mark.parametrize(
    ("updates", "now", "code"),
    [
        ({"purpose_id": "fraud-detection"}, NOW, DataLifecycleCode.PURPOSE_DENIED),
        ({"destination_residency": "us"}, NOW, DataLifecycleCode.RESIDENCY_DENIED),
        (
            {"principal": _principal(tenant_id="tenant-b")},
            NOW,
            DataLifecycleCode.TENANT_MISMATCH,
        ),
        (
            {"requested_retention_until": NOW + timedelta(days=60)},
            NOW,
            DataLifecycleCode.RETENTION_EXTENSION,
        ),
        (
            {"use_kind": DataUseKind.TRAINING},
            NOW,
            DataLifecycleCode.TRAINING_CONSENT_DENIED,
        ),
        ({}, NOW + timedelta(days=31), DataLifecycleCode.RETENTION_EXPIRED),
        ({"expected_record_digest": "f" * 64}, NOW, DataLifecycleCode.SOURCE_CHANGED),
    ],
)
def test_use_blocks_incompatible_purpose_residency_tenant_retention_and_training(
    updates, now, code
):
    manager = DataLifecycleManager(_policy())
    record = manager.require_registration(_record(), now=NOW)

    decision = manager.authorize_use(_use(record, **updates), now=now)

    assert decision.is_blocked
    assert code in {finding.code for finding in decision.findings}


def test_training_requires_both_consent_and_classification_policy():
    public = _record(
        metadata=_metadata(
            classification=DataClassification.PUBLIC,
            training_consent=TrainingConsent.GRANTED,
        )
    )
    manager = DataLifecycleManager(_policy())
    registered = manager.require_registration(public, now=NOW)

    assert (
        manager.require_use(_use(registered, use_kind=DataUseKind.TRAINING), now=NOW).use_kind
        == DataUseKind.TRAINING
    )

    confidential = _record(
        "confidential-training",
        metadata=_metadata(training_consent=TrainingConsent.GRANTED),
        location=_location("objects/confidential-training"),
    )
    assert manager.register(confidential, now=NOW).findings[0].code == (
        DataLifecycleCode.TRAINING_CONSENT_DENIED
    )


def test_derivation_preserves_the_most_restrictive_combined_metadata():
    hold = "legal-hold-1"
    other_subject = lifecycle_reference("subject-99")
    first = _record(
        metadata=_metadata(
            classification=DataClassification.CONFIDENTIAL,
            allowed_purposes=frozenset({"support", "analytics"}),
            allowed_residencies=frozenset({"eu", "de"}),
            retention_until=NOW + timedelta(days=20),
            legal_hold_ids=frozenset({hold}),
        )
    )
    second = _record(
        "document-2",
        kind=DataArtifactKind.RETRIEVED_DOCUMENT,
        metadata=_metadata(
            classification=DataClassification.RESTRICTED,
            allowed_purposes=frozenset({"support"}),
            allowed_residencies=frozenset({"eu"}),
            retention_until=NOW + timedelta(days=10),
            training_consent=TrainingConsent.WITHDRAWN,
            subject_refs=frozenset({other_subject}),
        ),
        location=_location("objects/document-2"),
    )
    manager = DataLifecycleManager(_policy())
    first = manager.require_registration(first, now=NOW)
    second = manager.require_registration(second, now=NOW)
    propagated = _metadata(
        classification=DataClassification.RESTRICTED,
        allowed_purposes=frozenset({"support"}),
        allowed_residencies=frozenset({"eu"}),
        retention_until=NOW + timedelta(days=10),
        training_consent=TrainingConsent.WITHDRAWN,
        legal_hold_ids=frozenset({hold}),
        subject_refs=frozenset({SUBJECT, other_subject}),
    )

    derived = manager.require_derivation(
        _derived_request((first, second), metadata=propagated), now=NOW
    )

    assert derived.lifecycle == propagated
    assert derived.derivation_depth == 1
    assert {source.artifact_id for source in derived.sources} == {"prompt-1", "document-2"}


@pytest.mark.parametrize(
    ("metadata", "code"),
    [
        (
            _metadata(classification=DataClassification.INTERNAL),
            DataLifecycleCode.CLASSIFICATION_DOWNGRADE,
        ),
        (
            _metadata(allowed_purposes=frozenset({"support", "fraud-detection"})),
            DataLifecycleCode.PURPOSE_DOWNGRADE,
        ),
        (
            _metadata(allowed_residencies=frozenset({"eu", "de"})),
            DataLifecycleCode.RESIDENCY_DOWNGRADE,
        ),
        (
            _metadata(retention_until=NOW + timedelta(days=20)),
            DataLifecycleCode.RETENTION_DOWNGRADE,
        ),
        (
            _metadata(training_consent=TrainingConsent.GRANTED),
            DataLifecycleCode.CONSENT_DOWNGRADE,
        ),
        (
            _metadata(legal_hold_ids=frozenset()),
            DataLifecycleCode.LEGAL_HOLD_DROPPED,
        ),
        (
            _metadata(subject_refs=frozenset()),
            DataLifecycleCode.SUBJECT_LINEAGE_DROPPED,
        ),
    ],
)
def test_derivation_rejects_every_metadata_downgrade(metadata, code):
    source_metadata = _metadata(
        classification=DataClassification.CONFIDENTIAL,
        allowed_purposes=frozenset({"support"}),
        allowed_residencies=frozenset({"eu"}),
        retention_until=NOW + timedelta(days=10),
        training_consent=TrainingConsent.DENIED,
        legal_hold_ids=frozenset({"legal-hold-1"}),
    )
    manager = DataLifecycleManager(_policy())
    source = manager.require_registration(_record(metadata=source_metadata), now=NOW)

    decision = manager.derive(_derived_request((source,), metadata=metadata), now=NOW)

    assert decision.is_blocked
    assert code in {finding.code for finding in decision.findings}
    assert manager.get_record("output-1") is None


def test_derivation_rejects_unknown_changed_and_non_active_sources():
    manager = DataLifecycleManager(_policy())
    source = manager.require_registration(_record(), now=NOW)
    unknown = _derived_request((source,)).model_copy(
        update={"sources": (DataLifecycleSource(artifact_id="missing", record_digest="f" * 64),)}
    )
    changed = _derived_request((source,)).model_copy(
        update={
            "sources": (
                DataLifecycleSource(artifact_id=source.artifact_id, record_digest="f" * 64),
            )
        }
    )

    assert manager.derive(unknown, now=NOW).findings[0].code == DataLifecycleCode.SOURCE_UNKNOWN
    assert manager.derive(changed, now=NOW).findings[0].code == DataLifecycleCode.SOURCE_CHANGED


def test_subject_deletion_plan_cascades_to_derived_artifacts_and_honors_holds():
    manager = DataLifecycleManager(_policy())
    root = manager.require_registration(_record(), now=NOW)
    derived = manager.require_derivation(_derived_request((root,)), now=NOW)
    held = manager.require_registration(
        _record(
            "memory-held",
            kind=DataArtifactKind.MEMORY,
            metadata=_metadata(legal_hold_ids=frozenset({"case-7"})),
            location=_location("objects/memory-held"),
        ),
        now=NOW,
    )

    plan_result = manager.plan_deletion(_deletion(), now=NOW)

    assert plan_result.is_planned
    assert plan_result.plan is not None
    assert {action.artifact_id for action in plan_result.plan.actions} == {
        root.artifact_id,
        derived.artifact_id,
    }
    assert plan_result.plan.held_artifact_ids == (held.artifact_id,)
    assert DataLifecycleCode.LEGAL_HOLD_ACTIVE in {finding.code for finding in plan_result.findings}


def test_deletion_planning_requires_scope_known_targets_and_expired_retention():
    manager = DataLifecycleManager(_policy())
    manager.require_registration(_record(), now=NOW)

    missing_scope = manager.plan_deletion(
        _deletion(principal=_principal(scopes=frozenset())), now=NOW
    )
    unknown = manager.plan_deletion(
        _deletion(subject_refs=frozenset(), artifact_ids=frozenset({"missing"})), now=NOW
    )
    premature = manager.plan_deletion(
        _deletion(reason=DataDeletionReason.RETENTION_EXPIRED), now=NOW
    )

    assert missing_scope.findings[0].code == DataLifecycleCode.DELETION_SCOPE_MISSING
    assert unknown.findings[0].code == DataLifecycleCode.DELETION_TARGET_UNKNOWN
    assert premature.findings[0].code == DataLifecycleCode.POLICY_INVALID


def test_successful_deletion_tombstones_before_delete_and_verifies_every_copy():
    connector = _Connector()
    manager = DataLifecycleManager(_policy(), connectors=(connector,))
    record = manager.require_registration(_record(), now=NOW)
    plan = manager.require_deletion_plan(_deletion(), now=NOW)

    result = manager.execute_deletion(plan, now=NOW + timedelta(seconds=1))

    assert result.is_complete
    assert connector.calls == ["tombstone:prompt-1", "delete:prompt-1", "verify:prompt-1"]
    assert len(result.tombstones) == len(result.receipts) == len(result.verifications) == 1
    current = manager.get_record(record.artifact_id)
    assert current is not None
    assert current.deletion_state == DataDeletionState.VERIFIED
    denied = manager.authorize_use(_use(record), now=NOW + timedelta(seconds=2))
    assert DataLifecycleCode.SOURCE_NOT_ACTIVE in {finding.code for finding in denied.findings}


@pytest.mark.parametrize(
    ("connector", "expected"),
    [
        (None, DataLifecycleCode.CONNECTOR_MISSING),
        (_Connector("tombstone"), DataLifecycleCode.TOMBSTONE_FAILED),
        (_Connector("delete"), DataLifecycleCode.DELETION_FAILED),
        (_Connector("receipt-binding"), DataLifecycleCode.DELETION_FAILED),
        (_Connector("verify"), DataLifecycleCode.VERIFICATION_FAILED),
    ],
)
def test_connector_failures_leave_data_unavailable_and_fail_closed(connector, expected):
    connectors = () if connector is None else (connector,)
    manager = DataLifecycleManager(_policy(), connectors=connectors)
    record = manager.require_registration(_record(), now=NOW)
    plan = manager.require_deletion_plan(_deletion(), now=NOW)

    result = manager.execute_deletion(plan, now=NOW + timedelta(seconds=1))

    assert result.action.value == "block"
    assert expected in {finding.code for finding in result.findings}
    current = manager.get_record(record.artifact_id)
    assert current is not None
    assert current.deletion_state == DataDeletionState.FAILED


def test_failed_deletion_can_be_replanned_and_verified():
    connector = _Connector("delete")
    manager = DataLifecycleManager(_policy(), connectors=(connector,))
    record = manager.require_registration(_record(), now=NOW)
    first_plan = manager.require_deletion_plan(_deletion(), now=NOW)

    failed = manager.execute_deletion(first_plan, now=NOW + timedelta(seconds=1))
    connector.mode = "ok"
    connector.calls.clear()
    retry_plan = manager.require_deletion_plan(_deletion(), now=NOW + timedelta(seconds=2))
    retried = manager.execute_deletion(retry_plan, now=NOW + timedelta(seconds=3))

    assert not failed.is_complete
    assert retried.is_complete
    assert connector.calls == ["tombstone:prompt-1", "delete:prompt-1", "verify:prompt-1"]
    current = manager.get_record(record.artifact_id)
    assert current is not None
    assert current.deletion_state == DataDeletionState.VERIFIED


@pytest.mark.parametrize("mutation", ["digest", "expired", "stale"])
def test_tampered_expired_or_stale_deletion_plans_are_rejected_before_tombstoning(mutation):
    connector = _Connector()
    manager = DataLifecycleManager(_policy(), connectors=(connector,))
    manager.require_registration(_record(), now=NOW)
    plan = manager.require_deletion_plan(_deletion(), now=NOW)
    execution_time = NOW + timedelta(seconds=1)
    if mutation == "digest":
        plan = plan.model_copy(update={"plan_digest": "f" * 64})
    elif mutation == "expired":
        execution_time = plan.expires_at
    else:
        manager.execute_deletion(plan, now=NOW + timedelta(seconds=1))
        connector.calls.clear()

    result = manager.execute_deletion(plan, now=execution_time)

    assert result.action.value == "block"
    assert connector.calls == []


def test_backup_retention_keeps_deletion_tombstoned_until_backup_expiry():
    connector = _Connector()
    location = _location(
        capability=DataDeletionCapability.HARD_DELETE,
        backup_retention_until=NOW + timedelta(days=7),
    )
    manager = DataLifecycleManager(_policy(), connectors=(connector,))
    record = manager.require_registration(_record(location=location), now=NOW)
    plan_result = manager.plan_deletion(_deletion(), now=NOW)
    assert DataLifecycleCode.EXTERNAL_GUARANTEE_LIMITED in {
        finding.code for finding in plan_result.findings
    }
    assert plan_result.plan is not None

    result = manager.execute_deletion(plan_result.plan, now=NOW + timedelta(seconds=1))

    assert not result.is_complete
    current = manager.get_record(record.artifact_id)
    assert current is not None
    assert current.deletion_state == DataDeletionState.TOMBSTONED

    retry_plan = manager.require_deletion_plan(
        _deletion(request_id="delete-after-backup-expiry"),
        now=NOW + timedelta(days=8),
    )
    retried = manager.execute_deletion(retry_plan, now=NOW + timedelta(days=8, seconds=1))
    assert retried.is_complete
    final = manager.get_record(record.artifact_id)
    assert final is not None
    assert final.deletion_state == DataDeletionState.VERIFIED


@pytest.mark.parametrize(
    "capability",
    [DataDeletionCapability.TOMBSTONE_ONLY, DataDeletionCapability.UNKNOWN],
)
def test_weak_connector_guarantees_never_mark_deletion_verified(capability):
    connector = _Connector()
    manager = DataLifecycleManager(
        _policy(),
        connectors=(connector,),
    )
    record = manager.require_registration(
        _record(location=_location(capability=capability)),
        now=NOW,
    )
    plan = manager.require_deletion_plan(_deletion(), now=NOW)

    result = manager.execute_deletion(plan, now=NOW + timedelta(seconds=1))

    assert DataLifecycleCode.EXTERNAL_GUARANTEE_LIMITED in {
        finding.code for finding in result.findings
    }
    current = manager.get_record(record.artifact_id)
    assert current is not None
    assert current.deletion_state == DataDeletionState.TOMBSTONED


def test_audit_events_are_bounded_hashed_and_content_free():
    sink = MemoryDataLifecycleAuditSink(max_events=1)
    manager = DataLifecycleManager(_policy(), audit_sink=sink)
    secret_artifact_id = "private-artifact-42"
    secret_principal = "private-operator-42"
    record = _record(secret_artifact_id, location=_location("secret/storage/key"))
    manager.register(record, now=NOW)
    registered = manager.get_record(secret_artifact_id)
    assert registered is not None
    manager.authorize_use(
        _use(
            registered,
            principal=DataLifecyclePrincipal(
                principal_id=secret_principal,
                tenant_id="tenant-a",
                scopes=frozenset(),
            ),
        ),
        now=NOW,
    )

    serialized = sink.events[0].model_dump_json()
    assert len(sink.events) == 1
    assert secret_artifact_id not in serialized
    assert secret_principal not in serialized
    assert "secret/storage/key" not in serialized


def test_require_methods_raise_typed_lifecycle_error():
    manager = DataLifecycleManager(_policy())
    record = manager.require_registration(_record(), now=NOW)

    with pytest.raises(DataLifecycleError) as caught:
        manager.require_use(_use(record, purpose_id="advertising"), now=NOW)

    assert caught.value.result.findings[0].code == DataLifecycleCode.PURPOSE_DENIED
