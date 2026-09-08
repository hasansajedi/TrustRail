"""Integration coverage for complete lifecycle propagation and verified deletion."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

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
    DataLifecycleManager,
    DataLifecycleMetadata,
    DataLifecyclePolicy,
    DataLifecyclePrincipal,
    DataLifecycleRecord,
    DataLifecycleSource,
    DataStorageLocation,
    DataUseKind,
    DataUseRequest,
    TrainingConsent,
    lifecycle_reference,
)

NOW = datetime(2026, 9, 8, 21, tzinfo=UTC)
SUBJECT = lifecycle_reference("customer-42")


def _content_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _ObjectStoreConnector:
    connector_id = "object-store"

    def __init__(self, objects: dict[str, str]) -> None:
        self.objects = objects
        self.tombstoned: set[str] = set()

    def tombstone(self, action, tombstone) -> None:
        assert action.external_id in self.objects
        self.tombstoned.add(action.external_id)

    def delete(self, action, tombstone):
        self.objects.pop(action.external_id, None)
        return DataDeletionReceipt(
            artifact_id=action.artifact_id,
            connector_id=self.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=lifecycle_reference(f"receipt:{tombstone.tombstone_id}"),
            deleted=True,
            occurred_at=tombstone.created_at,
        )

    def verify_deletion(self, action, tombstone, receipt):
        return DataDeletionVerification(
            artifact_id=action.artifact_id,
            connector_id=self.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=receipt.receipt_ref,
            evidence_ref=lifecycle_reference(f"not-found:{action.external_id_ref}"),
            verified=action.external_id not in self.objects,
            verified_at=receipt.occurred_at,
        )


def _location(artifact_id: str) -> DataStorageLocation:
    return DataStorageLocation.create(
        connector_id="object-store",
        external_id=f"tenant-a/{artifact_id}",
        residency="eu",
        deletion_capability=DataDeletionCapability.HARD_DELETE,
    )


def test_prompt_to_embedding_memory_and_log_lifecycle_ends_in_verified_deletion():
    metadata = DataLifecycleMetadata(
        classification=DataClassification.CONFIDENTIAL,
        allowed_purposes=frozenset({"support"}),
        allowed_residencies=frozenset({"eu"}),
        retention_until=NOW + timedelta(days=14),
        training_consent=TrainingConsent.DENIED,
        subject_refs=frozenset({SUBJECT}),
    )
    policy = DataLifecyclePolicy(
        allowed_purposes=frozenset({"support"}),
        allowed_residencies=frozenset({"eu"}),
    )
    objects = {"tenant-a/prompt-1": "customer account question"}
    connector = _ObjectStoreConnector(objects)
    manager = DataLifecycleManager(policy, connectors=(connector,))
    current = manager.require_registration(
        DataLifecycleRecord.create(
            artifact_id="prompt-1",
            artifact_kind=DataArtifactKind.PROMPT,
            tenant_id="tenant-a",
            lifecycle=metadata,
            locations=(_location("prompt-1"),),
            content_digest=_content_digest(objects["tenant-a/prompt-1"]),
            created_at=NOW,
        ),
        now=NOW,
    )

    kinds = (
        DataArtifactKind.OUTPUT,
        DataArtifactKind.RETRIEVED_DOCUMENT,
        DataArtifactKind.CHUNK,
        DataArtifactKind.EMBEDDING,
        DataArtifactKind.CACHE,
        DataArtifactKind.TRACE,
        DataArtifactKind.LOG,
        DataArtifactKind.MEMORY,
    )
    for index, kind in enumerate(kinds, start=1):
        artifact_id = f"{kind.value}-{index}"
        content = f"derived:{kind.value}"
        objects[f"tenant-a/{artifact_id}"] = content
        current = manager.require_derivation(
            DataDerivationRequest(
                request_id=f"derive-{index}",
                artifact_id=artifact_id,
                artifact_kind=kind,
                tenant_id="tenant-a",
                sources=(
                    DataLifecycleSource(
                        artifact_id=current.artifact_id,
                        record_digest=current.record_digest,
                    ),
                ),
                proposed_lifecycle=metadata,
                locations=(_location(artifact_id),),
                content_digest=_content_digest(content),
                created_at=NOW + timedelta(minutes=index),
            ),
            now=NOW + timedelta(minutes=index),
        )

    principal = DataLifecyclePrincipal(
        principal_id="privacy-operator",
        tenant_id="tenant-a",
        scopes=frozenset({"data.delete"}),
    )
    training = manager.authorize_use(
        DataUseRequest(
            request_id="training-attempt",
            artifact_id=current.artifact_id,
            expected_record_digest=current.record_digest,
            principal=principal,
            purpose_id="support",
            destination_residency="eu",
            use_kind=DataUseKind.TRAINING,
        ),
        now=NOW + timedelta(hours=1),
    )
    assert training.is_blocked

    plan = manager.require_deletion_plan(
        DataDeletionRequest(
            request_id="subject-delete-42",
            tenant_id="tenant-a",
            principal=principal,
            reason=DataDeletionReason.SUBJECT_REQUEST,
            subject_refs=frozenset({SUBJECT}),
        ),
        now=NOW + timedelta(hours=1),
    )
    result = manager.execute_deletion(plan, now=NOW + timedelta(hours=1, seconds=1))

    assert result.is_complete
    assert len(result.verifications) == len(kinds) + 1
    assert objects == {}
    assert all(record.deletion_state == DataDeletionState.VERIFIED for record in manager.records)
