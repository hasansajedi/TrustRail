"""Bypass-oriented corpus for lifecycle downgrade and deletion controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trustrail import (
    DataArtifactKind,
    DataClassification,
    DataDeletionCapability,
    DataDeletionReason,
    DataDeletionRequest,
    DataDerivationRequest,
    DataLifecycleCode,
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

CORPUS_PATH = Path(__file__).parent.parent / "security_corpus" / "data_lifecycle.json"
CASES: list[dict[str, str]] = json.loads(CORPUS_PATH.read_text())
NOW = datetime(2026, 9, 8, 22, tzinfo=UTC)
SUBJECT = lifecycle_reference("security-subject")


def _metadata(**updates):
    values = {
        "classification": DataClassification.RESTRICTED,
        "allowed_purposes": frozenset({"support"}),
        "allowed_residencies": frozenset({"eu"}),
        "retention_until": NOW + timedelta(days=20),
        "training_consent": TrainingConsent.WITHDRAWN,
        "legal_hold_ids": frozenset({"investigation-1"}),
        "subject_refs": frozenset({SUBJECT}),
    }
    values.update(updates)
    return DataLifecycleMetadata(**values)


def _location(artifact_id: str):
    return DataStorageLocation.create(
        connector_id="security-store",
        external_id=f"private/{artifact_id}",
        residency="eu",
        deletion_capability=DataDeletionCapability.HARD_DELETE,
    )


def _manager():
    manager = DataLifecycleManager(
        DataLifecyclePolicy(
            allowed_purposes=frozenset({"support", "advertising"}),
            allowed_residencies=frozenset({"eu", "us"}),
        )
    )
    source = manager.require_registration(
        DataLifecycleRecord.create(
            artifact_id="source-1",
            artifact_kind=DataArtifactKind.PROMPT,
            tenant_id="tenant-a",
            lifecycle=_metadata(),
            locations=(_location("source-1"),),
            content_digest="a" * 64,
            created_at=NOW,
        ),
        now=NOW,
    )
    return manager, source


def _derive(source, metadata):
    return DataDerivationRequest(
        request_id="derive-attack",
        artifact_id="derived-attack",
        artifact_kind=DataArtifactKind.DERIVED_ARTIFACT,
        tenant_id="tenant-a",
        sources=(
            DataLifecycleSource(
                artifact_id=source.artifact_id,
                record_digest=source.record_digest,
            ),
        ),
        proposed_lifecycle=metadata,
        locations=(_location("derived-attack"),),
        content_digest="b" * 64,
        created_at=NOW + timedelta(minutes=1),
    )


def _use(source, **updates):
    values = {
        "request_id": "use-attack",
        "artifact_id": source.artifact_id,
        "expected_record_digest": source.record_digest,
        "principal": DataLifecyclePrincipal(
            principal_id="security-operator",
            tenant_id="tenant-a",
            scopes=frozenset({"data.delete"}),
        ),
        "purpose_id": "support",
        "destination_residency": "eu",
        "use_kind": DataUseKind.INFERENCE,
    }
    values.update(updates)
    return DataUseRequest(**values)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_lifecycle_bypass_corpus_fails_closed_without_raw_identifiers(case):
    manager, source = _manager()
    mutation = case["mutation"]
    value = case["value"]
    metadata = _metadata()

    if mutation == "classification":
        metadata = _metadata(classification=DataClassification(value))
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "derived_purpose":
        metadata = _metadata(allowed_purposes=frozenset({"support", value}))
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "derived_residency":
        metadata = _metadata(allowed_residencies=frozenset({"eu", value}))
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "derived_retention_days":
        metadata = _metadata(retention_until=NOW + timedelta(days=int(value)))
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "training_consent":
        metadata = _metadata(training_consent=TrainingConsent(value))
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "legal_hold":
        metadata = _metadata(legal_hold_ids=frozenset())
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "subject_lineage":
        metadata = _metadata(subject_refs=frozenset())
        decision = manager.derive(_derive(source, metadata), now=NOW)
    elif mutation == "use_purpose":
        decision = manager.authorize_use(_use(source, purpose_id=value), now=NOW)
    elif mutation == "use_residency":
        decision = manager.authorize_use(_use(source, destination_residency=value), now=NOW)
    elif mutation == "use_kind":
        decision = manager.authorize_use(_use(source, use_kind=DataUseKind(value)), now=NOW)
    elif mutation == "use_retention_days":
        decision = manager.authorize_use(
            _use(source, requested_retention_until=NOW + timedelta(days=int(value))),
            now=NOW,
        )
    else:
        principal = _use(source).principal
        if mutation == "deletion_scope":
            principal = principal.model_copy(update={"scopes": frozenset()})
        plan_result = manager.plan_deletion(
            DataDeletionRequest(
                request_id="delete-attack",
                tenant_id="tenant-a",
                principal=principal,
                reason=DataDeletionReason.SUBJECT_REQUEST,
                artifact_ids=frozenset({source.artifact_id}),
            ),
            now=NOW,
        )
        if mutation == "plan_digest":
            assert plan_result.plan is not None
            forged = plan_result.plan.model_copy(update={"plan_digest": value})
            decision = manager.execute_deletion(forged, now=NOW)
        else:
            decision = plan_result

    codes = {finding.code for finding in decision.findings}
    assert DataLifecycleCode(case["expected_code"]) in codes
    serialized = decision.model_dump_json()
    assert "source-1" not in serialized
    assert "security-operator" not in serialized
    assert "private/source-1" not in serialized
