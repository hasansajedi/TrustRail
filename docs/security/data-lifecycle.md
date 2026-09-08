# GenAI data lifecycle and verified deletion

GenAI applications copy data across prompts, outputs, retrieval documents,
chunks, embeddings, caches, traces, logs, memory, training datasets, and model
artifacts. Deleting the source object is therefore not enough. Every derived
copy must retain the source's use and deletion obligations.

`DataLifecycleManager` provides a typed, fail-closed control plane for those
obligations. It:

- integrity-binds classification, allowed purposes and residencies, retention,
  training consent, legal holds, subject references, locations, and lineage;
- prevents derived artifacts from weakening any source restriction;
- authorizes an exact use before artifact content is read or sent elsewhere;
- resolves deletion targets through subject references and transitive lineage;
- tombstones copies before physical deletion and checks connector receipts and
  verification evidence; and
- emits content-free decisions and audit events.

The manager stores metadata, digests, and connector routing identifiers. It does
not store the artifact content.

## Configure policy and register source data

Assign labels from authenticated application state at collection or ingestion.
Do not let a prompt, model, retrieved document, client request, or external store
choose its own classification, purpose, residency, consent, subject, or hold
labels.

```python
import hashlib
from datetime import UTC, datetime, timedelta

from trustrail import (
    DataArtifactKind,
    DataClassification,
    DataDeletionCapability,
    DataLifecycleManager,
    DataLifecycleMetadata,
    DataLifecyclePolicy,
    DataLifecycleRecord,
    DataStorageLocation,
    TrainingConsent,
    lifecycle_reference,
)

now = datetime.now(tz=UTC)
policy = DataLifecyclePolicy(
    allowed_purposes=frozenset({"customer-support", "fraud-review"}),
    allowed_residencies=frozenset({"eu", "de"}),
    deletion_required_scope="data.delete",
    deletion_plan_ttl_seconds=300,
    max_artifacts=100_000,
    max_sources_per_derivation=100,
    max_derivation_depth=32,
    max_deletion_actions=10_000,
)
# Supply your reviewed connector implementation at application startup. The
# connector contract and an example are shown below.
manager = DataLifecycleManager(policy, connectors=(object_store_connector,))

prompt_bytes = prompt_text.encode()
prompt_record = manager.require_registration(
    DataLifecycleRecord.create(
        artifact_id="prompt-42",
        artifact_kind=DataArtifactKind.PROMPT,
        tenant_id=authenticated_tenant_id,
        lifecycle=DataLifecycleMetadata(
            classification=DataClassification.CONFIDENTIAL,
            allowed_purposes=frozenset({"customer-support"}),
            allowed_residencies=frozenset({"eu"}),
            retention_until=now + timedelta(days=14),
            training_consent=TrainingConsent.DENIED,
            legal_hold_ids=frozenset(),
            subject_refs=frozenset(
                {lifecycle_reference(authenticated_customer_id)}
            ),
        ),
        locations=(
            DataStorageLocation.create(
                connector_id="primary-object-store",
                external_id="tenant-a/prompts/prompt-42",
                residency="eu",
                deletion_capability=DataDeletionCapability.HARD_DELETE,
            ),
        ),
        content_digest=hashlib.sha256(prompt_bytes).hexdigest(),
        created_at=now,
    ),
    now=now,
)
```

Default maximum retention is 365 days for public, 180 for internal, 90 for
confidential, and 30 for restricted data. Supply all four `DataRetentionRule`
entries to change those limits. A location's residency must also appear in the
record and global policy. External identifiers are retained in memory for
connector dispatch but excluded from Pydantic serialization and representations.

## Preserve restrictions through derivation

Call `require_derivation()` whenever a prompt or source produces an output,
retrieved document, chunk, embedding, cache entry, trace, log, memory, training
dataset, fine-tuning dataset, model artifact, or another derived artifact.

```python
from trustrail import (
    DataDerivationRequest,
    DataLifecycleSource,
)

output_record = manager.require_derivation(
    DataDerivationRequest(
        request_id="derive-output-42",
        artifact_id="output-42",
        artifact_kind=DataArtifactKind.OUTPUT,
        tenant_id=authenticated_tenant_id,
        sources=(
            DataLifecycleSource(
                artifact_id=prompt_record.artifact_id,
                record_digest=prompt_record.record_digest,
            ),
        ),
        proposed_lifecycle=prompt_record.lifecycle,
        locations=(
            DataStorageLocation.create(
                connector_id="primary-object-store",
                external_id="tenant-a/outputs/output-42",
                residency="eu",
                deletion_capability=DataDeletionCapability.HARD_DELETE,
            ),
        ),
        content_digest=hashlib.sha256(output_text.encode()).hexdigest(),
        created_at=now + timedelta(seconds=1),
    )
)
```

For multiple sources, the derived record must use at least the highest source
classification, no more than the intersection of source purposes and
residencies, no later than the earliest retention deadline, and no weaker than
the most restrictive consent. It must retain the union of legal holds and
subject references. Source record digests prevent silently rebinding lineage to
a changed version.

## Authorize every use

Build the principal and intended use from authenticated application state. Call
`require_use()` immediately before reading, retrieving, embedding, caching,
logging, storing in memory, training on, or exporting the artifact.

```python
from trustrail import (
    DataLifecyclePrincipal,
    DataUseKind,
    DataUseRequest,
)

permit = manager.require_use(
    DataUseRequest(
        request_id="use-output-42",
        artifact_id=output_record.artifact_id,
        expected_record_digest=output_record.record_digest,
        principal=DataLifecyclePrincipal(
            principal_id=authenticated_service_id,
            tenant_id=authenticated_tenant_id,
        ),
        purpose_id="customer-support",
        destination_residency="eu",
        use_kind=DataUseKind.INFERENCE,
    )
)

# Read and use the external object only after the exact request is authorized.
```

Authorization fails for stale record bindings, tenant mismatch, inactive or
expired data, incompatible purposes or destinations, retention extension, or
training without both granted consent and an allowed classification. A permit is
content-free and bound to one request and record version; it is not a bearer
credential for arbitrary future use.

## Connect and verify deletion

Implement one `DataDeletionConnector` per external store. `tombstone()` must
make the copy unavailable first. `delete()` returns an acknowledgement bound to
the action, tombstone, and plan. `verify_deletion()` should use an authoritative
read-back or provider deletion-status mechanism, not merely repeat the delete
response.

```python
from trustrail import DataDeletionReceipt, DataDeletionVerification


class ObjectStoreDeletionConnector:
    connector_id = "primary-object-store"

    def tombstone(self, action, tombstone) -> None:
        object_store.deny_reads(action.external_id, tombstone.tombstone_id)

    def delete(self, action, tombstone) -> DataDeletionReceipt:
        receipt_id = object_store.delete(action.external_id)
        return DataDeletionReceipt(
            artifact_id=action.artifact_id,
            connector_id=self.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=lifecycle_reference(receipt_id),
            deleted=True,
            occurred_at=datetime.now(tz=UTC),
        )

    def verify_deletion(
        self, action, tombstone, receipt
    ) -> DataDeletionVerification:
        absent = object_store.authoritative_head(action.external_id).is_missing
        return DataDeletionVerification(
            artifact_id=action.artifact_id,
            connector_id=self.connector_id,
            external_id_ref=action.external_id_ref,
            tombstone_id=tombstone.tombstone_id,
            plan_digest=tombstone.plan_digest,
            receipt_ref=receipt.receipt_ref,
            evidence_ref=lifecycle_reference(
                f"head:{action.external_id_ref}:{absent}"
            ),
            verified=absent,
            verified_at=datetime.now(tz=UTC),
        )
```

Pass connectors when creating the manager, then plan and execute deletion:

```python
from trustrail import DataDeletionReason, DataDeletionRequest

deletion_principal = DataLifecyclePrincipal(
    principal_id=authenticated_privacy_operator_id,
    tenant_id=authenticated_tenant_id,
    scopes=frozenset({"data.delete"}),
)
plan = manager.require_deletion_plan(
    DataDeletionRequest(
        request_id="erase-customer-42",
        tenant_id=authenticated_tenant_id,
        principal=deletion_principal,
        reason=DataDeletionReason.SUBJECT_REQUEST,
        subject_refs=frozenset(
            {lifecycle_reference(authenticated_customer_id)}
        ),
        cascade_derived=True,
    )
)
result = manager.execute_deletion(plan)
if not result.is_complete:
    route_to_privacy_operations(result.findings)
```

Planning resolves every outstanding matching artifact and, by default, every
transitive descendant. This permits failed or incomplete deletion to be retried.
Plans are short-lived and integrity-bound to exact record versions and locations.
Execution applies local and external tombstones before deletion, validates
receipts and verification evidence, and marks a record `VERIFIED` only when every
location has a strong current guarantee. Connector failure leaves the catalog
record unavailable in `FAILED`, although the external copy may still require
operator remediation; incomplete guarantees leave it `TOMBSTONED`. Both states
are denied by later use checks.

Legal holds keep selected artifacts active and make the overall deletion result
incomplete. A `TOMBSTONE_ONLY` or `UNKNOWN` connector, or a backup retention
deadline in the future, can acknowledge deletion but cannot produce a complete
result. Operators should retry failed actions, retain evidence, and reconcile
backup expiry through a durable workflow.

## Audit and operating requirements

`MemoryDataLifecycleAuditSink` is a bounded development and test sink. Production
audit implementations should persist events immutably and alert when delivery
fails. Events hash request, artifact, principal, tenant, and purpose identities;
they contain no artifact bytes or raw external identifiers.

### Assumptions, limitations, and residual risk

- The manager catalog and tombstone state are process-local. Production systems
  need shared, atomic, durable record, plan-claim, tombstone, retry, and audit
  storage at every ingestion, read, derivation, and deletion boundary.
- Complete mediation is an application requirement. A caller that reads a store,
  invokes a model, writes a cache or log, or starts training without these checks
  bypasses lifecycle enforcement.
- Labels and identities are only as trustworthy as their issuer. Authenticate
  collection, consent withdrawal, legal-hold, residency, ownership, purpose, and
  connector metadata outside model-controlled inputs.
- SHA-256 digests detect changes relative to protected metadata; they do not
  prove content safety, legal basis, identity, consent, or deletion. Protect the
  catalog against unauthorized rewrite and use keyed/pseudonymous subject
  references where low-entropy identifiers could be guessed.
- A connector's evidence and provider guarantees remain trusted dependencies.
  Verify replicas, caches, search indexes, snapshots, backups, observability
  exports, and downstream processors separately. Cryptographic erasure also
  depends on correct key isolation and destruction.
- Deleting training or fine-tuning data does not remove its influence from an
  existing model. Model unlearning, retraining, evaluation, model-version
  retirement, and provider-specific deletion remain separate governed actions.
- Legal and regulatory retention, deletion, residency, and consent requirements
  vary by jurisdiction. These APIs enforce configured technical policy; they do
  not decide which policy is lawful or sufficient.
- Content and external IDs still exist in application and connector memory.
  Apply least privilege, encryption, log redaction, bounded retention, incident
  response, and secure disposal to those systems.

Use these controls with [sensitive-data detection](sensitive-data.md),
[data and model poisoning checks](data-model-poisoning.md),
[vector and embedding security](vector-embedding-security.md), and
[persistent-memory security](memory-security.md).
