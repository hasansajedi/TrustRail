"""Integration tests for ingestion, labeling governance, and bias approval."""

import pytest

from trustrail import (
    ArtifactDigest,
    BiasEvaluationRequest,
    BiasGroupMetric,
    BiasMetricInput,
    BiasThreshold,
    DataAssetKind,
    DataIngestionRecord,
    DataPoisoningPolicy,
    DataPoisoningVerifier,
    DataProvenance,
    DatasetFeaturePolicy,
    DatasetFeatureUse,
    DataSourcePolicy,
    IngestionAuthorization,
    LabelOrigin,
    LabelQualityAssessment,
    SensitiveFieldHandling,
    TrainingAnnotation,
    TrainingDataGovernanceError,
    TrainingDataGovernanceVerifier,
    TrainingDatasetManifest,
    TrainingDatasetPolicy,
    TrainingDatasetSubmission,
    TrustLevel,
    annotation_set_digest,
)

DATASET_BYTES = b"approved minimized training dataset"


class IntegrationQualityEvaluator:
    evaluator_id = "integration-label-quality"

    def evaluate(self, annotation, manifest):
        del annotation, manifest
        return LabelQualityAssessment(
            consistency_score=0.97,
            clean_label_poisoning_score=0.02,
        )


def ingested_record() -> DataIngestionRecord:
    return DataIngestionRecord.from_content(
        item_id="training-release-12",
        kind=DataAssetKind.TRAINING_DATA,
        content=DATASET_BYTES,
        provenance=DataProvenance(
            source_id="training-store",
            source_uri="s3://approved-training/release-12",
            version="release-12",
            trust_level=TrustLevel.TRUSTED,
        ),
        authorization=IngestionAuthorization(
            writer_id="training-pipeline",
            tenant_id="tenant-a",
            purpose="model-training",
        ),
    )


def poisoning_verifier(record: DataIngestionRecord) -> DataPoisoningVerifier:
    return DataPoisoningVerifier(
        DataPoisoningPolicy(
            sources=(
                DataSourcePolicy(
                    source_id="training-store",
                    source_uri="s3://approved-training/release-12",
                    allowed_kinds=frozenset({DataAssetKind.TRAINING_DATA}),
                    trust_level=TrustLevel.TRUSTED,
                    authorized_writers=frozenset({"training-pipeline"}),
                    allowed_tenants=frozenset({"tenant-a"}),
                    allowed_purposes=frozenset({"model-training"}),
                    allowed_versions=frozenset({"release-12"}),
                ),
            ),
            expected_digests={record.item_id: record.observed_digest},
        )
    )


def governed_submission(
    label: str = "approved class",
) -> tuple[TrainingDatasetSubmission, TrainingDatasetPolicy]:
    annotation = TrainingAnnotation.from_label(
        annotation_id="annotation-12",
        sample_ref=ArtifactDigest.from_bytes(b"sample-12"),
        label=label,
        origin=LabelOrigin.AUTOMATED,
        writer_id="label-service",
        approver_id="dataset-reviewer",
        confidence=0.96,
    )
    manifest = TrainingDatasetManifest(
        dataset_id="training-release-12",
        dataset_version="release-12",
        intended_purpose="model-training",
        features=(
            DatasetFeatureUse(
                feature_id="normalized-text",
                handling=SensitiveFieldHandling.PLAIN,
            ),
        ),
        source_digest=ArtifactDigest.from_bytes(DATASET_BYTES),
        output_digest=ArtifactDigest.from_bytes(DATASET_BYTES),
    )
    submission = TrainingDatasetSubmission.from_annotations(
        manifest=manifest,
        annotations=(annotation,),
    )
    policy = TrainingDatasetPolicy(
        policy_id="training-governance-v1",
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        intended_purposes=frozenset({manifest.intended_purpose}),
        features=(DatasetFeaturePolicy(feature_id="normalized-text", required=True),),
        authorized_label_writers=frozenset({"label-service"}),
        authorized_label_approvers=frozenset({"dataset-reviewer"}),
        approved_annotation_set_digest=annotation_set_digest((annotation,)),
    )
    return submission, policy


def test_training_release_requires_ingestion_labels_and_bias_to_pass():
    record = ingested_record()
    accepted = poisoning_verifier(record).require(record)
    submission, policy = governed_submission()
    governance = TrainingDataGovernanceVerifier(
        policy,
        quality_evaluators=(IntegrationQualityEvaluator(),),
    )

    governed = governance.require_dataset(submission)
    bias = governance.require_bias(
        BiasEvaluationRequest(
            evaluation_id="training-release-12-bias",
            subject_ref=accepted.observed_digest,
            metrics=(
                BiasMetricInput(
                    metric_id="true-positive-rate",
                    metric_kind="true_positive_rate",
                    reference_group_id="reference",
                    groups=(
                        BiasGroupMetric(group_id="reference", sample_count=200, value=0.91),
                        BiasGroupMetric(group_id="evaluated", sample_count=180, value=0.88),
                    ),
                ),
            ),
            thresholds=(
                BiasThreshold(
                    metric_id="true-positive-rate",
                    minimum_group_samples=100,
                    maximum_absolute_gap=0.05,
                ),
            ),
        )
    )

    assert governed.manifest.dataset_id == accepted.item_id
    assert bias.passed


def test_ingestion_hash_does_not_allow_a_rewritten_label_set():
    record = ingested_record()
    poisoning_verifier(record).require(record)
    approved, policy = governed_submission()
    changed, _ = governed_submission(label="quietly rewritten class")
    assert approved.manifest.output_digest.matches(changed.manifest.output_digest)
    governance = TrainingDataGovernanceVerifier(
        policy,
        quality_evaluators=(IntegrationQualityEvaluator(),),
    )

    with pytest.raises(TrainingDataGovernanceError):
        governance.require_dataset(changed)
