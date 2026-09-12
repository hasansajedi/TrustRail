"""Unit tests for AISVS C1 training-data governance controls."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from trustrail import (
    ArtifactDigest,
    BiasEvaluationCode,
    BiasEvaluationError,
    BiasEvaluationRequest,
    BiasExceptionGrant,
    BiasGroupMetric,
    BiasMetricInput,
    BiasThreshold,
    DatasetFeaturePolicy,
    DatasetFeatureUse,
    DatasetTransformationPolicy,
    DataTransformation,
    FeatureSensitivity,
    GuardAction,
    LabelOrigin,
    LabelQualityAssessment,
    SensitiveFieldHandling,
    StaticBiasExceptionVerifier,
    TrainingAnnotation,
    TrainingDataCode,
    TrainingDataGovernanceError,
    TrainingDataGovernanceVerifier,
    TrainingDatasetManifest,
    TrainingDatasetPolicy,
    TrainingDatasetSubmission,
    annotation_set_digest,
)

NOW = datetime(2026, 9, 12, tzinfo=UTC)
SOURCE_DIGEST = ArtifactDigest.from_bytes(b"raw-dataset")
OUTPUT_DIGEST = ArtifactDigest.from_bytes(b"minimized-dataset")


class PassingQualityEvaluator:
    evaluator_id = "label-quality-v2"

    def evaluate(self, annotation, manifest):
        del annotation, manifest
        return LabelQualityAssessment(
            consistency_score=0.98,
            clean_label_poisoning_score=0.01,
            evidence_ref=ArtifactDigest.from_bytes(b"restricted-quality-report"),
        )


class FixedQualityEvaluator:
    evaluator_id = "label-quality-fixed"

    def __init__(self, consistency: float, poisoning: float) -> None:
        self.consistency = consistency
        self.poisoning = poisoning

    def evaluate(self, annotation, manifest):
        del annotation, manifest
        return LabelQualityAssessment(
            consistency_score=self.consistency,
            clean_label_poisoning_score=self.poisoning,
        )


def annotations() -> tuple[TrainingAnnotation, ...]:
    return (
        TrainingAnnotation.from_label(
            annotation_id="label-1",
            sample_ref=ArtifactDigest.from_bytes(b"sample-1"),
            label={"private_category": "approved"},
            origin=LabelOrigin.HUMAN,
            writer_id="annotator-a",
            approver_id="reviewer-a",
        ),
        TrainingAnnotation.from_label(
            annotation_id="label-2",
            sample_ref=ArtifactDigest.from_bytes(b"sample-2"),
            label="private automated label",
            origin=LabelOrigin.AUTOMATED,
            writer_id="label-service",
            approver_id="reviewer-a",
            confidence=0.95,
        ),
    )


def manifest(**overrides: object) -> TrainingDatasetManifest:
    values: dict[str, object] = {
        "dataset_id": "fraud-training",
        "dataset_version": "snapshot-7f8a",
        "intended_purpose": "fraud-detection",
        "features": (
            DatasetFeatureUse(
                feature_id="transaction-text",
                handling=SensitiveFieldHandling.PLAIN,
            ),
            DatasetFeatureUse(
                feature_id="customer-email",
                handling=SensitiveFieldHandling.ANONYMIZED,
            ),
        ),
        "source_digest": SOURCE_DIGEST,
        "output_digest": OUTPUT_DIGEST,
        "transformations": (
            DataTransformation(
                name="feature-minimizer",
                version="2.1.0",
                actor_id="training-pipeline",
                input_digest=SOURCE_DIGEST,
                output_digest=OUTPUT_DIGEST,
            ),
        ),
    }
    values.update(overrides)
    return TrainingDatasetManifest(**values)


def policy(
    approved_annotations: tuple[TrainingAnnotation, ...] | None = None,
    **overrides: object,
) -> TrainingDatasetPolicy:
    approved = approved_annotations or annotations()
    values: dict[str, object] = {
        "policy_id": "fraud-training-policy-v3",
        "dataset_id": "fraud-training",
        "dataset_version": "snapshot-7f8a",
        "intended_purposes": frozenset({"fraud-detection"}),
        "features": (
            DatasetFeaturePolicy(feature_id="transaction-text", required=True),
            DatasetFeaturePolicy(
                feature_id="customer-email",
                sensitivity=FeatureSensitivity.PERSONAL,
                allowed_handling=frozenset(
                    {
                        SensitiveFieldHandling.EXCLUDED,
                        SensitiveFieldHandling.ANONYMIZED,
                    }
                ),
            ),
        ),
        "transformations": (
            DatasetTransformationPolicy(
                name="feature-minimizer",
                allowed_versions=frozenset({"2.1.0"}),
                authorized_actors=frozenset({"training-pipeline"}),
            ),
        ),
        "authorized_label_writers": frozenset({"annotator-a", "label-service"}),
        "authorized_label_approvers": frozenset({"reviewer-a"}),
        "approved_annotation_set_digest": annotation_set_digest(approved),
    }
    values.update(overrides)
    return TrainingDatasetPolicy(**values)


def submission(
    candidate_annotations: tuple[TrainingAnnotation, ...] | None = None,
    **manifest_overrides: object,
) -> TrainingDatasetSubmission:
    return TrainingDatasetSubmission.from_annotations(
        manifest=manifest(**manifest_overrides),
        annotations=candidate_annotations or annotations(),
    )


def verifier(
    *,
    configured_policy: TrainingDatasetPolicy | None = None,
    quality_evaluators=None,
    bias_exception_verifier=None,
) -> TrainingDataGovernanceVerifier:
    evaluators = (PassingQualityEvaluator(),) if quality_evaluators is None else quality_evaluators
    return TrainingDataGovernanceVerifier(
        configured_policy or policy(),
        quality_evaluators=evaluators,
        bias_exception_verifier=bias_exception_verifier,
    )


def codes(result) -> set[TrainingDataCode]:
    return {finding.code for finding in result.findings}


def bias_request(
    *,
    candidate_value: float = 0.78,
    candidate_samples: int = 100,
    exceptions: tuple[BiasExceptionGrant, ...] = (),
    thresholds: tuple[BiasThreshold, ...] | None = None,
) -> BiasEvaluationRequest:
    return BiasEvaluationRequest(
        evaluation_id="model-release-42",
        subject_ref=ArtifactDigest.from_bytes(b"model-release-42"),
        metrics=(
            BiasMetricInput(
                metric_id="selection-rate",
                metric_kind="selection_rate",
                reference_group_id="reference",
                groups=(
                    BiasGroupMetric(group_id="reference", sample_count=100, value=0.8),
                    BiasGroupMetric(
                        group_id="candidate",
                        sample_count=candidate_samples,
                        value=candidate_value,
                    ),
                ),
            ),
        ),
        thresholds=thresholds
        if thresholds is not None
        else (
            BiasThreshold(
                metric_id="selection-rate",
                minimum_group_samples=30,
                maximum_absolute_gap=0.1,
                minimum_parity_ratio=0.8,
            ),
        ),
        exceptions=exceptions,
    )


class TestTrainingDataModels:
    def test_sensitive_feature_cannot_approve_plain_handling(self):
        with pytest.raises(ValidationError, match="cannot approve plain"):
            DatasetFeaturePolicy(
                feature_id="medical-code",
                sensitivity=FeatureSensitivity.HIGHLY_SENSITIVE,
            )

    def test_duplicate_features_and_annotations_are_rejected(self):
        duplicate_feature = DatasetFeatureUse(
            feature_id="duplicate",
            handling=SensitiveFieldHandling.PLAIN,
        )
        with pytest.raises(ValidationError, match="duplicate feature IDs"):
            manifest(features=(duplicate_feature, duplicate_feature))

        duplicate_annotation = annotations()[0]
        with pytest.raises(ValidationError, match="duplicate annotation IDs"):
            TrainingDatasetSubmission.from_annotations(
                manifest=manifest(),
                annotations=(duplicate_annotation, duplicate_annotation),
            )


class TestDatasetGovernance:
    def test_allows_minimized_integrity_checked_approved_dataset(self):
        result = verifier().verify_dataset(submission(), evaluated_at=NOW)

        assert result.action == GuardAction.ALLOW
        assert result.evidence.annotation_count == 2
        assert result.evidence.automated_annotation_count == 1
        assert result.evidence.evaluated_annotation_count == 2

    @pytest.mark.parametrize(
        ("candidate", "expected_code"),
        [
            (
                submission(intended_purpose="marketing"),
                TrainingDataCode.PURPOSE_NOT_APPROVED,
            ),
            (
                submission(
                    features=(
                        *manifest().features,
                        DatasetFeatureUse(
                            feature_id="unneeded-sensitive-field",
                            handling=SensitiveFieldHandling.ENCRYPTED,
                        ),
                    )
                ),
                TrainingDataCode.FEATURE_NOT_APPROVED,
            ),
            (
                submission(
                    features=(
                        DatasetFeatureUse(
                            feature_id="transaction-text",
                            handling=SensitiveFieldHandling.PLAIN,
                        ),
                        DatasetFeatureUse(
                            feature_id="customer-email",
                            handling=SensitiveFieldHandling.PLAIN,
                        ),
                    )
                ),
                TrainingDataCode.SENSITIVE_HANDLING_MISMATCH,
            ),
            (
                submission(features=(manifest().features[1],)),
                TrainingDataCode.REQUIRED_FEATURE_MISSING,
            ),
        ],
    )
    def test_rejects_purpose_and_feature_policy_violations(
        self,
        candidate: TrainingDatasetSubmission,
        expected_code: TrainingDataCode,
    ):
        result = verifier().verify_dataset(candidate)

        assert result.is_quarantined
        assert expected_code in codes(result)

    def test_rejects_unapproved_transformer_and_broken_lineage(self):
        unapproved = (
            manifest().transformations[0].model_copy(update={"actor_id": "model-chosen-actor"})
        )
        broken = unapproved.model_copy(
            update={"input_digest": ArtifactDigest.from_bytes(b"substituted-input")}
        )

        result = verifier().verify_dataset(submission(transformations=(broken,)))

        assert TrainingDataCode.TRANSFORMATION_NOT_APPROVED in codes(result)
        assert TrainingDataCode.TRANSFORMATION_LINEAGE_BROKEN in codes(result)

    @pytest.mark.parametrize(
        ("annotation_update", "expected_code"),
        [
            ({"writer_id": "model-supplied-writer"}, TrainingDataCode.LABEL_WRITER_NOT_AUTHORIZED),
            ({"approver_id": None}, TrainingDataCode.LABEL_APPROVER_REQUIRED),
            ({"approver_id": "label-service"}, TrainingDataCode.LABEL_SELF_APPROVAL),
            ({"confidence": 0.2}, TrainingDataCode.LABEL_CONFIDENCE_BELOW_THRESHOLD),
            ({"label": "tampered private label"}, TrainingDataCode.LABEL_INTEGRITY_MISMATCH),
        ],
    )
    def test_rejects_annotation_authorization_confidence_and_integrity_bypasses(
        self,
        annotation_update: dict[str, object],
        expected_code: TrainingDataCode,
    ):
        originals = annotations()
        changed = originals[1].model_copy(update=annotation_update)
        candidate = TrainingDatasetSubmission(
            manifest=manifest(),
            annotations=(originals[0], changed),
            observed_annotation_set_digest=annotation_set_digest(originals),
        )

        result = verifier().verify_dataset(candidate)

        assert result.is_quarantined
        assert expected_code in codes(result)
        if expected_code != TrainingDataCode.LABEL_INTEGRITY_MISMATCH:
            assert TrainingDataCode.ANNOTATION_SET_INTEGRITY_MISMATCH in codes(result)

    def test_quality_thresholds_and_missing_hook_fail_closed(self):
        low_quality = verifier(
            quality_evaluators=(FixedQualityEvaluator(0.4, 0.9),)
        ).verify_dataset(submission())
        missing = verifier(quality_evaluators=()).verify_dataset(submission())

        assert TrainingDataCode.LABEL_CONSISTENCY_BELOW_THRESHOLD in codes(low_quality)
        assert TrainingDataCode.CLEAN_LABEL_POISONING_SUSPECTED in codes(low_quality)
        assert TrainingDataCode.QUALITY_EVALUATOR_MISSING in codes(missing)

    def test_sensitive_labels_require_approved_protective_handling(self):
        configured_policy = policy(
            label_sensitivity=FeatureSensitivity.SENSITIVE,
            allowed_label_handling=frozenset({SensitiveFieldHandling.ENCRYPTED}),
        )

        result = verifier(configured_policy=configured_policy).verify_dataset(submission())

        assert TrainingDataCode.SENSITIVE_LABEL_HANDLING_MISMATCH in codes(result)

    def test_require_dataset_raises_with_content_safe_result(self):
        private_label = "private medical label value"
        originals = annotations()
        changed = originals[0].model_copy(update={"label": private_label})
        candidate = submission(candidate_annotations=(changed, originals[1]))

        with pytest.raises(TrainingDataGovernanceError) as exc_info:
            verifier().require_dataset(candidate)

        assert private_label not in exc_info.value.result.model_dump_json()


class TestBiasEvaluation:
    def test_allows_group_metrics_within_thresholds(self):
        report = verifier().evaluate_bias(bias_request(), evaluated_at=NOW)

        assert report.passed
        assert report.metrics[0].groups[1].absolute_gap == pytest.approx(0.02)
        assert report.metrics[0].groups[1].parity_ratio == pytest.approx(0.975)

    def test_blocks_disparity_small_samples_and_missing_threshold(self):
        disparity = verifier().evaluate_bias(bias_request(candidate_value=0.4))
        small_group = verifier().evaluate_bias(bias_request(candidate_samples=5))
        no_threshold = verifier().evaluate_bias(
            bias_request(thresholds=(BiasThreshold(metric_id="other-metric"),))
        )

        assert BiasEvaluationCode.BIAS_THRESHOLD_EXCEEDED in {
            finding.code for finding in disparity.findings
        }
        assert BiasEvaluationCode.GROUP_SAMPLE_TOO_SMALL in {
            finding.code for finding in small_group.findings
        }
        assert [finding.code for finding in no_threshold.findings] == [
            BiasEvaluationCode.METRIC_THRESHOLD_MISSING
        ]

    def test_only_authenticated_unexpired_exact_exception_is_applied(self):
        grant = BiasExceptionGrant(
            exception_id="approved-temporary-disparity",
            evaluation_id="model-release-42",
            metric_id="selection-rate",
            group_id="candidate",
            justification_ref=ArtifactDigest.from_bytes(b"restricted-review"),
            approval_ref=ArtifactDigest.from_bytes(b"signed-approval"),
            expires_at=NOW + timedelta(days=1),
        )
        request = bias_request(candidate_value=0.4, exceptions=(grant,))
        unverified = verifier().evaluate_bias(request, evaluated_at=NOW)
        verified = verifier(
            bias_exception_verifier=StaticBiasExceptionVerifier([grant])
        ).evaluate_bias(request, evaluated_at=NOW)

        assert not unverified.passed
        assert BiasEvaluationCode.EXCEPTION_UNVERIFIED in {
            finding.code for finding in unverified.findings
        }
        assert verified.passed
        assert verified.metrics[0].groups[1].exception_id == grant.exception_id

    def test_require_bias_raises_for_failed_policy(self):
        with pytest.raises(BiasEvaluationError):
            verifier().require_bias(bias_request(candidate_value=0.2))
