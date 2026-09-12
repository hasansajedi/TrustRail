"""Security regression tests for labeling integrity and bias exceptions."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from trustrail import (
    ArtifactDigest,
    BiasEvaluationCode,
    BiasEvaluationRequest,
    BiasExceptionGrant,
    DatasetFeaturePolicy,
    DatasetFeatureUse,
    DataTransformation,
    GuardAction,
    LabelOrigin,
    LabelQualityAssessment,
    SensitiveFieldHandling,
    StaticBiasExceptionVerifier,
    TrainingAnnotation,
    TrainingDataCode,
    TrainingDataGovernanceVerifier,
    TrainingDatasetManifest,
    TrainingDatasetPolicy,
    TrainingDatasetSubmission,
    annotation_set_digest,
    json_artifact_digest,
)

CORPUS = Path(__file__).parent.parent / "security_corpus" / "training_bias_evaluation.json"
NOW = datetime(2026, 9, 12, tzinfo=UTC)
DATASET_DIGEST = ArtifactDigest.from_bytes(b"approved-private-dataset")


class PassingEvaluator:
    evaluator_id = "quality-security-v1"

    def evaluate(self, annotation, manifest):
        del annotation, manifest
        return LabelQualityAssessment(
            consistency_score=1.0,
            clean_label_poisoning_score=0.0,
        )


class RaisingEvaluator:
    evaluator_id = "quality-raising-v1"

    def evaluate(self, annotation, manifest):
        del annotation, manifest
        raise RuntimeError("private label and provider credential")


def annotation() -> TrainingAnnotation:
    return TrainingAnnotation.from_label(
        annotation_id="annotation-7",
        sample_ref=ArtifactDigest.from_bytes(b"private-sample"),
        label="private health diagnosis label",
        origin=LabelOrigin.HUMAN,
        writer_id="private-annotator-identity",
        approver_id="private-reviewer-identity",
        label_handling=SensitiveFieldHandling.ENCRYPTED,
    )


def manifest() -> TrainingDatasetManifest:
    return TrainingDatasetManifest(
        dataset_id="health-training",
        dataset_version="snapshot-a91f",
        intended_purpose="diagnostic-research",
        features=(
            DatasetFeatureUse(
                feature_id="redacted-clinical-text",
                handling=SensitiveFieldHandling.ENCRYPTED,
            ),
        ),
        source_digest=DATASET_DIGEST,
        output_digest=DATASET_DIGEST,
    )


def policy(approved: TrainingAnnotation | None = None) -> TrainingDatasetPolicy:
    trusted = approved or annotation()
    return TrainingDatasetPolicy(
        policy_id="health-policy-v1",
        dataset_id="health-training",
        dataset_version="snapshot-a91f",
        intended_purposes=frozenset({"diagnostic-research"}),
        features=(
            DatasetFeaturePolicy(
                feature_id="redacted-clinical-text",
                required=True,
                allowed_handling=frozenset({SensitiveFieldHandling.ENCRYPTED}),
            ),
        ),
        authorized_label_writers=frozenset({"private-annotator-identity"}),
        authorized_label_approvers=frozenset({"private-reviewer-identity"}),
        label_sensitivity="highly_sensitive",
        allowed_label_handling=frozenset({SensitiveFieldHandling.ENCRYPTED}),
        approved_annotation_set_digest=annotation_set_digest((trusted,)),
    )


def test_findings_and_evidence_never_serialize_private_labels_or_identities():
    original = annotation()
    tampered = original.model_copy(update={"label": "private changed diagnosis"})
    submission = TrainingDatasetSubmission.from_annotations(
        manifest=manifest(), annotations=(tampered,)
    )

    result = TrainingDataGovernanceVerifier(
        policy(original), quality_evaluators=(PassingEvaluator(),)
    ).verify_dataset(submission, evaluated_at=NOW)

    output = result.model_dump_json()
    assert result.action == GuardAction.QUARANTINE
    assert "private changed diagnosis" not in output
    assert "private health diagnosis label" not in output
    assert "private-annotator-identity" not in output
    assert "private-reviewer-identity" not in output


def test_recalculating_untrusted_label_and_set_digests_cannot_bypass_approved_digest():
    original = annotation()
    changed_label = "attacker-controlled-clean-label"
    tampered = original.model_copy(
        update={
            "label": changed_label,
            "label_digest": json_artifact_digest(changed_label),
        }
    )
    candidate = TrainingDatasetSubmission.from_annotations(
        manifest=manifest(), annotations=(tampered,)
    )

    result = TrainingDataGovernanceVerifier(
        policy(original), quality_evaluators=(PassingEvaluator(),)
    ).verify_dataset(candidate)

    assert TrainingDataCode.LABEL_INTEGRITY_MISMATCH not in {
        finding.code for finding in result.findings
    }
    assert TrainingDataCode.ANNOTATION_SET_INTEGRITY_MISMATCH in {
        finding.code for finding in result.findings
    }


def test_quality_evaluator_failure_is_content_free_and_fails_closed():
    trusted = annotation()
    submission = TrainingDatasetSubmission.from_annotations(
        manifest=manifest(), annotations=(trusted,)
    )

    result = TrainingDataGovernanceVerifier(
        policy(trusted), quality_evaluators=(RaisingEvaluator(),)
    ).verify_dataset(submission)

    assert TrainingDataCode.QUALITY_EVALUATOR_FAILED in {
        finding.code for finding in result.findings
    }
    assert "provider credential" not in result.model_dump_json()


def test_untrusted_transformation_metadata_is_hashed_in_evidence_and_findings():
    private_name = "private transformation command --secret"
    transformation = DataTransformation(
        name=private_name,
        version="private version",
        actor_id="private actor",
        input_digest=DATASET_DIGEST,
        output_digest=DATASET_DIGEST,
    )
    changed_manifest = manifest().model_copy(update={"transformations": (transformation,)})
    trusted = annotation()
    submission = TrainingDatasetSubmission.from_annotations(
        manifest=changed_manifest,
        annotations=(trusted,),
    )

    result = TrainingDataGovernanceVerifier(
        policy(trusted), quality_evaluators=(PassingEvaluator(),)
    ).verify_dataset(submission)

    assert TrainingDataCode.TRANSFORMATION_NOT_APPROVED in {
        finding.code for finding in result.findings
    }
    assert private_name not in result.model_dump_json()


def test_repository_bias_corpus_detects_security_relevant_disparity():
    request = BiasEvaluationRequest.model_validate_json(CORPUS.read_text(encoding="utf-8"))

    report = TrainingDataGovernanceVerifier(
        policy(), quality_evaluators=(PassingEvaluator(),)
    ).evaluate_bias(request, evaluated_at=NOW)

    assert not report.passed
    assert [finding.code for finding in report.findings] == [
        BiasEvaluationCode.BIAS_THRESHOLD_EXCEEDED
    ]


def test_forged_or_expired_bias_exception_cannot_suppress_failure():
    request = BiasEvaluationRequest.model_validate_json(CORPUS.read_text(encoding="utf-8"))
    grant = BiasExceptionGrant(
        exception_id="reviewed-gap",
        evaluation_id=request.evaluation_id,
        metric_id="approval-rate",
        group_id="evaluated-group",
        justification_ref=ArtifactDigest.from_bytes(b"restricted-review"),
        approval_ref=ArtifactDigest.from_bytes(b"trusted-approval"),
        expires_at=NOW + timedelta(days=1),
    )
    forged = grant.model_copy(
        update={"approval_ref": ArtifactDigest.from_bytes(b"forged-approval")}
    )
    forged_request = request.model_copy(update={"exceptions": (forged,)})
    expired = grant.model_copy(update={"expires_at": NOW - timedelta(seconds=1)})
    expired_request = request.model_copy(update={"exceptions": (expired,)})
    verifier = TrainingDataGovernanceVerifier(
        policy(),
        quality_evaluators=(PassingEvaluator(),),
        bias_exception_verifier=StaticBiasExceptionVerifier([grant]),
    )

    forged_report = verifier.evaluate_bias(forged_request, evaluated_at=NOW)
    expired_report = verifier.evaluate_bias(expired_request, evaluated_at=NOW)

    assert BiasEvaluationCode.EXCEPTION_UNVERIFIED in {
        finding.code for finding in forged_report.findings
    }
    assert BiasEvaluationCode.EXCEPTION_EXPIRED in {
        finding.code for finding in expired_report.findings
    }
