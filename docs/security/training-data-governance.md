# Training-data labeling integrity and bias evaluation

`TrainingDataGovernanceVerifier` protects the boundary between approved source
data and a training or fine-tuning job. It complements `DataPoisoningVerifier`:
the ingestion verifier authenticates source bytes, while this verifier checks
feature minimization, annotation authorization and integrity, automated-label
quality, clean-label poisoning signals, and aggregate bias thresholds.

These controls map to OWASP AISVS C1, especially C1.1.1, C1.2.1–C1.2.3, and
C1.3.2–C1.3.5.

## Define an approved dataset policy

Build annotation digests from the raw label at the trusted labeling boundary.
The label remains available to the training pipeline, but never appears in a
`TrainingDataFinding` or `TrainingDataEvidence`.

```python
from trustrail import (
    ArtifactDigest,
    DatasetFeaturePolicy,
    FeatureSensitivity,
    LabelOrigin,
    SensitiveFieldHandling,
    TrainingAnnotation,
    TrainingDatasetPolicy,
    annotation_set_digest,
)

annotation = TrainingAnnotation.from_label(
    annotation_id="annotation-1042",
    sample_ref=ArtifactDigest.from_bytes(b"stable-private-sample-reference"),
    label="approved private label",
    origin=LabelOrigin.AUTOMATED,
    writer_id="label-service",
    approver_id="dataset-reviewer",
    confidence=0.96,
    label_handling=SensitiveFieldHandling.ENCRYPTED,
)

policy = TrainingDatasetPolicy(
    policy_id="support-training-v4",
    dataset_id="support-training",
    dataset_version="snapshot-7f8a",
    intended_purposes=frozenset({"intent-classification"}),
    features=(
        DatasetFeaturePolicy(feature_id="normalized-text", required=True),
        DatasetFeaturePolicy(
            feature_id="customer-region",
            sensitivity=FeatureSensitivity.PERSONAL,
            allowed_handling=frozenset({SensitiveFieldHandling.ANONYMIZED}),
        ),
    ),
    authorized_label_writers=frozenset({"label-service"}),
    authorized_label_approvers=frozenset({"dataset-reviewer"}),
    label_sensitivity=FeatureSensitivity.SENSITIVE,
    allowed_label_handling=frozenset({SensitiveFieldHandling.ENCRYPTED}),
    approved_annotation_set_digest=annotation_set_digest((annotation,)),
    minimum_automated_label_confidence=0.9,
    minimum_consistency_score=0.85,
    maximum_clean_label_poisoning_score=0.15,
)
```

The policy should come from authenticated control-plane state. Do not let a
model, dataset contributor, or labeling client supply its own allowed features,
writers, approvers, thresholds, or approved annotation-set digest.

## Verify labels and transformations

Create a `TrainingDatasetManifest` containing only purpose-approved features and
the integrity-linked `DataTransformation` chain. Then create the submission with
`TrainingDatasetSubmission.from_annotations()` and evaluate it:

```python
from trustrail import LabelQualityAssessment, TrainingDataGovernanceVerifier


class LabelQualityEvaluator:
    evaluator_id = "consistency-and-clean-label-v3"

    def evaluate(self, annotation, manifest):
        consistency, poisoning_risk, report_digest = evaluate_private_label(
            annotation.label,
            manifest,
        )
        return LabelQualityAssessment(
            consistency_score=consistency,
            clean_label_poisoning_score=poisoning_risk,
            evidence_ref=report_digest,
        )


verifier = TrainingDataGovernanceVerifier(
    policy,
    quality_evaluators=(LabelQualityEvaluator(),),
)
accepted = verifier.require_dataset(submission)
```

The verifier quarantines unknown or excessive features, unsafe sensitive-field
handling, wrong purpose/version, unapproved transformation code or actors,
broken digest lineage, unauthorized writers or approvers, self-approval,
modified labels or annotation sets, low-confidence automated labels, inconsistent
labels, suspicious clean-label scores, evaluator errors, and evaluation-budget
overflow.

## Evaluate aggregate bias metrics

Bias input is provider-neutral: supply a stable metric ID and kind, a reference
group, aggregate group sample counts and values, and an approved threshold.
`BiasEvidenceReport` records the thresholds, absolute gaps, parity ratios, and
decision without individual predictions or labels.

```python
from trustrail import (
    BiasEvaluationRequest,
    BiasGroupMetric,
    BiasMetricInput,
    BiasThreshold,
)

request = BiasEvaluationRequest(
    evaluation_id="model-release-42",
    subject_ref=model_artifact_digest,
    metrics=(
        BiasMetricInput(
            metric_id="selection-rate",
            metric_kind="selection_rate",
            reference_group_id="reference",
            groups=(
                BiasGroupMetric(group_id="reference", sample_count=500, value=0.81),
                BiasGroupMetric(group_id="evaluated", sample_count=480, value=0.79),
            ),
        ),
    ),
    thresholds=(
        BiasThreshold(
            metric_id="selection-rate",
            minimum_group_samples=100,
            maximum_absolute_gap=0.1,
            minimum_parity_ratio=0.8,
        ),
    ),
)
report = verifier.require_bias(request)
```

Use `BiasExceptionGrant` only for a narrow evaluation/metric/group combination.
Exceptions expire and are ignored unless an application-owned
`BiasExceptionVerifier` authenticates the exact grant. The included
`StaticBiasExceptionVerifier` is suitable for tests or grants loaded from
protected state; production deployments can verify signed approval records.

## Assumptions, limitations, and residual risk

- Digests prove equality, not label correctness or freedom from bias. Keep the
  approved digests, identities, policies, and exception approvals outside the
  data/model trust boundary.
- Annotation confidence is a claim from the label producer. Calibrate it against
  held-out human-reviewed data and use independent consistency evaluators.
- Clean-label poisoning and bias evaluation are application-specific. Hooks and
  group metrics can miss coordinated, semantic, intersectional, or delayed harm.
- Aggregate reports can still create privacy risk for small groups. Set meaningful
  `minimum_group_samples`, restrict report access, and apply disclosure controls.
- Absolute gap and parity ratio are generic comparisons, not legal fairness
  determinations. Select domain-appropriate outcomes, qualified reviewers, and
  documented thresholds.
- A valid exception records governance approval; it does not make a disparity
  safe. Keep exceptions narrow, short-lived, monitored, and independently
  reviewed.
- `DataPoisoningVerifier`, supply-chain verification, lifecycle/consent controls,
  held-out adversarial evaluation, rollback, deletion, and production monitoring
  remain necessary around this admission boundary.

These APIs are engineering controls aligned with the
[OWASP AISVS C1 Training Data Integrity and Traceability](https://github.com/OWASP/AISVS/blob/main/1.0/en/0x10-C01-Training-Data-Integrity-and-Traceability.md);
they are not certification or proof that a dataset or model is unbiased.
