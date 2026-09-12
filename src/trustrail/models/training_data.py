"""Typed training-data labeling integrity and bias evaluation models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from trustrail.models.enums import GuardAction, Severity
from trustrail.models.poisoning import DataTransformation
from trustrail.models.supply_chain import ArtifactDigest

SafeIdentifier = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"),
]


def json_artifact_digest(value: JsonValue) -> ArtifactDigest:
    """Digest a JSON label using a deterministic canonical representation."""
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return ArtifactDigest.from_bytes(payload)


class FeatureSensitivity(StrEnum):
    """Sensitivity assigned to one approved dataset feature."""

    NON_SENSITIVE = "non_sensitive"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    HIGHLY_SENSITIVE = "highly_sensitive"


class SensitiveFieldHandling(StrEnum):
    """Handling applied before a feature reaches a training artifact."""

    PLAIN = "plain"
    EXCLUDED = "excluded"
    REDACTED = "redacted"
    ANONYMIZED = "anonymized"
    ENCRYPTED = "encrypted"


class LabelOrigin(StrEnum):
    """Origin of an annotation."""

    HUMAN = "human"
    AUTOMATED = "automated"
    SYNTHETIC = "synthetic"


class TrainingDataCode(StrEnum):
    """Stable, content-free training-data governance outcomes."""

    DATASET_IDENTITY_MISMATCH = "dataset_identity_mismatch"
    PURPOSE_NOT_APPROVED = "purpose_not_approved"
    FEATURE_NOT_APPROVED = "feature_not_approved"
    REQUIRED_FEATURE_MISSING = "required_feature_missing"
    SENSITIVE_HANDLING_MISMATCH = "sensitive_handling_mismatch"
    SENSITIVE_LABEL_HANDLING_MISMATCH = "sensitive_label_handling_mismatch"
    TRANSFORMATION_NOT_APPROVED = "transformation_not_approved"
    TRANSFORMATION_LINEAGE_BROKEN = "transformation_lineage_broken"
    ANNOTATION_LIMIT_EXCEEDED = "annotation_limit_exceeded"
    LABEL_WRITER_NOT_AUTHORIZED = "label_writer_not_authorized"
    LABEL_APPROVER_REQUIRED = "label_approver_required"
    LABEL_APPROVER_NOT_AUTHORIZED = "label_approver_not_authorized"
    LABEL_SELF_APPROVAL = "label_self_approval"
    LABEL_INTEGRITY_MISMATCH = "label_integrity_mismatch"
    ANNOTATION_SET_INTEGRITY_MISMATCH = "annotation_set_integrity_mismatch"
    LABEL_CONFIDENCE_MISSING = "label_confidence_missing"
    LABEL_CONFIDENCE_BELOW_THRESHOLD = "label_confidence_below_threshold"
    QUALITY_EVALUATOR_MISSING = "quality_evaluator_missing"
    QUALITY_EVALUATOR_FAILED = "quality_evaluator_failed"
    LABEL_CONSISTENCY_BELOW_THRESHOLD = "label_consistency_below_threshold"
    CLEAN_LABEL_POISONING_SUSPECTED = "clean_label_poisoning_suspected"


class BiasEvaluationCode(StrEnum):
    """Stable, content-free bias evaluation outcomes."""

    METRIC_THRESHOLD_MISSING = "metric_threshold_missing"
    GROUP_SAMPLE_TOO_SMALL = "group_sample_too_small"
    BIAS_THRESHOLD_EXCEEDED = "bias_threshold_exceeded"
    EXCEPTION_UNVERIFIED = "exception_unverified"
    EXCEPTION_EXPIRED = "exception_expired"


class DatasetFeaturePolicy(BaseModel):
    """Control-plane approval for one necessary dataset feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: SafeIdentifier
    required: bool = False
    sensitivity: FeatureSensitivity = FeatureSensitivity.NON_SENSITIVE
    allowed_handling: frozenset[SensitiveFieldHandling] = Field(
        default_factory=lambda: frozenset({SensitiveFieldHandling.PLAIN}),
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_sensitive_handling(self) -> DatasetFeaturePolicy:
        if (
            self.sensitivity != FeatureSensitivity.NON_SENSITIVE
            and SensitiveFieldHandling.PLAIN in self.allowed_handling
        ):
            raise ValueError("Sensitive features cannot approve plain handling")
        return self


class DatasetFeatureUse(BaseModel):
    """Feature and handling declared by a candidate dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_id: SafeIdentifier
    handling: SensitiveFieldHandling


class DatasetTransformationPolicy(BaseModel):
    """Approved transformation implementation and actor identities."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: SafeIdentifier
    allowed_versions: frozenset[SafeIdentifier] = Field(min_length=1)
    authorized_actors: frozenset[SafeIdentifier] = Field(min_length=1)


class TrainingDatasetPolicy(BaseModel):
    """Trusted labeling, feature, quality, and integrity policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: SafeIdentifier
    dataset_id: SafeIdentifier
    dataset_version: SafeIdentifier
    intended_purposes: frozenset[SafeIdentifier] = Field(min_length=1)
    features: tuple[DatasetFeaturePolicy, ...] = Field(min_length=1)
    transformations: tuple[DatasetTransformationPolicy, ...] = ()
    authorized_label_writers: frozenset[SafeIdentifier] = Field(min_length=1)
    authorized_label_approvers: frozenset[SafeIdentifier] = Field(min_length=1)
    label_sensitivity: FeatureSensitivity = FeatureSensitivity.NON_SENSITIVE
    allowed_label_handling: frozenset[SensitiveFieldHandling] = Field(
        default_factory=lambda: frozenset({SensitiveFieldHandling.PLAIN}),
        min_length=1,
    )
    approved_annotation_set_digest: ArtifactDigest
    require_distinct_approver: bool = True
    minimum_automated_label_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    require_quality_evaluator: bool = True
    minimum_consistency_score: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    maximum_clean_label_poisoning_score: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    max_annotations: int = Field(default=1_000_000, ge=1, le=100_000_000)

    @model_validator(mode="after")
    def validate_unique_policy_ids(self) -> TrainingDatasetPolicy:
        feature_ids = [feature.feature_id for feature in self.features]
        transformation_names = [item.name for item in self.transformations]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("Training dataset policy contains duplicate feature IDs")
        if len(transformation_names) != len(set(transformation_names)):
            raise ValueError("Training dataset policy contains duplicate transformation names")
        if (
            self.label_sensitivity != FeatureSensitivity.NON_SENSITIVE
            and SensitiveFieldHandling.PLAIN in self.allowed_label_handling
        ):
            raise ValueError("Sensitive labels cannot approve plain handling")
        return self


class TrainingDatasetManifest(BaseModel):
    """Candidate dataset inventory and integrity-linked processing history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: SafeIdentifier
    dataset_version: SafeIdentifier
    intended_purpose: SafeIdentifier
    features: tuple[DatasetFeatureUse, ...] = Field(min_length=1)
    source_digest: ArtifactDigest
    output_digest: ArtifactDigest
    transformations: tuple[DataTransformation, ...] = ()

    @model_validator(mode="after")
    def validate_unique_features(self) -> TrainingDatasetManifest:
        feature_ids = [feature.feature_id for feature in self.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("Training dataset manifest contains duplicate feature IDs")
        return self


class TrainingAnnotation(BaseModel):
    """Annotation with private label content and integrity/authorization metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    annotation_id: SafeIdentifier
    sample_ref: ArtifactDigest
    label: JsonValue = Field(repr=False)
    label_digest: ArtifactDigest
    origin: LabelOrigin
    writer_id: SafeIdentifier
    approver_id: SafeIdentifier | None = None
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    label_handling: SensitiveFieldHandling = SensitiveFieldHandling.PLAIN

    @classmethod
    def from_label(
        cls,
        *,
        annotation_id: str,
        sample_ref: ArtifactDigest,
        label: JsonValue,
        origin: LabelOrigin,
        writer_id: str,
        approver_id: str | None = None,
        confidence: float | None = None,
        label_handling: SensitiveFieldHandling = SensitiveFieldHandling.PLAIN,
    ) -> TrainingAnnotation:
        """Create an annotation whose digest is calculated from its canonical label."""
        return cls(
            annotation_id=annotation_id,
            sample_ref=sample_ref,
            label=label,
            label_digest=json_artifact_digest(label),
            origin=origin,
            writer_id=writer_id,
            approver_id=approver_id,
            confidence=confidence,
            label_handling=label_handling,
        )


class TrainingDatasetSubmission(BaseModel):
    """Dataset manifest and private annotations presented for admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest: TrainingDatasetManifest
    annotations: tuple[TrainingAnnotation, ...] = Field(min_length=1)
    observed_annotation_set_digest: ArtifactDigest

    @model_validator(mode="after")
    def validate_unique_annotations(self) -> TrainingDatasetSubmission:
        annotation_ids = [annotation.annotation_id for annotation in self.annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("Training dataset submission contains duplicate annotation IDs")
        return self

    @classmethod
    def from_annotations(
        cls,
        *,
        manifest: TrainingDatasetManifest,
        annotations: tuple[TrainingAnnotation, ...],
    ) -> TrainingDatasetSubmission:
        """Build a submission with an integrity digest over ordered annotation metadata."""
        return cls(
            manifest=manifest,
            annotations=annotations,
            observed_annotation_set_digest=annotation_set_digest(annotations),
        )


def annotation_set_digest(annotations: tuple[TrainingAnnotation, ...]) -> ArtifactDigest:
    """Digest annotation identities, label hashes, provenance, and approvals."""
    values = [
        {
            "annotation_id": item.annotation_id,
            "sample_ref": item.sample_ref.model_dump(mode="json"),
            "label_digest": item.label_digest.model_dump(mode="json"),
            "origin": item.origin.value,
            "writer_id": item.writer_id,
            "approver_id": item.approver_id,
            "confidence": item.confidence,
            "label_handling": item.label_handling.value,
        }
        for item in annotations
    ]
    encoded = json.dumps(values, separators=(",", ":"), sort_keys=True).encode()
    return ArtifactDigest.from_bytes(encoded)


class LabelQualityAssessment(BaseModel):
    """Provider-neutral scores returned by an application quality hook."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    consistency_score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    clean_label_poisoning_score: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    evidence_ref: ArtifactDigest | None = None


class TrainingDataFinding(BaseModel):
    """Content-free explanation of a dataset admission failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: TrainingDataCode
    severity: Severity
    scope_id: SafeIdentifier
    evaluator_id: SafeIdentifier
    message: str


class TrainingDataEvidence(BaseModel):
    """Content-safe evidence for one governed dataset submission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["trustrail.training-data-evidence.v1"] = (
        "trustrail.training-data-evidence.v1"
    )
    policy_id: SafeIdentifier
    dataset_id: SafeIdentifier
    dataset_version: SafeIdentifier
    intended_purpose: SafeIdentifier
    output_digest: ArtifactDigest
    annotation_set_digest: ArtifactDigest
    features: tuple[DatasetFeatureUse, ...]
    transformation_digests: tuple[ArtifactDigest, ...]
    annotation_count: int = Field(ge=0)
    automated_annotation_count: int = Field(ge=0)
    evaluated_annotation_count: int = Field(ge=0)
    evaluated_at: AwareDatetime
    finding_codes: tuple[TrainingDataCode, ...]


class TrainingDataGovernanceResult(BaseModel):
    """Final dataset admission decision with content-safe evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: GuardAction
    findings: tuple[TrainingDataFinding, ...]
    evidence: TrainingDataEvidence

    @property
    def is_allowed(self) -> bool:
        return self.action == GuardAction.ALLOW

    @property
    def is_quarantined(self) -> bool:
        return self.action in (GuardAction.BLOCK, GuardAction.QUARANTINE)


class BiasGroupMetric(BaseModel):
    """Aggregate metric for one reference or evaluated group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: SafeIdentifier
    sample_count: int = Field(ge=0)
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class BiasMetricInput(BaseModel):
    """Provider-neutral group measurements for one bias metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: SafeIdentifier
    metric_kind: SafeIdentifier
    reference_group_id: SafeIdentifier
    groups: tuple[BiasGroupMetric, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_groups(self) -> BiasMetricInput:
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("Bias metric contains duplicate group IDs")
        if self.reference_group_id not in group_ids:
            raise ValueError("Bias metric reference group is missing")
        return self


class BiasThreshold(BaseModel):
    """Allowed disparity and minimum statistical support for a metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: SafeIdentifier
    minimum_group_samples: int = Field(default=30, ge=1)
    maximum_absolute_gap: float = Field(default=0.1, ge=0.0, le=1.0)
    minimum_parity_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class BiasExceptionGrant(BaseModel):
    """Narrow, expiring exception approved outside the evaluated system."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exception_id: SafeIdentifier
    evaluation_id: SafeIdentifier
    metric_id: SafeIdentifier
    group_id: SafeIdentifier
    justification_ref: ArtifactDigest
    approval_ref: ArtifactDigest
    expires_at: AwareDatetime


class BiasEvaluationRequest(BaseModel):
    """Bias metrics, thresholds, and candidate exceptions for one subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_id: SafeIdentifier
    subject_ref: ArtifactDigest
    metrics: tuple[BiasMetricInput, ...] = Field(min_length=1)
    thresholds: tuple[BiasThreshold, ...] = Field(min_length=1)
    exceptions: tuple[BiasExceptionGrant, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> BiasEvaluationRequest:
        metric_ids = [metric.metric_id for metric in self.metrics]
        threshold_ids = [threshold.metric_id for threshold in self.thresholds]
        exception_ids = [item.exception_id for item in self.exceptions]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("Bias evaluation contains duplicate metric IDs")
        if len(threshold_ids) != len(set(threshold_ids)):
            raise ValueError("Bias evaluation contains duplicate threshold metric IDs")
        if len(exception_ids) != len(set(exception_ids)):
            raise ValueError("Bias evaluation contains duplicate exception IDs")
        groups_by_metric = {
            metric.metric_id: {group.group_id for group in metric.groups} for metric in self.metrics
        }
        if any(
            exception.evaluation_id != self.evaluation_id
            or exception.metric_id not in groups_by_metric
            or exception.group_id not in groups_by_metric[exception.metric_id]
            for exception in self.exceptions
        ):
            raise ValueError("Bias exceptions must bind to this evaluation, metric, and group")
        return self


class BiasFinding(BaseModel):
    """Content-free bias threshold or exception finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: BiasEvaluationCode
    severity: Severity
    metric_id: SafeIdentifier
    group_id: SafeIdentifier | None = None
    message: str


class BiasGroupEvidence(BaseModel):
    """Aggregate group comparison safe for an evidence report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: SafeIdentifier
    sample_count: int
    value: float
    absolute_gap: float
    parity_ratio: float
    exception_id: SafeIdentifier | None = None


class BiasMetricEvidence(BaseModel):
    """Evidence for all group comparisons under one metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_id: SafeIdentifier
    metric_kind: SafeIdentifier
    reference_group_id: SafeIdentifier
    groups: tuple[BiasGroupEvidence, ...]


class BiasEvidenceReport(BaseModel):
    """Provider-neutral, aggregate bias evaluation evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["trustrail.bias-evidence.v1"] = "trustrail.bias-evidence.v1"
    evaluation_id: SafeIdentifier
    subject_ref: ArtifactDigest
    evaluated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    action: GuardAction
    thresholds: tuple[BiasThreshold, ...]
    metrics: tuple[BiasMetricEvidence, ...]
    findings: tuple[BiasFinding, ...]

    @property
    def passed(self) -> bool:
        return self.action == GuardAction.ALLOW
