"""Fail-closed training-data labeling integrity and bias evaluation controls."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from typing import Protocol

from trustrail.exceptions import BiasEvaluationError, TrainingDataGovernanceError
from trustrail.models.enums import GuardAction, Severity
from trustrail.models.supply_chain import ArtifactDigest
from trustrail.models.training_data import (
    BiasEvaluationCode,
    BiasEvaluationRequest,
    BiasEvidenceReport,
    BiasExceptionGrant,
    BiasFinding,
    BiasGroupEvidence,
    BiasMetricEvidence,
    DatasetFeaturePolicy,
    LabelOrigin,
    LabelQualityAssessment,
    TrainingAnnotation,
    TrainingDataCode,
    TrainingDataEvidence,
    TrainingDataFinding,
    TrainingDataGovernanceResult,
    TrainingDatasetManifest,
    TrainingDatasetPolicy,
    TrainingDatasetSubmission,
    annotation_set_digest,
    json_artifact_digest,
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class LabelQualityEvaluator(Protocol):
    """Application hook for consistency and clean-label poisoning analysis."""

    evaluator_id: str

    def evaluate(
        self,
        annotation: TrainingAnnotation,
        manifest: TrainingDatasetManifest,
    ) -> LabelQualityAssessment:
        """Return normalized quality scores without placing label content in evidence."""
        ...


class BiasExceptionVerifier(Protocol):
    """Authenticate a narrow bias exception outside model-controlled state."""

    def verify_exception(self, grant: BiasExceptionGrant) -> bool:
        """Return whether trusted application state approved the exact grant."""
        ...


class StaticBiasExceptionVerifier:
    """Exact-match exception verifier for tests and protected application state."""

    def __init__(self, grants: Iterable[BiasExceptionGrant]) -> None:
        self._grants = tuple(grant.model_copy(deep=True) for grant in grants)

    def verify_exception(self, grant: BiasExceptionGrant) -> bool:
        """Accept only an exact allowlisted exception."""
        return any(grant == expected for expected in self._grants)


class TrainingDataGovernanceVerifier:
    """Verify minimized features, labels, transformations, quality, and bias evidence."""

    def __init__(
        self,
        policy: TrainingDatasetPolicy,
        *,
        quality_evaluators: Iterable[LabelQualityEvaluator] = (),
        bias_exception_verifier: BiasExceptionVerifier | None = None,
    ) -> None:
        self._policy = policy.model_copy(deep=True)
        self._features = {feature.feature_id: feature for feature in self._policy.features}
        self._transformations = {item.name: item for item in self._policy.transformations}
        self._quality_evaluators = tuple(quality_evaluators)
        evaluator_ids = [evaluator.evaluator_id for evaluator in self._quality_evaluators]
        if any(not _SAFE_ID_RE.fullmatch(evaluator_id) for evaluator_id in evaluator_ids):
            raise ValueError("Quality evaluator IDs must be content-safe identifiers")
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise ValueError("Quality evaluator IDs must be unique")
        self._bias_exception_verifier = bias_exception_verifier

    @property
    def policy(self) -> TrainingDatasetPolicy:
        """Return a defensive copy of the trusted policy."""
        return self._policy.model_copy(deep=True)

    def verify_dataset(
        self,
        submission: TrainingDatasetSubmission,
        *,
        evaluated_at: datetime | None = None,
    ) -> TrainingDataGovernanceResult:
        """Evaluate one dataset without copying labels into findings or evidence."""
        findings = self._manifest_findings(submission.manifest)
        findings.extend(self._annotation_set_findings(submission))
        evaluated_ids: set[str] = set()
        for annotation in submission.annotations[: self._policy.max_annotations]:
            annotation_findings, evaluated = self._annotation_findings(
                annotation,
                submission.manifest,
            )
            findings.extend(annotation_findings)
            if evaluated:
                evaluated_ids.add(annotation.annotation_id)

        if len(submission.annotations) > self._policy.max_annotations:
            findings.append(
                self._finding(
                    TrainingDataCode.ANNOTATION_LIMIT_EXCEEDED,
                    submission.manifest.dataset_id,
                    "policy",
                    "Dataset exceeds the configured annotation evaluation limit",
                )
            )

        action = GuardAction.QUARANTINE if findings else GuardAction.ALLOW
        evidence = TrainingDataEvidence(
            policy_id=self._policy.policy_id,
            dataset_id=submission.manifest.dataset_id,
            dataset_version=submission.manifest.dataset_version,
            intended_purpose=submission.manifest.intended_purpose,
            output_digest=submission.manifest.output_digest,
            annotation_set_digest=submission.observed_annotation_set_digest,
            features=tuple(sorted(submission.manifest.features, key=lambda item: item.feature_id)),
            transformation_digests=tuple(
                ArtifactDigest.from_bytes(item.model_dump_json().encode())
                for item in submission.manifest.transformations
            ),
            annotation_count=len(submission.annotations),
            automated_annotation_count=sum(
                item.origin in (LabelOrigin.AUTOMATED, LabelOrigin.SYNTHETIC)
                for item in submission.annotations
            ),
            evaluated_annotation_count=len(evaluated_ids),
            evaluated_at=self._evaluation_time(evaluated_at),
            finding_codes=tuple(finding.code for finding in findings),
        )
        return TrainingDataGovernanceResult(
            action=action,
            findings=tuple(findings),
            evidence=evidence,
        )

    def require_dataset(
        self,
        submission: TrainingDatasetSubmission,
    ) -> TrainingDatasetSubmission:
        """Return an accepted submission or raise before training starts."""
        result = self.verify_dataset(submission)
        if result.is_quarantined:
            raise TrainingDataGovernanceError(result)
        return submission

    def evaluate_bias(
        self,
        request: BiasEvaluationRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> BiasEvidenceReport:
        """Apply provider-neutral group thresholds and authenticated exceptions."""
        now = self._evaluation_time(evaluated_at)
        thresholds = {threshold.metric_id: threshold for threshold in request.thresholds}
        findings: list[BiasFinding] = []
        metric_evidence: list[BiasMetricEvidence] = []
        for metric in request.metrics:
            threshold = thresholds.get(metric.metric_id)
            if threshold is None:
                findings.append(
                    self._bias_finding(
                        BiasEvaluationCode.METRIC_THRESHOLD_MISSING,
                        metric.metric_id,
                        None,
                        "Metric has no trusted bias threshold",
                    )
                )
                continue
            reference = next(
                group for group in metric.groups if group.group_id == metric.reference_group_id
            )
            groups: list[BiasGroupEvidence] = []
            for group in metric.groups:
                gap = abs(group.value - reference.value)
                parity_ratio = self._parity_ratio(group.value, reference.value)
                exception_id: str | None = None
                if (
                    group.sample_count < threshold.minimum_group_samples
                    or reference.sample_count < threshold.minimum_group_samples
                ):
                    findings.append(
                        self._bias_finding(
                            BiasEvaluationCode.GROUP_SAMPLE_TOO_SMALL,
                            metric.metric_id,
                            group.group_id,
                            "Group does not meet the minimum aggregate sample size",
                        )
                    )
                exceeded = group.group_id != reference.group_id and (
                    gap > threshold.maximum_absolute_gap
                    or (
                        threshold.minimum_parity_ratio is not None
                        and parity_ratio < threshold.minimum_parity_ratio
                    )
                )
                if exceeded:
                    grant = self._matching_exception(request, metric.metric_id, group.group_id)
                    exception_id, exception_findings = self._apply_exception(
                        grant,
                        metric.metric_id,
                        group.group_id,
                        now,
                    )
                    findings.extend(exception_findings)
                    if exception_id is None:
                        findings.append(
                            self._bias_finding(
                                BiasEvaluationCode.BIAS_THRESHOLD_EXCEEDED,
                                metric.metric_id,
                                group.group_id,
                                "Group metric exceeds the approved disparity threshold",
                            )
                        )
                groups.append(
                    BiasGroupEvidence(
                        group_id=group.group_id,
                        sample_count=group.sample_count,
                        value=group.value,
                        absolute_gap=gap,
                        parity_ratio=parity_ratio,
                        exception_id=exception_id,
                    )
                )
            metric_evidence.append(
                BiasMetricEvidence(
                    metric_id=metric.metric_id,
                    metric_kind=metric.metric_kind,
                    reference_group_id=metric.reference_group_id,
                    groups=tuple(groups),
                )
            )

        return BiasEvidenceReport(
            evaluation_id=request.evaluation_id,
            subject_ref=request.subject_ref,
            evaluated_at=now,
            action=GuardAction.BLOCK if findings else GuardAction.ALLOW,
            thresholds=request.thresholds,
            metrics=tuple(metric_evidence),
            findings=tuple(findings),
        )

    def require_bias(self, request: BiasEvaluationRequest) -> BiasEvidenceReport:
        """Return passing bias evidence or raise before model approval."""
        report = self.evaluate_bias(request)
        if not report.passed:
            raise BiasEvaluationError(report)
        return report

    def _manifest_findings(
        self,
        manifest: TrainingDatasetManifest,
    ) -> list[TrainingDataFinding]:
        findings: list[TrainingDataFinding] = []
        if (
            manifest.dataset_id != self._policy.dataset_id
            or manifest.dataset_version != self._policy.dataset_version
        ):
            findings.append(
                self._finding(
                    TrainingDataCode.DATASET_IDENTITY_MISMATCH,
                    manifest.dataset_id,
                    "policy",
                    "Dataset identity or immutable version differs from policy",
                )
            )
        if manifest.intended_purpose not in self._policy.intended_purposes:
            findings.append(
                self._finding(
                    TrainingDataCode.PURPOSE_NOT_APPROVED,
                    manifest.dataset_id,
                    "policy",
                    "Dataset purpose is not approved",
                )
            )
        observed_features = {feature.feature_id: feature for feature in manifest.features}
        for feature in manifest.features:
            approved = self._features.get(feature.feature_id)
            if approved is None:
                findings.append(
                    self._finding(
                        TrainingDataCode.FEATURE_NOT_APPROVED,
                        feature.feature_id,
                        "feature_policy",
                        "Dataset contains a feature not required for the approved purpose",
                    )
                )
            elif feature.handling not in approved.allowed_handling:
                findings.append(self._sensitive_handling_finding(feature.feature_id, approved))
        for required_feature in self._policy.features:
            if required_feature.required and required_feature.feature_id not in observed_features:
                findings.append(
                    self._finding(
                        TrainingDataCode.REQUIRED_FEATURE_MISSING,
                        required_feature.feature_id,
                        "feature_policy",
                        "Dataset lacks a feature required by the approved schema",
                    )
                )
        findings.extend(self._transformation_findings(manifest))
        return findings

    def _sensitive_handling_finding(
        self,
        feature_id: str,
        feature: DatasetFeaturePolicy,
    ) -> TrainingDataFinding:
        del feature
        return self._finding(
            TrainingDataCode.SENSITIVE_HANDLING_MISMATCH,
            feature_id,
            "feature_policy",
            "Feature handling does not match the approved sensitivity policy",
        )

    def _transformation_findings(
        self,
        manifest: TrainingDatasetManifest,
    ) -> list[TrainingDataFinding]:
        findings: list[TrainingDataFinding] = []
        for transformation in manifest.transformations:
            approved = self._transformations.get(transformation.name)
            if (
                approved is None
                or transformation.version not in approved.allowed_versions
                or transformation.actor_id not in approved.authorized_actors
            ):
                findings.append(
                    self._finding(
                        TrainingDataCode.TRANSFORMATION_NOT_APPROVED,
                        manifest.dataset_id,
                        "transformation_policy",
                        "Dataset transformation identity, version, or actor is not approved",
                    )
                )
        lineage_broken = False
        if manifest.transformations:
            lineage_broken = not manifest.source_digest.matches(
                manifest.transformations[0].input_digest
            ) or not manifest.transformations[-1].output_digest.matches(manifest.output_digest)
            lineage_broken = lineage_broken or any(
                not previous.output_digest.matches(current.input_digest)
                for previous, current in pairwise(manifest.transformations)
            )
        elif not manifest.source_digest.matches(manifest.output_digest):
            lineage_broken = True
        if lineage_broken:
            findings.append(
                self._finding(
                    TrainingDataCode.TRANSFORMATION_LINEAGE_BROKEN,
                    manifest.dataset_id,
                    "transformation_integrity",
                    "Dataset transformation lineage contains an integrity gap",
                )
            )
        return findings

    def _annotation_set_findings(
        self,
        submission: TrainingDatasetSubmission,
    ) -> list[TrainingDataFinding]:
        calculated = annotation_set_digest(submission.annotations)
        if calculated.matches(submission.observed_annotation_set_digest) and calculated.matches(
            self._policy.approved_annotation_set_digest
        ):
            return []
        return [
            self._finding(
                TrainingDataCode.ANNOTATION_SET_INTEGRITY_MISMATCH,
                submission.manifest.dataset_id,
                "annotation_set_integrity",
                "Annotation set differs from its observed or approved digest",
            )
        ]

    def _annotation_findings(
        self,
        annotation: TrainingAnnotation,
        manifest: TrainingDatasetManifest,
    ) -> tuple[list[TrainingDataFinding], bool]:
        findings: list[TrainingDataFinding] = []
        if annotation.writer_id not in self._policy.authorized_label_writers:
            findings.append(
                self._finding(
                    TrainingDataCode.LABEL_WRITER_NOT_AUTHORIZED,
                    annotation.annotation_id,
                    "annotation_authorization",
                    "Annotation writer is not authorized",
                )
            )
        if annotation.approver_id is None:
            findings.append(
                self._finding(
                    TrainingDataCode.LABEL_APPROVER_REQUIRED,
                    annotation.annotation_id,
                    "annotation_authorization",
                    "Annotation lacks required approval",
                )
            )
        elif annotation.approver_id not in self._policy.authorized_label_approvers:
            findings.append(
                self._finding(
                    TrainingDataCode.LABEL_APPROVER_NOT_AUTHORIZED,
                    annotation.annotation_id,
                    "annotation_authorization",
                    "Annotation approver is not authorized",
                )
            )
        if (
            self._policy.require_distinct_approver
            and annotation.approver_id == annotation.writer_id
        ):
            findings.append(
                self._finding(
                    TrainingDataCode.LABEL_SELF_APPROVAL,
                    annotation.annotation_id,
                    "annotation_authorization",
                    "Annotation writer cannot approve the same label",
                )
            )
        try:
            label_integrity_valid = annotation.label_digest.matches(
                json_artifact_digest(annotation.label)
            )
        except (TypeError, ValueError):
            label_integrity_valid = False
        if not label_integrity_valid:
            findings.append(
                self._finding(
                    TrainingDataCode.LABEL_INTEGRITY_MISMATCH,
                    annotation.annotation_id,
                    "label_integrity",
                    "Annotation label differs from its integrity digest",
                )
            )
        if annotation.label_handling not in self._policy.allowed_label_handling:
            findings.append(
                self._finding(
                    TrainingDataCode.SENSITIVE_LABEL_HANDLING_MISMATCH,
                    annotation.annotation_id,
                    "label_sensitivity_policy",
                    "Label handling does not match the approved sensitivity policy",
                )
            )
        if annotation.origin in (LabelOrigin.AUTOMATED, LabelOrigin.SYNTHETIC):
            if annotation.confidence is None:
                findings.append(
                    self._finding(
                        TrainingDataCode.LABEL_CONFIDENCE_MISSING,
                        annotation.annotation_id,
                        "label_confidence",
                        "Automated annotation lacks a confidence score",
                    )
                )
            elif annotation.confidence < self._policy.minimum_automated_label_confidence:
                findings.append(
                    self._finding(
                        TrainingDataCode.LABEL_CONFIDENCE_BELOW_THRESHOLD,
                        annotation.annotation_id,
                        "label_confidence",
                        "Automated annotation confidence is below policy",
                    )
                )
        quality_findings, evaluated = self._quality_findings(annotation, manifest)
        findings.extend(quality_findings)
        return findings, evaluated

    def _quality_findings(
        self,
        annotation: TrainingAnnotation,
        manifest: TrainingDatasetManifest,
    ) -> tuple[list[TrainingDataFinding], bool]:
        if not self._quality_evaluators:
            if not self._policy.require_quality_evaluator:
                return [], False
            return (
                [
                    self._finding(
                        TrainingDataCode.QUALITY_EVALUATOR_MISSING,
                        annotation.annotation_id,
                        "quality_policy",
                        "Required annotation quality evaluation is unavailable",
                    )
                ],
                False,
            )
        findings: list[TrainingDataFinding] = []
        evaluated = False
        for evaluator in self._quality_evaluators:
            try:
                assessment = LabelQualityAssessment.model_validate(
                    evaluator.evaluate(annotation, manifest)
                )
            except Exception:
                findings.append(
                    self._finding(
                        TrainingDataCode.QUALITY_EVALUATOR_FAILED,
                        annotation.annotation_id,
                        evaluator.evaluator_id,
                        "Annotation quality evaluator failed closed",
                    )
                )
                continue
            evaluated = True
            if assessment.consistency_score < self._policy.minimum_consistency_score:
                findings.append(
                    self._finding(
                        TrainingDataCode.LABEL_CONSISTENCY_BELOW_THRESHOLD,
                        annotation.annotation_id,
                        evaluator.evaluator_id,
                        "Annotation consistency is below policy",
                    )
                )
            if (
                assessment.clean_label_poisoning_score
                > self._policy.maximum_clean_label_poisoning_score
            ):
                findings.append(
                    self._finding(
                        TrainingDataCode.CLEAN_LABEL_POISONING_SUSPECTED,
                        annotation.annotation_id,
                        evaluator.evaluator_id,
                        "Annotation exceeded the clean-label poisoning threshold",
                    )
                )
        return findings, evaluated

    @staticmethod
    def _matching_exception(
        request: BiasEvaluationRequest,
        metric_id: str,
        group_id: str,
    ) -> BiasExceptionGrant | None:
        return next(
            (
                grant
                for grant in request.exceptions
                if grant.evaluation_id == request.evaluation_id
                and grant.metric_id == metric_id
                and grant.group_id == group_id
            ),
            None,
        )

    def _apply_exception(
        self,
        grant: BiasExceptionGrant | None,
        metric_id: str,
        group_id: str,
        now: datetime,
    ) -> tuple[str | None, list[BiasFinding]]:
        if grant is None:
            return None, []
        if grant.expires_at <= now:
            return None, [
                self._bias_finding(
                    BiasEvaluationCode.EXCEPTION_EXPIRED,
                    metric_id,
                    group_id,
                    "Bias exception has expired",
                )
            ]
        verified = False
        if self._bias_exception_verifier is not None:
            try:
                verified = self._bias_exception_verifier.verify_exception(grant)
            except Exception:
                verified = False
        if not verified:
            return None, [
                self._bias_finding(
                    BiasEvaluationCode.EXCEPTION_UNVERIFIED,
                    metric_id,
                    group_id,
                    "Bias exception could not be authenticated",
                )
            ]
        return grant.exception_id, []

    @staticmethod
    def _parity_ratio(value: float, reference: float) -> float:
        high = max(value, reference)
        if high == 0.0:
            return 1.0
        return min(value, reference) / high

    @staticmethod
    def _evaluation_time(value: datetime | None) -> datetime:
        evaluated_at = value or datetime.now(tz=UTC)
        if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
            raise ValueError("Evaluation timestamps must be timezone-aware")
        return evaluated_at

    @staticmethod
    def _finding(
        code: TrainingDataCode,
        scope_id: str,
        evaluator_id: str,
        message: str,
    ) -> TrainingDataFinding:
        return TrainingDataFinding(
            code=code,
            severity=Severity.HIGH,
            scope_id=scope_id,
            evaluator_id=evaluator_id,
            message=message,
        )

    @staticmethod
    def _bias_finding(
        code: BiasEvaluationCode,
        metric_id: str,
        group_id: str | None,
        message: str,
    ) -> BiasFinding:
        return BiasFinding(
            code=code,
            severity=Severity.HIGH,
            metric_id=metric_id,
            group_id=group_id,
            message=message,
        )
