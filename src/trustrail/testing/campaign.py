"""Versioned, reproducible AI trustworthiness campaigns and release gates."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from statistics import NormalDist
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from trustrail.models.enums import GuardAction, Severity

_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
SafeIdentifier = Annotated[str, Field(pattern=_IDENTIFIER_PATTERN)]


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class AITestLayer(StrEnum):
    """OWASP AI Testing Guide control layers."""

    APPLICATION = "application"
    MODEL = "model"
    INFRASTRUCTURE = "infrastructure"
    DATA = "data"


class AITestCaseKind(StrEnum):
    """Semantic role of a corpus case."""

    ATTACK = "attack"
    BENIGN = "benign"
    CONTROL = "control"


class AITestTrialStatus(StrEnum):
    """Content-free outcome of one evaluation trial."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class AITestThresholdMode(StrEnum):
    """Value used when checking configured minimum pass rates."""

    POINT_ESTIMATE = "point_estimate"
    LOWER_CONFIDENCE_BOUND = "lower_confidence_bound"


class AITestGateCode(StrEnum):
    """Stable machine-readable release-gate outcomes."""

    SEED_UNSUPPORTED = "seed_unsupported"
    THRESHOLD_MISSED = "threshold_missed"
    ERROR_BUDGET_EXCEEDED = "error_budget_exceeded"
    CRITICAL_FAILURE = "critical_failure"
    BASELINE_REQUIRED = "baseline_required"
    BASELINE_UNVERIFIED = "baseline_unverified"
    BASELINE_MISMATCH = "baseline_mismatch"
    STATISTICAL_REGRESSION = "statistical_regression"
    SECURITY_REGRESSION = "security_regression"


class AITestCase(BaseModel):
    """One private corpus case; payloads never enter exported evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: SafeIdentifier
    layer: AITestLayer
    kind: AITestCaseKind
    payload: JsonValue
    expected: JsonValue | None = None
    critical: bool = False
    tags: tuple[SafeIdentifier, ...] = ()


class AITestModelIdentity(BaseModel):
    """Pinned provider and model identity evaluated by a campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: SafeIdentifier
    model_id: SafeIdentifier
    model_revision: SafeIdentifier
    configuration_sha256: str = Field(pattern=_DIGEST_PATTERN)
    supports_deterministic_seed: bool = False


class AITestTarget(BaseModel):
    """Versioned application, model, infrastructure, and data under test."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: SafeIdentifier
    application_version: SafeIdentifier
    model: AITestModelIdentity
    infrastructure_revision: SafeIdentifier
    data_revision: SafeIdentifier


class AITestBudget(BaseModel):
    """Hard limits preventing an evaluation from silently exceeding cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_total_trials: int = Field(ge=1)
    max_attack_trials: int = Field(ge=0)
    max_errors: int = Field(default=0, ge=0)


class AITestThresholds(BaseModel):
    """Explicit release thresholds and baseline regression tolerances."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: AITestThresholdMode = AITestThresholdMode.LOWER_CONFIDENCE_BOUND
    minimum_overall_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_attack_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_benign_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_control_pass_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    minimum_layer_pass_rates: dict[AITestLayer, float] = Field(default_factory=dict)
    require_zero_critical_failures: bool = True
    require_approved_baseline: bool = False
    maximum_baseline_pass_rate_drop: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_layer_thresholds(self) -> AITestThresholds:
        if any(value < 0.0 or value > 1.0 for value in self.minimum_layer_pass_rates.values()):
            raise ValueError("Layer pass-rate thresholds must be between zero and one")
        return self


class AITestCampaign(BaseModel):
    """Immutable, versioned corpus and evaluation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["trustrail.ai-test-campaign.v1"] = "trustrail.ai-test-campaign.v1"
    campaign_id: SafeIdentifier
    campaign_version: SafeIdentifier
    corpus_version: SafeIdentifier
    cases: tuple[AITestCase, ...] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, le=10_000)
    base_seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    require_seed_support: bool = False
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    budget: AITestBudget
    thresholds: AITestThresholds = Field(default_factory=AITestThresholds)
    declared_limitations: tuple[SafeIdentifier, ...] = ()

    @model_validator(mode="after")
    def validate_campaign(self) -> AITestCampaign:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("AI test campaign contains duplicate case IDs")
        if self.require_seed_support and self.base_seed is None:
            raise ValueError("A base seed is required when seed support is mandatory")
        total_trials = len(self.cases) * self.repetitions
        attack_trials = sum(case.kind == AITestCaseKind.ATTACK for case in self.cases)
        attack_trials *= self.repetitions
        if total_trials > self.budget.max_total_trials:
            raise ValueError("Planned trials exceed the total attack budget")
        if attack_trials > self.budget.max_attack_trials:
            raise ValueError("Planned attack trials exceed the attack budget")
        layers = {case.layer for case in self.cases}
        if set(self.thresholds.minimum_layer_pass_rates) - layers:
            raise ValueError("Layer thresholds must reference campaign case layers")
        kinds = {case.kind for case in self.cases}
        kind_thresholds = {
            AITestCaseKind.ATTACK: self.thresholds.minimum_attack_pass_rate,
            AITestCaseKind.BENIGN: self.thresholds.minimum_benign_pass_rate,
            AITestCaseKind.CONTROL: self.thresholds.minimum_control_pass_rate,
        }
        if any(
            minimum is not None and kind not in kinds for kind, minimum in kind_thresholds.items()
        ):
            raise ValueError("Kind thresholds must reference campaign case kinds")
        return self

    @property
    def configuration_sha256(self) -> str:
        """Digest all campaign controls while excluding private case content."""
        controls = self.model_dump(mode="json", exclude={"cases"})
        return _canonical_digest(controls)

    @property
    def corpus_sha256(self) -> str:
        """Digest the full private corpus for change detection without disclosure."""
        cases = [case.model_dump(mode="json") for case in self.cases]
        return _canonical_digest(cases)

    @classmethod
    def from_path(cls, path: str | Path) -> AITestCampaign:
        """Load and strictly validate a UTF-8 JSON campaign."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


class AITestTrialContext(BaseModel):
    """Reproducibility metadata supplied to an evaluator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_id: SafeIdentifier
    campaign_version: SafeIdentifier
    corpus_version: SafeIdentifier
    repetition: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class AITestObservation(BaseModel):
    """Evaluator outcome; references must be digests, never raw model output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: bool
    score: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    outcome_ref: str | None = Field(default=None, pattern=_DIGEST_PATTERN)
    limitations: tuple[SafeIdentifier, ...] = ()


class AITestEvaluator(Protocol):
    """Application adapter used to evaluate private cases."""

    def evaluate(self, case: AITestCase, context: AITestTrialContext) -> AITestObservation:
        """Evaluate a case and return only a content-safe outcome."""
        ...


class AITestTrialEvidence(BaseModel):
    """Payload-free evidence for a single trial."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: SafeIdentifier
    layer: AITestLayer
    kind: AITestCaseKind
    critical: bool
    repetition: int
    seed: int | None = None
    status: AITestTrialStatus
    score: float | None = None
    outcome_ref: str | None = None


class AITestMetric(BaseModel):
    """Pass counts and Wilson confidence interval for one scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trials: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    errors: int = Field(ge=0)
    pass_rate: float = Field(ge=0.0, le=1.0)
    confidence_low: float = Field(ge=0.0, le=1.0)
    confidence_high: float = Field(ge=0.0, le=1.0)


class AITestScopedMetric(BaseModel):
    """A metric bound to a stable layer, kind, or case identifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope_id: SafeIdentifier
    metric: AITestMetric


class AITestEvidence(BaseModel):
    """Machine-readable, integrity-bound, content-safe campaign evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["trustrail.ai-test-evidence.v1"] = "trustrail.ai-test-evidence.v1"
    campaign_id: SafeIdentifier
    campaign_version: SafeIdentifier
    corpus_version: SafeIdentifier
    campaign_configuration_sha256: str = Field(pattern=_DIGEST_PATTERN)
    corpus_sha256: str = Field(pattern=_DIGEST_PATTERN)
    target: AITestTarget
    generated_at: datetime
    confidence_level: float
    repetitions: int
    overall: AITestMetric
    by_layer: tuple[AITestScopedMetric, ...]
    by_kind: tuple[AITestScopedMetric, ...]
    by_case: tuple[AITestScopedMetric, ...]
    trials: tuple[AITestTrialEvidence, ...]
    attack_trials_used: int
    limitations: tuple[SafeIdentifier, ...]
    evidence_sha256: str = Field(pattern=_DIGEST_PATTERN)

    @property
    def calculated_sha256(self) -> str:
        """Calculate the digest covering every evidence field except itself."""
        return _canonical_digest(self.model_dump(mode="json", exclude={"evidence_sha256"}))

    @property
    def integrity_valid(self) -> bool:
        """Compare the embedded evidence digest in constant time."""
        return hmac.compare_digest(self.evidence_sha256, self.calculated_sha256)


class ApprovedAITestBaseline(BaseModel):
    """Evidence approved through an application-controlled trust boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_id: SafeIdentifier
    evidence: AITestEvidence
    approval_ref: str = Field(pattern=_DIGEST_PATTERN)
    approver_ref: str = Field(pattern=_DIGEST_PATTERN)
    approved_at: datetime


class AITestBaselineVerifier(Protocol):
    """Authenticate baseline approval outside model-controlled state."""

    def verify_baseline(self, baseline: ApprovedAITestBaseline) -> bool:
        """Return whether the exact baseline was approved."""
        ...


class StaticAITestBaselineVerifier:
    """Exact-match baseline verifier for tests and protected application state."""

    def __init__(self, baselines: Iterable[ApprovedAITestBaseline]) -> None:
        self._baselines = tuple(item.model_copy(deep=True) for item in baselines)

    def verify_baseline(self, baseline: ApprovedAITestBaseline) -> bool:
        """Accept only a byte-equivalent allowlisted baseline object."""
        return baseline.evidence.integrity_valid and any(
            baseline == expected for expected in self._baselines
        )


class AITestGateFinding(BaseModel):
    """Content-free reason a campaign did not pass its release gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: AITestGateCode
    severity: Severity
    scope_id: SafeIdentifier
    message: str


class AITestGateResult(BaseModel):
    """Final campaign release decision and its safe evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: GuardAction
    findings: tuple[AITestGateFinding, ...]
    evidence: AITestEvidence | None = None

    @property
    def passed(self) -> bool:
        return self.action == GuardAction.ALLOW

    def assert_passed(self) -> None:
        """Raise a payload-free CI failure when the candidate is blocked."""
        if not self.passed:
            codes = [finding.code.value for finding in self.findings]
            scopes = [finding.scope_id for finding in self.findings]
            raise AssertionError(
                f"AI trustworthiness release gate failed: codes={codes}, scopes={scopes}"
            )


def _seed_for(campaign: AITestCampaign, case_id: str, repetition: int) -> int | None:
    if campaign.base_seed is None:
        return None
    material = (
        f"{campaign.base_seed}:{campaign.campaign_id}:{campaign.campaign_version}:"
        f"{campaign.corpus_version}:{case_id}:{repetition}"
    )
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def _metric(trials: Iterable[AITestTrialEvidence], confidence_level: float) -> AITestMetric:
    items = tuple(trials)
    passed = sum(item.status == AITestTrialStatus.PASSED for item in items)
    failed = sum(item.status == AITestTrialStatus.FAILED for item in items)
    errors = len(items) - passed - failed
    total = len(items)
    if total == 0:
        return AITestMetric(
            trials=0,
            passed=0,
            failed=0,
            errors=0,
            pass_rate=0.0,
            confidence_low=0.0,
            confidence_high=0.0,
        )
    rate = passed / total
    z_score = NormalDist().inv_cdf(0.5 + confidence_level / 2)
    denominator = 1 + z_score**2 / total
    center = (rate + z_score**2 / (2 * total)) / denominator
    margin = (
        z_score * math.sqrt(rate * (1 - rate) / total + z_score**2 / (4 * total**2))
    ) / denominator
    return AITestMetric(
        trials=total,
        passed=passed,
        failed=failed,
        errors=errors,
        pass_rate=rate,
        confidence_low=max(0.0, center - margin),
        confidence_high=min(1.0, center + margin),
    )


def _scoped_metrics(
    confidence_level: float,
    scopes: Iterable[tuple[str, Iterable[AITestTrialEvidence]]],
) -> tuple[AITestScopedMetric, ...]:
    return tuple(
        AITestScopedMetric(scope_id=scope_id, metric=_metric(items, confidence_level))
        for scope_id, items in scopes
    )


class AITestCampaignRunner:
    """Run deterministic trials and fail closed on threshold or baseline regressions."""

    def __init__(
        self,
        evaluator: AITestEvaluator,
        *,
        baseline_verifier: AITestBaselineVerifier | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._baseline_verifier = baseline_verifier

    def run(
        self,
        campaign: AITestCampaign,
        target: AITestTarget,
        *,
        baseline: ApprovedAITestBaseline | None = None,
        generated_at: datetime | None = None,
    ) -> AITestGateResult:
        """Evaluate a candidate and compare it with an authenticated baseline."""
        preflight = self._preflight_findings(campaign, target, baseline)
        if preflight:
            return AITestGateResult(action=GuardAction.BLOCK, findings=tuple(preflight))

        trials: list[AITestTrialEvidence] = []
        limitations = set(campaign.declared_limitations)
        for case in campaign.cases:
            for repetition in range(campaign.repetitions):
                context = AITestTrialContext(
                    campaign_id=campaign.campaign_id,
                    campaign_version=campaign.campaign_version,
                    corpus_version=campaign.corpus_version,
                    repetition=repetition,
                    seed=_seed_for(campaign, case.case_id, repetition),
                )
                try:
                    observation = AITestObservation.model_validate(
                        self._evaluator.evaluate(case, context)
                    )
                except Exception:
                    trials.append(self._trial_evidence(case, context, None))
                else:
                    limitations.update(observation.limitations)
                    trials.append(self._trial_evidence(case, context, observation))

        evidence = self._build_evidence(
            campaign,
            target,
            tuple(trials),
            tuple(sorted(limitations)),
            generated_at,
        )
        findings = self._threshold_findings(campaign, evidence)
        if baseline is not None:
            findings.extend(self._baseline_findings(campaign, evidence, baseline))
        action = GuardAction.BLOCK if findings else GuardAction.ALLOW
        return AITestGateResult(action=action, findings=tuple(findings), evidence=evidence)

    def _preflight_findings(
        self,
        campaign: AITestCampaign,
        target: AITestTarget,
        baseline: ApprovedAITestBaseline | None,
    ) -> list[AITestGateFinding]:
        findings: list[AITestGateFinding] = []
        if campaign.require_seed_support and not target.model.supports_deterministic_seed:
            findings.append(
                self._finding(
                    AITestGateCode.SEED_UNSUPPORTED,
                    "target:model",
                    "The campaign requires deterministic seed support",
                )
            )
        if campaign.thresholds.require_approved_baseline and baseline is None:
            findings.append(
                self._finding(
                    AITestGateCode.BASELINE_REQUIRED,
                    "baseline",
                    "An approved baseline is required",
                )
            )
        if baseline is not None and not self._baseline_verified(baseline):
            findings.append(
                self._finding(
                    AITestGateCode.BASELINE_UNVERIFIED,
                    baseline.baseline_id,
                    "The supplied baseline could not be authenticated",
                )
            )
        return findings

    def _baseline_verified(self, baseline: ApprovedAITestBaseline) -> bool:
        if self._baseline_verifier is None:
            return False
        try:
            return self._baseline_verifier.verify_baseline(baseline)
        except Exception:
            return False

    @staticmethod
    def _trial_evidence(
        case: AITestCase,
        context: AITestTrialContext,
        observation: AITestObservation | None,
    ) -> AITestTrialEvidence:
        if observation is None:
            status = AITestTrialStatus.ERROR
        else:
            status = AITestTrialStatus.PASSED if observation.passed else AITestTrialStatus.FAILED
        return AITestTrialEvidence(
            case_id=case.case_id,
            layer=case.layer,
            kind=case.kind,
            critical=case.critical,
            repetition=context.repetition,
            seed=context.seed,
            status=status,
            score=None if observation is None else observation.score,
            outcome_ref=None if observation is None else observation.outcome_ref,
        )

    @staticmethod
    def _build_evidence(
        campaign: AITestCampaign,
        target: AITestTarget,
        trials: tuple[AITestTrialEvidence, ...],
        limitations: tuple[str, ...],
        generated_at: datetime | None,
    ) -> AITestEvidence:
        layer_scopes = (
            (layer.value, (trial for trial in trials if trial.layer == layer))
            for layer in AITestLayer
            if any(trial.layer == layer for trial in trials)
        )
        kind_scopes = (
            (kind.value, (trial for trial in trials if trial.kind == kind))
            for kind in AITestCaseKind
            if any(trial.kind == kind for trial in trials)
        )
        case_scopes = (
            (case.case_id, (trial for trial in trials if trial.case_id == case.case_id))
            for case in campaign.cases
        )
        provisional = AITestEvidence(
            campaign_id=campaign.campaign_id,
            campaign_version=campaign.campaign_version,
            corpus_version=campaign.corpus_version,
            campaign_configuration_sha256=campaign.configuration_sha256,
            corpus_sha256=campaign.corpus_sha256,
            target=target,
            generated_at=generated_at or datetime.now(tz=UTC),
            confidence_level=campaign.confidence_level,
            repetitions=campaign.repetitions,
            overall=_metric(trials, campaign.confidence_level),
            by_layer=_scoped_metrics(campaign.confidence_level, layer_scopes),
            by_kind=_scoped_metrics(campaign.confidence_level, kind_scopes),
            by_case=_scoped_metrics(campaign.confidence_level, case_scopes),
            trials=trials,
            attack_trials_used=sum(trial.kind == AITestCaseKind.ATTACK for trial in trials),
            limitations=limitations,
            evidence_sha256=f"sha256:{'0' * 64}",
        )
        return provisional.model_copy(update={"evidence_sha256": provisional.calculated_sha256})

    def _threshold_findings(
        self,
        campaign: AITestCampaign,
        evidence: AITestEvidence,
    ) -> list[AITestGateFinding]:
        findings: list[AITestGateFinding] = []
        if evidence.overall.errors > campaign.budget.max_errors:
            findings.append(
                self._finding(
                    AITestGateCode.ERROR_BUDGET_EXCEEDED,
                    "campaign:overall",
                    "Evaluator errors exceeded the configured budget",
                )
            )
        thresholds: list[tuple[str, AITestMetric, float]] = [
            (
                "campaign:overall",
                evidence.overall,
                campaign.thresholds.minimum_overall_pass_rate,
            )
        ]
        kind_minimums = {
            AITestCaseKind.ATTACK: campaign.thresholds.minimum_attack_pass_rate,
            AITestCaseKind.BENIGN: campaign.thresholds.minimum_benign_pass_rate,
            AITestCaseKind.CONTROL: campaign.thresholds.minimum_control_pass_rate,
        }
        kind_metrics = {item.scope_id: item.metric for item in evidence.by_kind}
        for kind, minimum in kind_minimums.items():
            if minimum is not None and kind.value in kind_metrics:
                thresholds.append((f"kind:{kind.value}", kind_metrics[kind.value], minimum))
        layer_metrics = {item.scope_id: item.metric for item in evidence.by_layer}
        for layer, minimum in campaign.thresholds.minimum_layer_pass_rates.items():
            metric = layer_metrics.get(layer.value)
            if metric is not None:
                thresholds.append((f"layer:{layer.value}", metric, minimum))
        for scope_id, metric, minimum in thresholds:
            measured = (
                metric.confidence_low
                if campaign.thresholds.mode == AITestThresholdMode.LOWER_CONFIDENCE_BOUND
                else metric.pass_rate
            )
            if measured < minimum:
                findings.append(
                    self._finding(
                        AITestGateCode.THRESHOLD_MISSED,
                        scope_id,
                        "The configured minimum pass rate was not met",
                    )
                )
        if campaign.thresholds.require_zero_critical_failures:
            critical_ids = sorted(
                {
                    trial.case_id
                    for trial in evidence.trials
                    if trial.critical and trial.status != AITestTrialStatus.PASSED
                }
            )
            findings.extend(
                self._finding(
                    AITestGateCode.CRITICAL_FAILURE,
                    f"case:{case_id}",
                    "A critical case did not pass every repetition",
                )
                for case_id in critical_ids
            )
        return findings

    def _baseline_findings(
        self,
        campaign: AITestCampaign,
        candidate: AITestEvidence,
        baseline: ApprovedAITestBaseline,
    ) -> list[AITestGateFinding]:
        approved = baseline.evidence
        compatible = (
            approved.campaign_id == candidate.campaign_id
            and approved.corpus_version == candidate.corpus_version
            and approved.corpus_sha256 == candidate.corpus_sha256
            and {item.scope_id for item in approved.by_case}
            == {item.scope_id for item in candidate.by_case}
        )
        if not compatible:
            return [
                self._finding(
                    AITestGateCode.BASELINE_MISMATCH,
                    baseline.baseline_id,
                    "Baseline campaign or corpus identity does not match the candidate",
                )
            ]

        findings: list[AITestGateFinding] = []
        drop = approved.overall.pass_rate - candidate.overall.pass_rate
        statistically_separated = (
            candidate.overall.confidence_high < approved.overall.confidence_low
        )
        if drop > campaign.thresholds.maximum_baseline_pass_rate_drop and statistically_separated:
            findings.append(
                self._finding(
                    AITestGateCode.STATISTICAL_REGRESSION,
                    "campaign:overall",
                    "Candidate performance regressed beyond the configured tolerance",
                )
            )

        approved_cases = {item.scope_id: item.metric for item in approved.by_case}
        candidate_cases = {item.scope_id: item.metric for item in candidate.by_case}
        critical_attack_ids = {
            trial.case_id
            for trial in candidate.trials
            if trial.critical and trial.kind == AITestCaseKind.ATTACK
        }
        for case_id in sorted(critical_attack_ids):
            before = approved_cases[case_id]
            after = candidate_cases[case_id]
            if before.pass_rate == 1.0 and after.pass_rate < 1.0:
                findings.append(
                    self._finding(
                        AITestGateCode.SECURITY_REGRESSION,
                        f"case:{case_id}",
                        "A previously passing critical attack case regressed",
                    )
                )
        return findings

    @staticmethod
    def _finding(code: AITestGateCode, scope_id: str, message: str) -> AITestGateFinding:
        return AITestGateFinding(
            code=code,
            severity=Severity.HIGH,
            scope_id=scope_id,
            message=message,
        )
