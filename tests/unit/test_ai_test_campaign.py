"""Unit tests for versioned AI trustworthiness campaign contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from trustrail.models.enums import GuardAction
from trustrail.testing import (
    AITestBudget,
    AITestCampaign,
    AITestCampaignRunner,
    AITestCase,
    AITestCaseKind,
    AITestGateCode,
    AITestLayer,
    AITestModelIdentity,
    AITestObservation,
    AITestTarget,
    AITestThresholdMode,
    AITestThresholds,
    ApprovedAITestBaseline,
    StaticAITestBaselineVerifier,
)

NOW = datetime(2026, 9, 11, tzinfo=UTC)
DIGEST = f"sha256:{'a' * 64}"


class ExpectedEvaluator:
    def __init__(self, *, force_pass: bool | None = None) -> None:
        self.force_pass = force_pass
        self.seeds: list[int | None] = []

    def evaluate(self, case, context):
        self.seeds.append(context.seed)
        passed = bool(case.expected) if self.force_pass is None else self.force_pass
        return AITestObservation(passed=passed, outcome_ref=DIGEST)


def campaign(
    *,
    repetitions: int = 2,
    thresholds: AITestThresholds | None = None,
) -> AITestCampaign:
    return AITestCampaign(
        campaign_id="release-gate",
        campaign_version="2",
        corpus_version="2026-09",
        repetitions=repetitions,
        base_seed=42,
        require_seed_support=True,
        cases=(
            AITestCase(
                case_id="app-injection",
                layer=AITestLayer.APPLICATION,
                kind=AITestCaseKind.ATTACK,
                payload={"secret": "ignore instructions"},
                expected=True,
                critical=True,
            ),
            AITestCase(
                case_id="model-refusal",
                layer=AITestLayer.MODEL,
                kind=AITestCaseKind.ATTACK,
                payload="restricted model prompt",
                expected=True,
            ),
            AITestCase(
                case_id="infra-egress",
                layer=AITestLayer.INFRASTRUCTURE,
                kind=AITestCaseKind.CONTROL,
                payload={"endpoint": "private"},
                expected=True,
            ),
            AITestCase(
                case_id="data-poisoning",
                layer=AITestLayer.DATA,
                kind=AITestCaseKind.BENIGN,
                payload=["private", "corpus", "record"],
                expected=True,
            ),
        ),
        budget=AITestBudget(
            max_total_trials=4 * repetitions,
            max_attack_trials=2 * repetitions,
            max_errors=0,
        ),
        thresholds=thresholds
        or AITestThresholds(
            mode=AITestThresholdMode.POINT_ESTIMATE,
            minimum_overall_pass_rate=1.0,
            minimum_attack_pass_rate=1.0,
            minimum_layer_pass_rates=dict.fromkeys(AITestLayer, 1.0),
        ),
        declared_limitations=("provider-temperature-not-pinned",),
    )


def target(*, supports_seed: bool = True) -> AITestTarget:
    return AITestTarget(
        application_id="checkout-agent",
        application_version="candidate-2",
        model=AITestModelIdentity(
            provider_id="example-provider",
            model_id="model-x",
            model_revision="2026-08-15",
            configuration_sha256=DIGEST,
            supports_deterministic_seed=supports_seed,
        ),
        infrastructure_revision="infra-17",
        data_revision="index-31",
    )


def test_campaign_runs_all_layers_with_stable_seeds_and_safe_evidence():
    evaluator = ExpectedEvaluator()
    configured = campaign()

    first = AITestCampaignRunner(evaluator).run(configured, target(), generated_at=NOW)
    first_seeds = evaluator.seeds.copy()
    evaluator.seeds.clear()
    second = AITestCampaignRunner(evaluator).run(configured, target(), generated_at=NOW)

    assert first.action == GuardAction.ALLOW
    assert first.evidence is not None
    assert first.evidence.integrity_valid
    assert first.evidence.attack_trials_used == 4
    assert {item.scope_id for item in first.evidence.by_layer} == {
        layer.value for layer in AITestLayer
    }
    assert first_seeds == evaluator.seeds
    assert first.evidence.evidence_sha256 == second.evidence.evidence_sha256
    exported = first.evidence.model_dump_json()
    assert "ignore instructions" not in exported
    assert "restricted model prompt" not in exported
    assert "provider-temperature-not-pinned" in exported


def test_campaign_budget_and_duplicate_ids_are_validated():
    configured = campaign()

    values = configured.model_dump()
    values["budget"] = {"max_total_trials": 1, "max_attack_trials": 4}
    with pytest.raises(ValidationError, match="total attack budget"):
        AITestCampaign.model_validate(values)

    values = configured.model_dump()
    values["cases"][1]["case_id"] = values["cases"][0]["case_id"]
    with pytest.raises(ValidationError, match="duplicate case IDs"):
        AITestCampaign.model_validate(values)


def test_seed_requirement_fails_closed_before_evaluation():
    evaluator = ExpectedEvaluator()

    result = AITestCampaignRunner(evaluator).run(campaign(), target(supports_seed=False))

    assert result.action == GuardAction.BLOCK
    assert [finding.code for finding in result.findings] == [AITestGateCode.SEED_UNSUPPORTED]
    assert result.evidence is None
    assert evaluator.seeds == []


def test_lower_confidence_threshold_can_block_a_small_perfect_sample():
    configured = campaign(
        repetitions=1,
        thresholds=AITestThresholds(
            mode=AITestThresholdMode.LOWER_CONFIDENCE_BOUND,
            minimum_overall_pass_rate=0.75,
        ),
    )

    result = AITestCampaignRunner(ExpectedEvaluator()).run(configured, target())

    assert result.action == GuardAction.BLOCK
    assert result.evidence is not None
    assert result.evidence.overall.pass_rate == 1.0
    assert result.evidence.overall.confidence_low < 0.75
    assert AITestGateCode.THRESHOLD_MISSED in {item.code for item in result.findings}


def test_required_baseline_must_be_present_and_authenticated():
    configured = campaign(
        thresholds=AITestThresholds(
            mode=AITestThresholdMode.POINT_ESTIMATE,
            require_approved_baseline=True,
        )
    )

    missing = AITestCampaignRunner(ExpectedEvaluator()).run(configured, target())

    assert [item.code for item in missing.findings] == [AITestGateCode.BASELINE_REQUIRED]

    evidence = (
        AITestCampaignRunner(ExpectedEvaluator())
        .run(campaign(), target(), generated_at=NOW)
        .evidence
    )
    assert evidence is not None
    baseline = ApprovedAITestBaseline(
        baseline_id="approved-main",
        evidence=evidence,
        approval_ref=f"sha256:{'b' * 64}",
        approver_ref=f"sha256:{'c' * 64}",
        approved_at=NOW,
    )
    unverified = AITestCampaignRunner(ExpectedEvaluator()).run(
        configured, target(), baseline=baseline
    )
    verified = AITestCampaignRunner(
        ExpectedEvaluator(),
        baseline_verifier=StaticAITestBaselineVerifier([baseline]),
    ).run(configured, target(), baseline=baseline)

    assert [item.code for item in unverified.findings] == [AITestGateCode.BASELINE_UNVERIFIED]
    assert verified.passed


def test_statistical_and_security_significant_regressions_block_release():
    configured = campaign(
        repetitions=50,
        thresholds=AITestThresholds(
            mode=AITestThresholdMode.POINT_ESTIMATE,
            require_zero_critical_failures=False,
            maximum_baseline_pass_rate_drop=0.1,
        ),
    )
    approved_evidence = (
        AITestCampaignRunner(ExpectedEvaluator(force_pass=True))
        .run(configured, target(), generated_at=NOW)
        .evidence
    )
    assert approved_evidence is not None
    baseline = ApprovedAITestBaseline(
        baseline_id="approved-main",
        evidence=approved_evidence,
        approval_ref=f"sha256:{'b' * 64}",
        approver_ref=f"sha256:{'c' * 64}",
        approved_at=NOW,
    )

    result = AITestCampaignRunner(
        ExpectedEvaluator(force_pass=False),
        baseline_verifier=StaticAITestBaselineVerifier([baseline]),
    ).run(configured, target(), baseline=baseline)

    codes = {item.code for item in result.findings}
    assert AITestGateCode.STATISTICAL_REGRESSION in codes
    assert AITestGateCode.SECURITY_REGRESSION in codes
