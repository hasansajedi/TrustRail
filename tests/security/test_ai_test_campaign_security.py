"""Security and bypass tests for AI campaign evidence and baselines."""

from datetime import UTC, datetime
from pathlib import Path

from trustrail.testing import (
    AITestCampaign,
    AITestCampaignRunner,
    AITestGateCode,
    AITestObservation,
    AITestTarget,
    ApprovedAITestBaseline,
    StaticAITestBaselineVerifier,
)

CORPUS = Path(__file__).parent.parent / "security_corpus" / "owasp_ai_testing_campaign.json"
DIGEST = f"sha256:{'a' * 64}"
NOW = datetime(2026, 9, 11, tzinfo=UTC)


class FixtureEvaluator:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises

    def evaluate(self, case, context):
        del case, context
        if self.raises:
            raise RuntimeError("private model response and credential")
        return AITestObservation(passed=True, outcome_ref=DIGEST)


def load_campaign() -> AITestCampaign:
    return AITestCampaign.from_path(CORPUS)


def target() -> AITestTarget:
    return AITestTarget.model_validate(
        {
            "application_id": "security-agent",
            "application_version": "1.1.0",
            "model": {
                "provider_id": "provider",
                "model_id": "model",
                "model_revision": "pinned-revision",
                "configuration_sha256": DIGEST,
                "supports_deterministic_seed": True,
            },
            "infrastructure_revision": "cluster-12",
            "data_revision": "dataset-8",
        }
    )


def test_fixture_covers_every_owasp_layer_and_exports_no_payloads():
    configured = load_campaign()

    result = AITestCampaignRunner(FixtureEvaluator()).run(configured, target(), generated_at=NOW)

    assert result.passed
    assert result.evidence is not None
    exported = result.evidence.model_dump_json()
    assert {item.scope_id for item in result.evidence.by_layer} == {
        "application",
        "model",
        "infrastructure",
        "data",
    }
    assert all(str(case.payload) not in exported for case in configured.cases)


def test_evaluator_exception_is_reduced_to_content_free_error_evidence():
    result = AITestCampaignRunner(FixtureEvaluator(raises=True)).run(
        load_campaign(), target(), generated_at=NOW
    )

    assert not result.passed
    assert result.evidence is not None
    output = result.model_dump_json()
    assert "private model response" not in output
    assert "credential" not in output
    assert AITestGateCode.ERROR_BUDGET_EXCEEDED in {item.code for item in result.findings}


def test_modified_baseline_cannot_bypass_exact_verifier():
    configured = load_campaign()
    evidence = (
        AITestCampaignRunner(FixtureEvaluator())
        .run(configured, target(), generated_at=NOW)
        .evidence
    )
    assert evidence is not None
    baseline = ApprovedAITestBaseline(
        baseline_id="main-approved",
        evidence=evidence,
        approval_ref=f"sha256:{'b' * 64}",
        approver_ref=f"sha256:{'c' * 64}",
        approved_at=NOW,
    )
    verifier = StaticAITestBaselineVerifier([baseline])
    modified = baseline.model_copy(update={"approval_ref": f"sha256:{'d' * 64}"})

    result = AITestCampaignRunner(FixtureEvaluator(), baseline_verifier=verifier).run(
        configured, target(), baseline=modified
    )

    assert not result.passed
    assert [item.code for item in result.findings] == [AITestGateCode.BASELINE_UNVERIFIED]


def test_baseline_verifier_errors_fail_closed():
    class RaisingVerifier:
        def verify_baseline(self, baseline):
            del baseline
            raise RuntimeError("private verifier detail")

    configured = load_campaign()
    evidence = (
        AITestCampaignRunner(FixtureEvaluator())
        .run(configured, target(), generated_at=NOW)
        .evidence
    )
    assert evidence is not None
    baseline = ApprovedAITestBaseline(
        baseline_id="main-approved",
        evidence=evidence,
        approval_ref=f"sha256:{'b' * 64}",
        approver_ref=f"sha256:{'c' * 64}",
        approved_at=NOW,
    )

    result = AITestCampaignRunner(FixtureEvaluator(), baseline_verifier=RaisingVerifier()).run(
        configured, target(), baseline=baseline
    )

    assert [item.code for item in result.findings] == [AITestGateCode.BASELINE_UNVERIFIED]
    assert "private verifier detail" not in result.model_dump_json()
