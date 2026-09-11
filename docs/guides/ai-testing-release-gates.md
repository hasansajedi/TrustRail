# Build AI trustworthiness release gates

`AITestCampaignRunner` turns a private adversarial corpus into a repeatable release
decision. A campaign pins its schema, campaign and corpus versions, application and
model identity, repetitions, optional deterministic seed, attack budget, confidence
level, and explicit pass thresholds.

The framework covers the four control layers in the OWASP AI Testing Guide v1:

| OWASP layer | `AITestLayer` | Typical adapter responsibility |
| --- | --- | --- |
| AI application | `APPLICATION` | Prompt injection, excessive agency, unsafe output |
| AI model | `MODEL` | Model behavior, robustness, bias, and refusal behavior |
| AI infrastructure | `INFRASTRUCTURE` | Isolation, network, deployment, and resource controls |
| AI data | `DATA` | Provenance, poisoning, privacy, and retrieval integrity |

The runner orchestrates tests; your `AITestEvaluator` adapter implements the actual
control-specific assertion. This separation lets the same campaign contract wrap a
local guard, a model provider, an integration test, or an infrastructure probe.

## Configure and run a campaign

Start from the repository's `tests/security_corpus/owasp_ai_testing_campaign.json`
fixture, keep corpus payloads in an access-controlled location, and give every
material corpus change a new `corpus_version`.

```python
from trustrail.testing import (
    AITestCampaign,
    AITestCampaignRunner,
    AITestModelIdentity,
    AITestObservation,
    AITestTarget,
)


class ApplicationEvaluator:
    def evaluate(self, case, context):
        # Pass context.seed to providers that support deterministic seeds.
        passed, safe_result_digest = run_private_assertion(case, seed=context.seed)
        return AITestObservation(passed=passed, outcome_ref=safe_result_digest)


campaign = AITestCampaign.from_path("security/ai-release-campaign.json")
target = AITestTarget(
    application_id="support-agent",
    application_version="git-4f2a8d1",
    model=AITestModelIdentity(
        provider_id="provider",
        model_id="model-x",
        model_revision="2026-08-15",
        configuration_sha256="sha256:" + "a" * 64,
        supports_deterministic_seed=True,
    ),
    infrastructure_revision="deploy-81",
    data_revision="index-42",
)
result = AITestCampaignRunner(ApplicationEvaluator()).run(campaign, target)
result.assert_passed()

# Safe to publish as a CI artifact: payloads and expected values are omitted.
print(result.evidence.model_dump_json(indent=2) if result.evidence else "preflight failed")
```

Use `lower_confidence_bound` for conservative thresholds. In that mode, the lower
Wilson confidence bound—not only the observed pass rate—must meet each minimum.
Small samples therefore need enough repetitions to support the claimed confidence.
`point_estimate` is available for deterministic, exact control checks.

## Approve and compare baselines

Set `require_approved_baseline` when every candidate must be compared with a known
release. Baselines are fail-closed: the runner requires an application-owned
`AITestBaselineVerifier`, an intact evidence digest, the same campaign/corpus
identity, and the same case IDs. `StaticAITestBaselineVerifier` is intended for
tests or baselines loaded from protected application state; production systems can
implement signature, transparency-log, or CI-attestation verification.

A release is blocked when confidence intervals show a pass-rate regression larger
than `maximum_baseline_pass_rate_drop`. A newly failing critical attack case is a
`security_regression` even when the sample is too small for statistical separation.

## Evidence and operational limits

Evidence contains configuration and corpus digests, target/model identity, seeds,
counts, confidence intervals, stable case IDs, outcome digests, limitations, and an
integrity digest. It never contains `payload`, `expected`, evaluator exception text,
or model output. Limitation values and outcome references are validated as safe IDs
or SHA-256 references to reduce accidental disclosure.

Assumptions and residual risks:

- A deterministic seed improves repeatability but does not make different model or
  provider revisions equivalent.
- Confidence intervals quantify sampling uncertainty; they do not prove corpus
  completeness or independence of repeated trials.
- A passing campaign covers only its versioned corpus and adapter assertions. It is
  not an OWASP certification and does not replace exploratory red teaming.
- Protect corpus files, raw provider traces, baseline approvals, and verifier keys
  outside model-controlled state. The evidence digest detects changes but is not a
  signature.
- Set hard budgets before running paid or externally hosted evaluations. Evaluator
  exceptions count against `max_errors` and their text is deliberately discarded.

The older `PromptInjectionRegressionGate` remains available for the repository's
lightweight deterministic prompt-injection checks.
