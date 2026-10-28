# Staged acquisition: novelty and experimental audit

## Decision

**The proposed novelty has NOT been achieved. No complete research paper is created for this attempt.** This report records the theory check, executed experiments and reasons for stopping before full final evaluation. A practical improvement over a weak sampling rule would not repair the novelty issue.

## Exact reproduction

The previous selective smoke experiment was rerun. Every saved raw outcome reproduced exactly (SHA256 checked), and all four original tests passed. The larger audit adds four passing checks, including hidden-reference loss preservation, an exact coupling argument and numerical optimization. No original real-data final or fresh-confirmation pool was accessed.

## Restricted dominance argument

Let H be information available before querying the reference, q(H) the probability of purchasing it, and l(H) the probability of labeling a disagreement after that purchase. Agreement labels are skipped. Let D be the signed paired error, which is zero whenever predictions agree.

The observed inverse-propensity-weighted loss is Q*L*D/(q*l). Conditional on H and the full case, Q*L is Bernoulli(q*l). Instead, purchase the reference with probability q'=q*l and label every observed disagreement. The signed observation has the same distribution. Under a coupling it is identical on every case. Expected label expenditure is unchanged, but expected reference expenditure drops from c_R*q to c_R*q*l. It drops strictly wherever q>0 and l<1. Thus a separate context-only label-subsampling stage is dominated for this estimator class.

This argument requires that l is measurable using cheap-visible H apart from the agreement skip. It does NOT cover label decisions that exploit genuinely new reference-side information, augmented residual estimators, arbitrary loss functions with no zero-agreement property, or all possible acquisition algorithms. It is an elementary diagnosis, not a claim of a new general lower bound.

## Numerical optimization

For two cheap-visible strata with weights w, disagreement rates g and binary disagreements, the raw-IPW second moment is sum(w*g/(q*l)). Expected acquisition cost is sum(w*q*(c_R+c_Y*g*l)). Optimizing the nested design at a fixed expected cost returns l=1 in both cost settings and exactly the same q and objective as optimizing the collapsed design. The square-root allocation follows a standard constrained sampling calculation; it is not minimum expected stopping-cost optimality or a new theorem.

## Executed pilot

Three predefined seeds (811,822,833), 50 repetitions per seed, two label/reference cost ratios (.2 and 5), and three true excess errors (.02,.05,.08). There are 900 base streams and seven policies per stream: 6,300 evaluated policy trajectories. Each stream permits 10,000 cases; each policy has a total acquisition budget of 4,000 reference-cost units. The target is excess error <=.05 with failure probability .05.

Policies include all-reference acquisition with agreement label skipping; uniform reference sampling; oracle nested and collapsed designs; an allocation learned from a separate 500-case reference-only scout whose cost is included; and the earlier smoke policy with its collapsed counterpart. No human scout labels are needed to estimate disagreement. All methods use the same known importance-weighting and betting principles. Oracles are diagnostic upper benchmarks for the chosen second-moment objective, not attainable learned baselines.

A hidden common binary label permutation preserves paired losses while preventing a constant reference from trivially revealing disagreement through the cheap prediction. This refinement and a correction removing unnecessary scout-label charges were documented before the final rerun; earlier outputs are preserved in the result folder. No scenario, cost ratio or seed was dropped.

### Safe case: true excess .02

| Label cost / reference cost | Method | Certified within budget | Mean total cost |
|---:|---|---:|---:|
| 0.2 | all_reference | 94.0% | 1751.5 |
| 0.2 | learned_allocation | 97.3% | 1732.1 |
| 0.2 | oracle_collapsed | 99.3% | 1293.0 |
| 0.2 | oracle_joint | 99.3% | 1293.0 |
| 0.2 | smoke_collapsed | 94.7% | 1591.5 |
| 0.2 | smoke_joint | 96.7% | 1841.5 |
| 0.2 | uniform_reference | 95.3% | 1686.1 |
| 5.0 | all_reference | 71.3% | 2601.1 |
| 5.0 | learned_allocation | 70.7% | 2793.9 |
| 5.0 | oracle_collapsed | 75.3% | 2418.2 |
| 5.0 | oracle_joint | 75.3% | 2418.2 |
| 5.0 | smoke_collapsed | 56.7% | 2982.9 |
| 5.0 | smoke_joint | 48.0% | 3213.6 |
| 5.0 | uniform_reference | 70.7% | 2643.7 |

Costs include unsuccessful trajectories through their budget/horizon limit. Certification success and expenditure must be read together; a smaller cost with fewer successful certificates is not necessarily better. Exact oracle nested/collapsed outcomes matched on all 900 paired streams. Marginal 95% binomial intervals for every scenario and method are in summary.json. Three seeds are not sufficient publication evidence, and these are synthetic structural diagnostics.

The maximum observed boundary/unsafe certification rate across methods and cost settings was 0.0267. This is a Monte Carlo diagnostic, not a proof or simultaneous guarantee audit. Runtime excluding imports was 32.46 seconds.

## Literature and novelty check

- Xu, Karampatziakis and Mineiro, *Active, anytime-valid risk controlling prediction sets*, NeurIPS 2024: https://arxiv.org/abs/2406.10490 . Already establishes active labeling, sequential guarantees and optimal allocation results.
- Brawand et al., *Active Multiple-Prediction-Powered Inference*, May 2026 preprint: https://arxiv.org/abs/2605.08429 . Jointly optimizes predictor purchases, labels and prediction weights under cost constraints; its guarantees are asymptotic. This directly overlaps the broad proposed allocation claim.
- Ao et al., *Best Arm Identification with LLM Judges and Limited Human*, January 2026 preprint: https://arxiv.org/abs/2601.21471 . Develops selective audits, anytime-valid inference and near-oracle allocation for biased proxies.

These are not exact reproductions of the authors' implementations. No claim of experimental superiority over those papers is made. The restricted collapse result plus close prior work is enough to reject the proposed broad novelty claim without spending final data.

## What has and has not been achieved

Achieved: runnable staged acquisition code, exact reproduction of prior smoke results, a restricted structural diagnosis, matched-budget multi-seed simulations, scout-inclusive costs, uncertainty estimates and retained raw data.

Not achieved: a novel finite-sample cost-optimal allocation theorem, a matching information-theoretic lower bound, superiority against the closest published implementations, realistic AI-serving evaluation, fresh confirmation of a selected new method, or full final-test results. The proposed complete-paper condition is therefore unmet.

A possible remaining research question concerns finite-sample acquisition that uses information revealed only after a reference purchase. It requires a separate novelty audit and proof; merely combining AM-PPI-style allocation with an existing betting certificate would not establish it. No positive claim is made for that untested extension.

## Reproduction

From this folder in the installed clean environment:

```powershell
.\.venv\Scripts\python.exe selective_smoke.py
.\.venv\Scripts\python.exe staged_audit.py
.\.venv\Scripts\python.exe staged_report.py
```

Use the same dependency lockfile as the existing codebase. Results are in results/staged_audit/. The existing final-run gate remains closed.
