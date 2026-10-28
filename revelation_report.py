"""Summarize the reference-revelation experiment without making a paper claim."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from revelation_audit import parameters,variance

ROOT=Path(__file__).resolve().parent
out=ROOT/'results/revelation_pilot'
summary=json.loads((out/'summary.json').read_text())
raw=json.loads((out/'raw.json').read_text())
gate=json.loads((out/'gate.json').read_text())
meta=json.loads((out/'execution_metadata.json').read_text())
checks={}
old=out/'before-homogeneous-control-correction/raw.json'
if old.exists():
    earlier=[r for r in json.loads(old.read_text()) if r['rho']<1.]
    later=[r for r in raw if r['rho']<1.]
    assert earlier==later,'Unchanged heterogeneous scenarios did not reproduce'
    checks['heterogeneous_outputs_identical_after_control_fix']=True
paired=[]
for g in gate['gains']:
    differences=np.array(g['seed_gains'])
    half=float(t.ppf(.975,2)*differences.std(ddof=1)/np.sqrt(3))
    paired.append({**g,'conditional_seed_t95_interval':[float(differences.mean()-half),float(differences.mean()+half)]})
theory=[]
for rho in [.1,.5]:
    p,m=parameters(rho,.02)
    b=.2;v=1-m*m
    pre=np.repeat(b,3)
    post=b*np.sqrt(v)/(p@np.sqrt(v))
    assert np.max(post)<1 and np.min(post)>.05
    before=variance(p,m,m,pre);after=variance(p,m,m,post)
    theory.append({'rho':rho,'expected_label_fraction':b,'pre_variance':before,'post_variance':after,
      'variance_reduction_fraction':1-after/before,'post_label_probabilities':post.tolist()})
with (out/'table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
analysis={'paired_gains':paired,'known_theory_examples':theory,'checks':checks,
 'null_max_empirical_rate':max(r['rate'] for r in summary if r['target']>=.05),
 'warning':'Only three trained-surrogate seeds; pooled binomial intervals are descriptive in this heterogeneous replication design. Seed t intervals are approximate and wide. This is not an empirical proof of safety or impossibility.'}
(out/'analysis.json').write_text(json.dumps(analysis,indent=2))
table=['| Revealed-information regime | Label/reference cost | Precommit learned | Post-revelation learned | Raw label-all |',
 '|---|---:|---:|---:|---:|']
for rho in [.1,.5,1.]:
    for cy in [1.,20.]:
        pick=lambda m:next(r['rate'] for r in summary if r['rho']==rho and r['label_cost']==cy and r['target']==.02 and r['method']==m)
        table.append(f"| {rho} | {cy:g} | {100*pick('learned_pre'):.1f}% | {100*pick('learned_post'):.1f}% | {100*pick('raw_all'):.1f}% |")
report='''# Does purchased reference information create a novel contribution?

**Decision: no defensible novelty has been established. Both the pilot opportunity gate and the separate novelty gate fail. No complete paper or final evaluation is justified for this attempt.**

## What changed from the previous experiment

The previous label-purchase rule used only cheap-visible information, so it collapsed to a simpler acquisition policy. This experiment genuinely reveals new information: the expensive reference output identifies one of three groups whose paired-error uncertainty differs. The cheap prediction is class 3 and conveys no group identity. The reference predicts class 0, 1 or 2; the ground-truth label determines whether cheap or reference is right. Labels are purchased only after the reference prediction when using the post-revelation policy.

The data are intentionally controlled synthetic classifications, not a realistic model-serving benchmark. Two heterogeneous regimes use conditional paired-error means +.95, -.95 and 0. The noise fraction (.1 or .5) controls the frequency of the unpredictable group. A homogeneous control uses all three reference categories with identical conditional paired-error distributions. Its oracle has no useful allocation information. Three true excess levels (.02,.05,.08) test safe, boundary and unsafe cases relative to epsilon=.05.

## Fair comparison and implementation

The matched precommit and post-revelation policies share the same trained correction function, estimator, 120-case independent scout, cost accounting, allowed label-probability range [.05,1], three optimizer starts, iteration cap, 95% confidence level, and sequential certificate. Precommit uses one label probability independent of the reference output. Post-revelation can use three probabilities, one for each output. This is the sole intended acquisition difference. Reference predictions are paid for on all evaluation cases; this isolates the value of information revealed by that purchase rather than claiming a complete joint-reference-query optimizer.

The observed outcome is Z = mhat(R) + L/l(R)*(D-mhat(R)). Both policies use exactly this correction after buying R, so a weak uncorrected estimator is not substituted for the precommit baseline. Conditional label sampling is independent of Y given R and the independently trained model. Therefore E[Z]=E[D], even if mhat is wrong. For fixed positive label probabilities, Z has a known finite upper bound. A fixed mixture of nonnegative betting products provides the same known anytime-valid testing machinery to both policies. This is a standard construction, not a new result.

An oracle counterpart of each allocation uses true conditional distributions and pays no scout cost; it is labeled an unattainable diagnostic, never a practical competitor. A strong raw-all comparator needs no surrogate or scout and buys all labels. It is retained even when it beats the candidate. All costs of the learned methods' scout are included. Label-all and learned methods each face a total budget of 30,000 reference-cost units and a maximum 16,000 evaluation arrivals. We compare certification success together with expenditure, including failed attempts, rather than comparing savings from unequal success rates.

## Executed evidence

18 scenarios = three information regimes x two cost ratios x three risk settings. Each uses three independently trained-surrogate seeds and 20 independent evaluation streams per seed. Five policies produce 5,400 evaluated trajectories. These are development pilots, not a full benchmark. No original real-data train/calibration/evaluation/final pools are opened. Fresh synthetic seeds are reserved and gated off.

The safe case has true excess .02:

'''+ '\n'.join(table)+f'''

For expensive labels, the learned post-revelation policy loses 3.33 percentage points to its matched precommit counterpart in BOTH heterogeneous regimes. It wins on only one of three seeds in each. Cheap-label successes do not satisfy the predefined costly-label objective, and the strong label-all comparator often does better. These findings do not prove the approach can never help: the conditional-seed uncertainty intervals in analysis.json are wide. They do establish that this implementation lacks the required consistent pilot opportunity.

There were {meta['checks_passed']} passing correctness tests: reconstruction of the paired classification loss; exact unbiasedness and null wealth expectation; the oracle variance identity; inclusion of precommit allocation in the post-revelation class; and identical oracle variance allocation in the homogeneous case. The earlier heterogeneous outputs reproduced exactly after correcting the control. All initial outputs remain in before-homogeneous-control-correction/.

The maximum empirical boundary/unsafe certification rate was {analysis['null_max_empirical_rate']:.4f} over cells with only 60 streams. A cell above .05 is not sufficient to establish a mathematical validity failure, and these small simulations do not establish validity either. The guarantee rests on the explicitly stated IID and sampling assumptions and the e-value argument. Pooled binomial intervals are descriptive because the trained surrogate differs across seeds; paired seed t intervals are also approximate with only three seeds.

Runtime was {meta['seconds_excluding_imports']:.2f} seconds excluding imports. No GPU was used. The acquisition learner is frozen after the scout; online adaptation and nonstationarity are untested.

## The sought theoretical advantage exists, but it is already known

With a correct fixed conditional mean m(R), label probability l(R) and paired loss D, the per-case variance is

    Var(m(R)) + E[Var(D | R) / l(R)].

At a fixed expected label fraction b, a constant precommit allocation contributes E[v]/b to the second term. If probabilities do not hit their bounds, allocating proportionally to sqrt(v) contributes (E[sqrt(v)])^2/b. Jensen's inequality gives an improvement whenever conditional uncertainty varies. For b=.2, the controlled heterogeneous examples reduce variance by approximately '''+f"{100*theory[0]['variance_reduction_fraction']:.1f}% and {100*theory[1]['variance_reduction_fraction']:.1f}%"+'''. These are analytic oracle variance calculations, NOT observed gains in certification success or a new theorem.

This is the same auxiliary-assisted two-phase sampling structure already studied in statistics. It answers the narrow question 'can purchased information improve label allocation?' positively, but it does not give us a novel research contribution. A finite-sample betting wrapper does not itself turn that established result into a new one. In practice, estimating the conditional means and accounting for conservative importance-weight bounds can consume that oracle advantage, as the pilot illustrates.

## Closest primary sources

1. *Optimal Auxiliary-Covariate Based Two-Phase Sampling Design for Semiparametric Efficient Estimation of a Mean or Mean Difference*, Statistics in Medicine (2014): https://pmc.ncbi.nlm.nih.gov/articles/PMC4021041/ . Derives optimal label/outcome subsampling after observing auxiliary information, accounting for acquisition costs. This directly overlaps our variance and allocation argument.
2. Xu et al., *Active, anytime-valid risk controlling prediction sets*, NeurIPS 2024: https://arxiv.org/abs/2406.10490 . Active labeling, predictor-assisted risk control, optimal allocation and sequential guarantees are established here.
3. Brawand et al., *Active Multiple-Prediction-Powered Inference*, 2026 preprint: https://arxiv.org/abs/2605.08429 . Predictor and label costs are jointly optimized. Its asymptotic setting is different from our finite-sample certificate, but the broad acquisition novelty claim substantially overlaps.

These observations come from primary-source reading, not an exhaustive assertion that every possible extension has already been solved. No exact reproduction of those authors' full implementations or superiority claim is made. A matching information-theoretic lower bound and a nontrivial new finite-sample optimality theorem have not been produced. Renaming the auxiliary variable as an AI model prediction does not resolve that deficit.

## Attempt history and decision

The first pilot control exposed only one reference category in the homogeneous setting. That could create bound artifacts and was inadequate as a no-information control. It was changed to retain all categories with identical risk, and every earlier raw outcome was archived. The heterogeneous scenarios and their predefined unfavorable results were unchanged and reproduced exactly. There was no seed, scenario or metric removal.

The opportunity gate requires a >=.05 success-rate improvement over matched precommit acquisition in both heterogeneous costly-label settings, with gains in at least two thirds of seeds, plus a valid control and a distinct technical contribution. The measured heterogeneous gains are -.0333 and -.0333; the gate fails before any broader study is warranted. The novelty assessment also fails because the demonstrated separation is established sampling theory. Therefore the fresh/full stages remain closed. The complete-paper condition is unmet.

This attempt narrows the conclusion: reference-revealed information CAN be useful, but that fact is not new, and this implementation does not establish a consistent certification advantage. It does NOT establish that useful research on model-replacement certification is impossible. It does establish that the proposed claimed contribution should not be pursued under its present formulation.

## Exact reproduction commands

From the installed codebase environment:

```powershell
.\\.venv\\Scripts\\python.exe revelation_audit.py --stage pilot
.\\.venv\\Scripts\\python.exe revelation_report.py
# Correctly refuses while either gate is false:
.\\.venv\\Scripts\\python.exe revelation_audit.py --stage fresh
```

The existing requirements-lock.txt applies. Raw outcomes, every learned design, optimizer diagnostics, per-seed gains, uncertainty estimates, gates, metadata and CSV tables are under results/revelation_pilot/. The existing final-test gate and all original final data remain unchanged.
'''
(ROOT/'REFERENCE_REVELATION_ASSESSMENT.md').write_text(report)
print('Wrote verified assessment and CSV table. No paper created.')
