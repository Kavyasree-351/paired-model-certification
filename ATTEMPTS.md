# Attempt log

## Before experiments

1. Conformal early-exit selection and inference-configuration calibration screened out after finding close recent work. No experiments or originality claims.
2. Chose a narrow ternary paired-loss certificate as a falsifiable candidate. Wrote RESEARCH_PLAN.md and saved its SHA256 before pilot execution. Known ingredients explicitly credited. No global novelty asserted.
3. Five smoke tests passed, including fixed-sequence no-restart and global split isolation. Initial dependency import test failed before the local scientific-library installation completed; it was rerun successfully. No experiment outcomes depended on that failed import.

## Original pilot

Seeds 11,22,33; all four predefined datasets; all eight methods and both testing corrections; robustness checks at both noise levels. Candidate macro routing .6806067, best valid baseline .6993019; paired difference -.0186952; positive pair fraction 0.0. Gate failed. No full experiment authorized by this evidence.

Diagnosis: ties dominate and benefits are uncommon in these model pairs. An exact harm-only binomial test can be more powerful than estimating the difference between two rare events. A likelihood mixture pays prior regret for a nuisance distribution without enough beneficial outcomes to offset it. This diagnosis is a hypothesis, to be checked against controlled simulations and saved raw results.

## Post-pilot alternatives (exploratory)

Both alternatives keep models, model-tuning budgets, threshold grid, epsilon, delta, seed list and data pools fixed. They use the tuning losses, which are independent of certificate data even though they also select model hyperparameters.

4. Tuning likelihood: a Jeffreys-smoothed ternary distribution learned on tuning rows supplies an independent numerator in the same universal-inference certificate. This tests whether integrating a broad prior was the practical weakness. Known universal-inference specialization; no new-method claim.
5. Tuning stake: maximize tuning log wealth over a fixed stake grid, freeze it, then evaluate calibration wealth. This tests whether adaptive/mixture betting overhead was the weakness. Known split betting principle; no new-method claim.

Fresh confirmation must be run only after alternatives have been frozen. Its purpose is confirmation of the feasibility assessment, not permission to rename an existing method as novel. Final evaluation remains gated on the original predefined opportunity criterion; it is not bypassed by an exploratory alternative.

## Frozen fresh confirmation

Before opening fresh rows, saved source-file SHA256 hashes in frozen-before-fresh.json. Seeds 101,202,303 use globally separate fresh rows. Original candidate .6932566 versus best baseline .7075000; no positive paired wins. Tuning likelihood .6026645; tuning stake .7045760. Neither alternative improves on the strongest baseline. Full experiments remain unrun. Fresh rows are now spent confirmation data and cannot serve as a fresh holdout for further method selection.

## Component diagnosis after confirmation

Ran development-only matched ablations: same ternary certificate with benefit outcomes replaced by ties gives .6782919 routing; using only a symmetric Jeffreys prior gives .4356019; full gives .6806067. Thus benefit credit contributes only .0023148 in this pilot, and the sparse-tie prior matters substantially relative to a symmetric prior. Neither establishes an advantage over exact harm-only testing. These analyses were post hoc and never used to select a model for final testing.

## Verification and derived-metric correction

Expanded correctness suite to six passing tests, adding exact small-sample enumeration of e-value expectations under several nulls. The attempted full command exits before loading any dataset, as intended. Audited raw cascade accuracy against reference accuracy minus signed excess, all seed counts, and global final-pool separation.

Found a reporting-only issue after confirmation: the estimated latency formula charged cheap inference even when no threshold passed and the system falls back directly to the reference. Corrected these fallback ratios to 1.0 in code and saved reports; original raw files were retained as raw-before-latency-correction.json. Selection, routing, accuracy, gates and all scientific conclusions are unchanged. The fresh snapshot records the pre-reporting-fix source; later changes added diagnostics and reports, not candidate-selection changes.

## Reference-revealed information follow-up

Implemented a genuinely reference-dependent label-purchasing policy, matched against a precommit policy using the same correction estimator, scout, optimizer budget and total acquisition budget. Ran 18 synthetic scenarios, three trained-surrogate seeds, 20 evaluation streams per seed and five policies: 5,400 trajectories. Five correctness tests passed. Expensive-label learned gains versus matched precommit were -.0333 in both heterogeneous settings, with positive gains in only one of three seeds per setting. Pilot opportunity gate failed; reserved fresh synthetic evaluation was exercised only through its guard and refused before generation. Original real final pools were never imported.

A deeper literature search identified direct overlap with 2014 auxiliary-covariate two-phase sampling theory, including the conditional-variance allocation and cost tradeoff. The simple variance-separation argument therefore does not establish novelty. This was an important correction to the earlier, explicitly tentative suggestion of a research gap. It does not rule out every research contribution on this problem, but invalidates the specific broad claim pursued here.

The homogeneous control initially used a single reference category; corrected it to expose all three categories with identical conditional risk. All initial outputs were archived, and all heterogeneous outputs reproduced exactly after correction. No negative scenario was removed. Current code, derivations, uncertainty, primary sources and exact commands are recorded in REFERENCE_REVELATION_ASSESSMENT.md. No new complete paper was created because the user's conditional novelty requirement remains unmet.
