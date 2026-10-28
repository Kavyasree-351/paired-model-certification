# Research question and frozen pilot protocol

Recorded before the first experiment, 26 October 2026. This is an exploratory project, not a submission-ready novel method.

Can a ternary likelihood certificate exploit the sparse paired errors of a cheap classifier and a reference more efficiently than signed bounded-mean betting, without assuming that increasing routing aggressiveness increases population risk?

## Directions considered

| Direction | Engineering value | Closest work / reason to reject or test | Compute |
|---|---|---|---|
| Selection-safe conformal early exit | Adaptive inference with uncertainty | Recent selection-safe early-exit work and existing risk-controlled exits make a simple wrapper inadequate novelty | Medium; multi-exit training |
| Conformal calibration after quantization | Deployment reliability | Configuration-shift and quantization-aware calibration papers already study this; matched recalibration alone is not a new method | Medium to high |
| Ternary paired-loss certification | Decide how much traffic can use a cheap model while limiting accuracy degradation | Fast yet Safe already handles signed losses and WSR betting. The narrow question is whether explicit ternary modeling adds useful power. Universal inference is also established; novelty remains unproved | CPU; small tabular/image classifiers |

The selected direction fits ESANN's emphasis on generic algorithms, statistical foundations and trustworthy AI. The currently advertised edition is ESANN 2027, 21–23 April 2027, with 18 November 2026 submission deadline. Verify official instructions before submission. ESANN fit and a rigorous workflow do not imply A* acceptance or established novelty.

## Primary question, gate and budget

For a cheap classifier c, reference r and confidence threshold t, let D_t = 1[c is wrong and routed] - 1[r is wrong and routed]. It lies in {-1,0,1}; its expectation is the cascade's excess classification error. Target epsilon = .05 and failure probability delta = .05. Select among 12 thresholds fixed on [.99,.5], in decreasing order. On certificate failure, stop; use the last certified threshold, or the unchanged reference if none passes. This is fixed-sequence testing and needs no monotonicity assumption on the risks. Each method receives exactly the same candidate models, predictions, labels, threshold grid and confidence budget. Bonferroni is a matched secondary comparison.

The primary exploratory objective is routed fraction under the certificate. The pilot passes only if the candidate improves by >= .03 over the valid baseline with the highest macro mean routing fraction, wins on at least 2/3 of dataset-seed pairs, has observed excess error <= .05 on every pair, and exceeds both the harm-only and bounded-mixture betting ablations. These are opportunity criteria, not a significance test or a guarantee that noisy empirical evaluation risk never exceeds .05. Runtime must also be reported; routing need not yield latency savings if cheap inference is expensive.

Pilot: seeds 11,22,33. Fresh validation: 101,202,303, on a globally separate pool, with all choices frozen. Full, only after both gates: seeds 1000–1019. Three seeds cannot establish a reliable uncertainty interval or population violation rate.

## Data and leakage prevention

Offline sklearn digits (1797 rows, 64 features), breast cancer Wisconsin diagnostic (569,30), wine (178,13), and a fixed generated nonlinear classification problem (12000,24). Real datasets are small convenience benchmarks, not sufficient breadth for a strong paper. No patient linkage is supplied in these packaged versions; do not claim patient-external validation.

A single global random permutation, seed 904, reserves 50% training+tuning, 20% calibration, 10% pilot evaluation, 10% fresh evaluation, 10% final. Only assignment inside training+tuning varies: 40% training and 10% tuning. No evaluation pool is ever fitted, scaled or calibrated. All final rows remain unused even across different seeds. Random splitting is not stratified. The resulting trained-model seeds share calibration and evaluation pools: intervals describe training variation conditional on these pools, not independent dataset replication.

Both classifiers fit StandardScaler using training rows only. Cheap logistic regression searches C={.1,1,10}; RBF SVC reference searches C={1,10,100}; three fits per model, same tuning accuracy rule and deterministic tie-breaking. Hyperparameters are not re-fit using calibration. No ground-truth reference prediction is required during deployment, but its labels/predictions are needed on calibration. Reference and cheap models are evaluated identically across certificate methods.

## Candidate and baselines

Candidate: fixed mixture of three Dirichlet multinomial sequence likelihoods, divided by the maximum likelihood over the ternary null p_harm-p_benefit >= epsilon. The multinomial coefficient cancels. A concave tangent upper bound guards the numerically computed null denominator. This is an instantiation of known likelihood/e-value ideas, not a new general theorem.

For each null distribution P, E_P[mixture_likelihood / max_null_likelihood] <= E_P[mixture_likelihood / likelihood_P] <= 1. Markov gives a valid fixed-sample test at threshold 1/delta. Cutting the statistic to zero when the empirical mean exceeds epsilon preserves validity. Fixed-sequence FWER follows by considering the first true null. IID calibration and a candidate family fixed independently of calibration are essential. The full matrix of thresholds is not optimized using calibration outcomes. Naive empirical selection is explicitly uncertified.

Baselines: fixed mixture signed betting, predictable signed betting using past data only, empirical Bernstein, Hoeffding, split Clopper–Pearson bounds for harmful-minus-beneficial outcomes, and harm-only exact binomial testing. Predictable betting is a strong generic comparator but is NOT an exact reproduction of the WSR implementation in Fast yet Safe. Direct execution of that authors' implementation remains necessary before claims against their method.

Ablations: remove benefits (harm-only); remove ternary likelihood modeling (signed mixture betting); replace fixed sequence with Bonferroni. These are matched functional contrasts, not orthogonal parameter ablations. The removal-of-benefits contrast also changes the statistical test, so it cannot isolate the sign information alone.

## Reporting and limits

Save per-seed raw outcomes, data hashes, exact split indices, chosen hyperparameters, tuning scores, timing and protocol hash. Report reference and cheap accuracy, cascade accuracy, signed excess error, routed fraction, cheap NLL, certificate wall time, training/tuning time, and a batch-latency estimate including cheap inference. Seven warmed timing repetitions; median per row. This is not measured end-to-end routed serving latency or energy. Distribution shifts add Gaussian feature perturbations with train-only scales at .1 and .3; IID certificates do not extend to these shifts.

Report conditional-seed t intervals without clipping them to [0,1]. Do not interpret overlapping pools or methods as independent replications. Empirical excess-risk exceedance is not a population guarantee audit. A dedicated controlled null simulation is a correctness check; broader type-I error/power studies and independent sampling are needed for publication.

If the gate fails: do not run full. Diagnose sparse-sample and prior-regret costs; test at least one principled alternative using the same development pools; freeze and use fresh validation for an honest confirmation, without opening final data. Record every attempted alternative and label post-pilot changes exploratory.

## Primary sources consulted

- ESANN scope: https://www.esann.org/node/3 ; dates: https://www.esann.org/
- Jazbec et al., Fast yet Safe: Early-Exiting with Risk Control, NeurIPS 2024: https://arxiv.org/abs/2405.20915 ; https://github.com/metodj/RC-EENN
- Waudby-Smith and Ramdas, Estimating means of bounded random variables by betting, JRSS B 2024: https://arxiv.org/abs/2010.09686
- Angelopoulos et al., Learn then Test: https://arxiv.org/abs/2110.01052
- Wasserman, Ramdas, Balakrishnan, Universal inference, PNAS 2020: https://doi.org/10.1073/pnas.1922664117
- Selection-Safe Conformal Early Exit, October 2026 preprint: https://arxiv.org/abs/2610.20773 (screening only; do not treat preprints as settled results)
- Conformalized Large Language Models under Configuration Shift, August 2026 preprint: https://arxiv.org/abs/2608.01460
- Rethinking Calibration for Early-Exit Neural Networks, ICML 2026 author page: https://fszatkowski.github.io/publication/failure-prediction-early-exit/

This is a targeted literature search, not evidence that no equivalent test exists. The established ingredients substantially limit the defensible novelty claim. A publishable contribution requires either a genuinely distinct statistical result or a sufficiently informative, broad new empirical finding.
