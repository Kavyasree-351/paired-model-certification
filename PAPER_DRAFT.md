# Ternary Likelihood Certificates for Cheap Model Replacement: A Negative Feasibility Study

**Working draft, 26 October 2026. Not submission-ready.** All numerical results below are from executed development or fresh-confirmation experiments. Final-test results are **PENDING AND GATED OFF**. No claim of a novel general inference procedure is made.

## Abstract

Replacing some predictions of an accurate model with a cheap classifier requires balancing computation against accuracy degradation. We investigate whether explicitly modeling harmful, beneficial and tied paired outcomes improves certification efficiency relative to signed bounded-mean methods and exact harm-only testing. A fixed Dirichlet-mixture likelihood divided by a constrained null likelihood provides a finite-sample e-value for excess classification error. We combine this established principle with fixed-sequence testing over confidence thresholds. On four small benchmarks and three training seeds, the candidate routes 68.06% of examples, compared with 69.93% for the strongest baseline. Frozen confirmation on globally separate rows and three new seeds yields 69.33% versus 70.75%. Two tuning-based alternatives fail to outperform the baseline. Controlled simulations show that benefit credit can matter when discordance is substantial, but this does not translate into a practical advantage on the tested model pairs. We therefore decline to run final experiments or claim a new superior method. The contribution of this draft is a reproducible, narrowly scoped negative assessment; broader scientific novelty remains unestablished.

## 1. Motivation and relation to previous work

Risk-controlled adaptive inference is useful when a deployment pipeline must decide how often to invoke an expensive reference. ESANN emphasizes generic learning algorithms and statistical foundations, making this methodological question more appropriate than novelty based solely on a niche application [1].

The closest work is *Fast yet Safe* [2]. It already studies risk-controlled early exits, performance differences that credit improvements over the final model, and WSR betting bounds. Thus neither signed excess risk nor betting-based calibration is new. Learn then Test [3] supplies fixed-sequence family-wise error control without requiring monotone risks. Waudby-Smith and Ramdas [4] supply general bounded-mean betting methods. Universal inference [5] establishes likelihood-based e-value constructions with constrained null likelihoods.

Our narrow hypothesis was that an explicit ternary likelihood would use sparse discordance more efficiently than generic signed-mean bounds. It is an empirical hypothesis about an implementation of known principles. Simply combining those principles does not establish a new algorithmic contribution. The experiments below fail to demonstrate the practical advantage needed to justify a stronger claim.

Recent conformal early-exit selection and inference-configuration shift papers [6,7] were also screened. Their existence weakens the case for pursuing simple calibration wrappers as new contributions. This targeted search cannot establish the absence of equivalent paired tests elsewhere in statistics.

## 2. Setup and certificate

Let c(x) be a cheap classifier and r(x) a reference, both trained before calibration. A fixed policy routes to c if its maximum class probability exceeds threshold t; otherwise it invokes r. Define

\[
D_t=\mathbf 1\{\text{routed}\}\left(\mathbf 1\{c(X)\ne Y\}-\mathbf 1\{r(X)\ne Y\}\right).
\]

Then D_t is -1, 0 or 1 and its mean is the cascade's excess error over r. We target mean excess at most epsilon=.05, with calibration failure probability delta=.05. This allows five percentage points of degradation; it is not an absolute accuracy guarantee.

For counts N_- , N_0 , N_+ in n IID calibration observations, let M(D) be a fixed mixture of three Dirichlet-integrated sequence likelihoods with concentrations (.5,.5,.5), (.1,5,.1) and (1,10,1). Let

\[
L_0(D)=\sup_{p_+-p_-\ge\epsilon}\prod_{v\in\{-,0,+\}}p_v^{N_v},\qquad E(D)=M(D)/L_0(D).
\]

We set E to zero when the empirical excess is at least epsilon, a conservative truncation. Under any ternary null distribution P, the denominator dominates its likelihood. Hence E_P[E] <= E_P[M/L_P] <= 1. Markov's inequality permits rejection of the null at E >= 1/delta. This is a standard e-value argument, not a claimed new theorem. The null maximization reduces to one concave dimension at p_+-p_-=epsilon when empirical excess is below epsilon. The implementation upper-bounds the optimum using a tangent to avoid inflating evidence through numerical underestimation.

Twelve thresholds decrease from .99 to .5 and are fixed independently of calibration. We test in that order and stop on the first failure. The fallback uses r everywhere and has identically zero excess risk. By the first-true-null argument, any selection of a bad policy requires rejection of the first true null, with probability at most delta. No assumption of monotone policy risk is required. IID calibration, fixed models and stable deployment distribution are required. A Bonferroni variant is secondary. Shifted evaluations have no such guarantee.

## 3. Experimental protocol

We use sklearn digits, breast cancer Wisconsin diagnostic and wine, plus a fixed synthetic classification dataset with 12,000 observations and 24 features. These are feasibility benchmarks rather than a publication-scale suite. We fix global training+tuning, calibration, pilot, confirmation and final pools at 50%, 20%, 10%, 10% and 10%. The training+tuning pool is reassigned by seed into 40% training and 10% tuning. No confirmation or final rows enter training, scaling, tuning or calibration under any seed.

Cheap logistic regression tunes C in {.1,1,10}; the RBF SVC reference tunes C in {1,10,100}. Both get three fits and use tuning accuracy with deterministic ties. All certificate methods use exactly the same fitted models and predictions. Neither model is re-fit on calibration. Pilot seeds are 11,22,33; confirmation seeds are 101,202,303.

Baselines include fixed-mixture signed betting, predictable signed betting, empirical Bernstein, Hoeffding, split Clopper–Pearson bounds for harmful-minus-beneficial probability, and exact harm-only binomial testing. A naive empirical rule is explicitly uncertified. The betting baselines implement established generic principles but do not reproduce the exact WSR implementation of [2]; we therefore make no superiority claim against that paper.

Before experiments, we specified a gate requiring at least .03 higher macro routed fraction than the strongest valid baseline, positive paired gains in at least two thirds of dataset–seed pairs, observed excess no larger than .05 on each pair, and evidence beyond the harm-only and signed-mixture variants. This is an exploratory opportunity rule, not a formal statistical significance threshold. Full experiments require both pilot and fresh-confirmation gates to pass.

## 4. Verified results

| Method | Pilot routed fraction | Fresh confirmation routed fraction |
|---|---:|---:|
| Ternary mixture candidate | .6806 | .6933 |
| Harm-only exact binomial | .6993 | .7075 |
| Split Clopper–Pearson | .6993 | .7075 |
| Predictable signed betting | .6346 | .6499 |
| Fixed-mixture signed betting | .5326 | .4535 |
| Empirical Bernstein | .2805 | .2082 |
| Hoeffding | .0142 | .0000 |
| Naive empirical, uncertified | .9632 | .9610 |

Macro means equally weight all four datasets and all three training seeds. These are descriptive quantities; methods share observations. The original candidate's paired gain against the strongest baseline is -.0187 in pilot and -.0142 in confirmation. It has no positive paired wins in either stage. Both gates fail.

| Dataset | Pilot candidate routing | Fresh candidate routing | Fresh excess error |
|---|---:|---:|---:|
| Digits | .9444 | .9722 | .0056 |
| Breast cancer | .9708 | .9883 | .0000 |
| Wine | .0000 | .0000 | .0000 |
| Synthetic | .8072 | .8125 | .0236 |

All three seeds per dataset are retained. Per-dataset training-seed t intervals are in the generated table sources. Because seeds reuse fixed calibration/evaluation pools, those intervals describe training variability conditional on the pools. They do not estimate uncertainty across new datasets or independently sampled evaluation pools. Three seeds and tiny real-data evaluation sets give weak inferential resolution. Zero empirical excess on a small pool is not evidence of zero population risk.

### Alternatives and component diagnosis

After the initial failure, we fixed two principled alternatives before opening confirmation data. A ternary likelihood numerator estimated on separate tuning rows routes .6935 in pilot but only .6027 in confirmation. A stake selected on tuning rows routes .6993 and .7046. Both are known inference specializations and neither outperforms the strongest baseline.

Post-confirmation ablations use only the original development rows. Removing benefit credit from the same candidate yields .6783 routing versus .6806 for the full candidate, a .0023 contribution. Retaining only the symmetric Jeffreys prior yields .4356. Sparse-tie prior structure matters relative to that particular prior, while benefit credit contributes little on these pairs. These diagnoses do not rescue the main claim and were not used for final-test selection.

### Controlled statistical and robustness checks

We run 36 ternary-distribution scenarios varying sample size (250,1000,4000), discordance (.06,.2,.6) and true excess (0,.04,.05,.06), with 300 independent draws per scenario across three fixed simulation seeds. The maximum observed null rejection rate across valid methods/scenarios is .0267; the maximum marginal 95% interval upper endpoint is .0519. These diagnostic intervals are not simultaneous validation or a substitute for proof.

At n=4000 and true excess zero, the candidate certifies in 100%, 99% and 16.3% of draws at discordance .06, .2 and .6. Predictable signed betting certifies in 91.3%, 100% and 77.3%; harm-only testing certifies in 100%, 0% and 0%. Benefit-aware tests therefore help in controlled settings where improvements and degradations coexist, but explicit ternary modeling is not uniformly better than generic signed betting. This prevents presenting the real-data negative result as a universal impossibility result.

Gaussian feature noise with train-only feature scales at severities .1 and .3 produces fresh macro candidate excess .0080 and .0058. These mild perturbations are limited stress tests, with no distribution-shift certificate or robustness-superiority claim.

### Efficiency

Measured model fitting plus tuning totals 18.05 seconds for the original 12-pair pilot and 17.00 seconds for confirmation, excluding imports and reporting. Seven warmed batch timings give per-row cost estimates. Candidate certificate evaluation is approximately milliseconds to tens of milliseconds in these runs. Estimated cascade/reference latency ratios include cheap inference whenever a cascade is selected; a reference-only fallback has ratio 1. These estimates omit serving, dispatch and changing batch-size overhead. End-to-end latency, memory and energy claims remain pending. No routing percentage is represented as an equal percentage latency saving.

## 5. Limitations, pending evidence and decision

The study has not established defensible method novelty or a meaningful empirical advantage. Its small benchmarks, narrow model families, three seeds and shared calibration pools fall short of a strong conference main-method evaluation. We therefore do not open the final pool or run the configured 20-seed full experiment. Fresh confirmation is now spent data and cannot be reused as an untouched validation set for new method selection.

A publishable positive claim would require a distinct technical result beyond known e-value composition, an exact reproduction of the closest WSR baseline, stronger model families and datasets, independent calibration replications, calibration-budget scaling, paired uncertainty at the appropriate sampling unit, and genuine deployment measurements. Final-test performance, large-scale superiority and broad distribution-shift safety remain **PENDING**. The present evidence supports discontinuing this particular candidate rather than lowering the bar or relabeling a known combination as novel.

## References

1. ESANN. [Scope](https://www.esann.org/node/3), [conference dates](https://www.esann.org/).
2. Jazbec et al. [Fast yet Safe: Early-Exiting with Risk Control](https://arxiv.org/abs/2405.20915). NeurIPS 2024.
3. Angelopoulos et al. [Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052).
4. Waudby-Smith and Ramdas. [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686). JRSS B, 2024.
5. Wasserman, Ramdas and Balakrishnan. [Universal inference](https://doi.org/10.1073/pnas.1922664117). PNAS, 2020.
6. [Selection-Safe Conformal Early Exit: Distribution-Free Coverage under Adaptive Depth](https://arxiv.org/abs/2610.20773). October 2026 preprint; screening source.
7. [Conformalized Large Language Models under Configuration Shift](https://arxiv.org/abs/2608.01460). August 2026 preprint; screening source.
