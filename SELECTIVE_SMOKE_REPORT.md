# Selective certification smoke test

Executed 26 October 2026. This tests a minimal two-stage acquisition mechanism, not the proposed optimal allocation research contribution. Existing real-data pilot, fresh and final pools were not accessed.

## Outcome

Four correctness tests passed: agreement implies zero paired error difference; skipping agreement labels preserves the entire loss and certificate sequence; two-stage observations and importance weights obey their bounds; repeated selective sampling gives an unbiased estimate within Monte Carlo tolerance on fixed cases.

Ran 900 synthetic trials (three risk settings, three seeds, 100 repetitions per seed), each with 3,000 observations, comparing four acquisition policies on matched underlying data. Fixed IID data and known query probabilities are assumed. The certificate is an anytime-valid fixed-mixture betting construction with importance-weighted signed losses and appropriately restricted stakes. It is established machinery, not a novelty claim. Marginal binomial confidence intervals are in results/selective_smoke/summary.json.

At true excess error .02, with permitted excess .05 and confidence .95:

| Policy | Certification by 3,000 observations | Mean reference calls | Mean labels |
|---|---:|---:|---:|
| Acquire everything | 82.0% | 1662.8 | 1662.8 |
| Skip labels when predictions agree | 82.0% | 1662.8 | 295.7 |
| Uniform reference sampling + skip agreement labels | 42.3% | 1058.7 | 187.8 |
| Fixed joint acquisition policy | 37.7% | 1135.4 | 352.1 |

Costs include failed trials through the observation cap, and successful trials through certificate time. Lower resource use is not meaningful without comparable certification probability. Uniform and joint policies have the same expected reference acquisition rate per observation, but differing label budgets; they are not matched in both resource costs. The joint policy is worse on certification rate and both observed resource counts here, so it supplies no evidence of an allocation improvement.

Skipping agreement labels saves about 82.2% of labels with exactly unchanged certification, but is a simple structural baseline rather than the sought contribution. The joint policy suffers from inverse-probability weighting and its global range bound. A better allocation rule and sharper valid bounds would need new testing and a genuine technical contribution beyond generic active risk control.

At boundary true excess .05, erroneous certifications were 4/300 for acquire-all and agreement-skip, and 3/300 for each reference-sampling policy. At true excess .08, all policies had 0/300. These are small diagnostics, not proof or a precise guarantee audit.

## Time and reproduction

Simulation and tests took 8.44 seconds excluding Python imports. No GPU was used. From the codebase folder in a clean installed environment:

```powershell
.\.venv\Scripts\python.exe selective_smoke.py
```

For the existing workspace installation, from the workspace root:

```powershell
$env:PYTHONPATH=(Resolve-Path work/deps).Path
python outputs/paired-certification/selective_smoke.py
```

Estimated next steps, not measured commitments: 2–4 hours to implement and validate a focused adaptive-allocation pilot, with roughly 5–30 minutes of CPU experiments at this scale. Establishing a defensible novel result, stronger baselines and broader evidence could take 1–3 weeks or longer; it may still yield a negative result. Novelty and a useful theorem are the main uncertainties, rather than compute availability.
