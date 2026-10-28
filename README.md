# Paired Model Certification

A CPU research-engineering project for evaluating when a cheaper classifier can replace a reference model while limiting expected excess classification error.

The repository includes paired-risk certificates, matched-budget baselines, reproducible dataset splits, multiple seeds, selective-labeling investigations, robustness checks, saved results, and plotting scripts. The central method hypothesis did not outperform strong baselines consistently; the negative findings and limitations are retained.

Run the bounded publication checks with `python run_checks.py`. They do not launch full experiments or evaluate reserved final data. See [PUBLICATION.md](PUBLICATION.md) for scope and license status, [THIRD_PARTY.md](THIRD_PARTY.md) for attribution, and [VERIFICATION.md](VERIFICATION.md) for the original execution record.

## Reproducible feasibility study

Latest staged-acquisition investigation: see STAGED_ACQUISITION_ASSESSMENT.md. Run `selective_smoke.py`, `staged_audit.py`, then `staged_report.py` to reproduce its checks, matched-budget simulations and tables. No complete paper for this extension is claimed.

Reference-revealed information follow-up: see REFERENCE_REVELATION_ASSESSMENT.md. Run `revelation_audit.py --stage pilot` and `revelation_report.py`. This separates practical usefulness from research novelty and includes the matched precommit baseline and classical two-phase sampling comparison.

**Status:** the proposed ternary certificate failed the predefined pilot gate. No defensible new-method claim has been established. Do not use this package as evidence of an improved algorithm or a submission-ready paper. Full-test results are intentionally pending.

The engineering problem is to route examples to a cheap classifier while limiting excess classification error relative to a reference. The statistical candidate models paired harmful, beneficial and tied outcomes explicitly. Known e-value, universal-inference and fixed-sequence-testing principles are credited in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). That file compares directions, cites closest work, states assumptions and defines the gate. [ATTEMPTS.md](ATTEMPTS.md) records failures and alternatives. [PAPER_DRAFT.md](PAPER_DRAFT.md) reports verified findings and pending evidence.

## Run from a clean environment

Python 3.12, CPU, no GPU or online dataset downloads required. The three real datasets are bundled with scikit-learn. Run the following inside this folder. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest discover -s . -p test_correctness.py -v
.\.venv\Scripts\python.exe research.py run --stage pilot
.\.venv\Scripts\python.exe alternatives.py --stage pilot
.\.venv\Scripts\python.exe simulate.py --reps-per-seed 100
.\.venv\Scripts\python.exe ablate.py
# Freeze all choices before opening fresh evaluation data:
.\.venv\Scripts\python.exe research.py run --stage fresh
.\.venv\Scripts\python.exe alternatives.py --stage fresh
```

On Linux/macOS replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`. Exact installed dependencies are in requirements-lock.txt; requirements.txt lists direct scientific dependencies. All reported runs used the lockfile versions. The original installation was isolated in the workspace's work/deps folder; it did not replace system packages.

## Full experiments, aggregation and figures

```powershell
# This correctly REFUSES to run with the delivered failed pilot gate.
# Do not edit gate.json or lower the gate to force evaluation.
.\.venv\Scripts\python.exe research.py run --stage full
.\.venv\Scripts\python.exe research.py aggregate --stage full
# Regenerate the verified pilot/fresh tables and plots:
.\.venv\Scripts\python.exe research.py aggregate --stage pilot
.\.venv\Scripts\python.exe research.py aggregate --stage fresh
.\.venv\Scripts\python.exe report.py
```

Full mode is implemented for 20 predefined seeds but requires both pilot and fresh gates to pass. A scientifically distinct future method needs a new preregistration and new development/confirmation data; it must not repeatedly optimize against the now-open fresh pool. Final rows remain reserved globally across all training seeds. No bypass flag is provided.

Each stage writes results/STAGE/raw.json (all methods/seeds), manifest.json (data hashes, exact splits, training/tuning and inference times, parameters, protocol hash), summary.json, gate.json, table.csv, table.tex, routing.pdf and routing.png. Figures are exportable vector PDF and PNG; the table's intervals describe training-seed variation on fixed data pools. They are not dataset-level inferential intervals. `report.py` creates a verified comparison table and simulation figure. Alternatives and controlled simulations have separate raw JSON files.

The primary metric is the fraction routed under a 95% confidence certificate for at most five percentage points of expected excess error. Batch timing estimates include cheap-model cost; they do not measure routed serving latency. Runtime, noise robustness, baseline accuracies and cascade errors are retained even if unfavorable. Naive empirical selection has no safety certificate and is never counted as a valid baseline.

## Practical scope

Pilot and fresh runs each contain 12 dataset–seed pairs with 6 model fits per pair (three settings for each model), plus certificate comparisons. A single CPU machine is sufficient. The included execution report gives measured totals; timing varies with machine and system load. Controlled simulation uses 36 scenarios, 300 independent draws per scenario, eight methods. Seeds are fixed and all outcomes are retained.

The closest NeurIPS method already handles signed losses and WSR betting. Included generic betting implementations are useful strong comparators but not an exact reproduction of that authors' code. Such a reproduction, larger datasets, independently repeated calibration pools, genuine serving measurements and stronger novelty evidence are required before a main-method paper. The narrow negative pilot is not an A* result by itself.
