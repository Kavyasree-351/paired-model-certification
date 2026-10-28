# Verified execution record

26 October 2026, Python 3.12.6 on Windows; exact scientific-library versions in requirements-lock.txt.

- Six correctness tests passed in the final run: exact small-sample e-value expectation enumeration; extreme outcomes; fixed-sequence stopping; independent boundary-null simulation; globally isolated splits across seeds; permutation invariance of the ternary likelihood.
- Original pilot: 4 datasets × 3 seeds × 8 methods × 2 corrections, plus two noise conditions for every method. All completed.
- Original fresh confirmation: same conditions, three new training seeds and globally separate evaluation rows. All completed.
- Two independent-tuning alternatives: all 12 dataset–seed pairs in each phase. All completed.
- Matched component ablations: all 12 original development pairs; explicitly post hoc, never used to open final data.
- Controlled simulation: 36 scenarios × 300 IID draws; all 8 methods use matched draws. Marginal exact-binomial intervals saved.
- Raw-outcome audit: cascade/reference accuracy difference matches signed excess error; three seeds retained for every dataset; every final pool is globally disjoint from training, tuning, calibration, pilot and fresh pools.
- Full-run guard exercised. It exits with `Full evaluation denied: pilot gate has not passed.` before data loading. No final predictions/results were produced.
- CSV and TeX tables, PNG and vector-PDF figures generated from saved outcomes. The power figure was visually checked and its tick/legend layout corrected.

Observed opportunity gates: FAIL in pilot and fresh confirmation. Original paired gains -.018695 and -.014243. No defensible new-method or conference-quality result claimed. No final test outcome inferred from pilot results.

Known limits: tiny real datasets; conditional training-seed intervals; shared calibration pools; no exact reproduction of the closest WSR baseline; simple robustness perturbations; estimated batch inference costs rather than measured serving latency; no energy measurements. These prevent broad scientific or deployment claims.
