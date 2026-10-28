# Attribution and external materials

The statistical constructions and nearest prior methods are credited in `RESEARCH_PLAN.md`, `PAPER_DRAFT.md`, and the follow-up assessments. This repository is not an exact reproduction of the closest authors' code and does not claim ownership of their methods.

Scientific dependencies are installed separately; their implementations are not vendored here:

- [NumPy](https://github.com/numpy/numpy)
- [SciPy](https://github.com/scipy/scipy)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [Matplotlib](https://github.com/matplotlib/matplotlib)

The real-data studies use scikit-learn's bundled digits, breast-cancer and wine loaders; synthetic data are generated locally. Dataset descriptions, provenance and references are available in the respective official loaders: [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), [breast cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html), [wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html). Raw datasets are not redistributed in this repository.

Plots and saved JSON/CSV tables are this project's experiment outputs, not copies of third-party paper figures. External papers are linked and cited rather than bundled.
