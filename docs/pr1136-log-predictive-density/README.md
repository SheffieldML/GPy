# PR #1136 log_predictive_density verification

devel problems:
1. Integrand returned a 1-element ndarray → TypeError under NumPy 2 `quad`
2. Non-finite values hit leftover `ipdb.set_trace()`
3. ±∞ bounds evaluated overflowing link/logpdf regions

PR: `float(np.squeeze(res))`, remove ipdb, integrate over μ±20σ.

Figures: 00_summary, 01_broken_examples, 02_fixed_vs_trapezoid, 03_integration_window
