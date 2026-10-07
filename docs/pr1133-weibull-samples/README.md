# PR #1133 Weibull samples verification

GPy `logpdf` matches `scipy.stats.weibull_min(r, scale=link(f)**(1/r))`.
On `devel`, `samples` used `scale=link(f)`, a different distribution.
This PR raises the scale to `1/r` so samples follow the likelihood.

Figures:
- `00_summary.png` — numeric summary
- `01_samples_vs_logpdf.png` — density overlays
- `02_mean_comparison.png` — means vs theory
