# GPy demos

Runnable notebooks and scripts that show library behaviour with minimal setup.
Design context for the October 2026 batch: [CIP-0006](../cip/cip0006.md).

| Artifact | Topic |
|----------|--------|
| [october_2026_fixes.ipynb](october_2026_fixes.ipynb) | Serialization, Gamma/Laplace, predictive moments, Weibull, Symmetric, Bernoulli VE, LPD, MixedNoise LPD, normalizer (#1124–#1138) |
| [validate_pr1138_normalizer.py](validate_pr1138_normalizer.py) | Minimal check: `posterior_samples` / `log_predictive_density` vs `predict` with `normalizer=True` |

**Credit:** fix PRs in this batch were authored by [Raashish Aggarwal](https://github.com/raashish1601) (@raashish1601).

Install GPy from this repository (or a matching release), then open a notebook with Jupyter or VS Code/Cursor.
