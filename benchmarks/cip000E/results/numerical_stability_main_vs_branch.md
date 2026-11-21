# Numerical stability: main vs CIP-000E branch

**Date:** 2026-11-17  
**Tool:** CIP-0011 `numerical_stability/compare.py`  
**Suite source:** branch `cip0011-numerical-stability` @ `9739a6b6` (suite not yet on either compared tree; `--suite-from` used)  
**Baseline:** `origin/main` @ `c8a095cd`  
**Candidate:** `origin/cip000E-numpy-blas-primitives` @ `533a8e9f`  
**Platform:** macOS arm64, Python 3.11.7, NumPy 2.4.3 / SciPy 1.16.3 (separate venvs under `/tmp/gpy-num-compare-main-vs-cythonless/`)

## Verdict

**Pass — outcome-identical.** 0 improvements, 0 regressions across 408 collected tests. Both trees share the same **20 xfailed** catalogue entries from CIP-0011/CIP-0012 (distance cancellation, Bernoulli/EP, softplus, cloglog, Gamma, Poisson, jitchol, bound gradients, debugger breakpoints).

**Implication for CIP-000E:** replacing Cython kernel/linalg paths with NumPy/BLAS does **not**, by itself, change these numerical-stability outcomes. Defect remediation belongs in CIP-0012 (fixes landed on `cip0011-numerical-stability` after this compare). Re-run after cherry-picking CIP-0012 onto this branch to confirm xfails clear without introducing outcome diffs vs post-fix `main`.

## How to reproduce

```bash
# From a checkout that has numerical_stability/ (CIP-0011):
uv run python numerical_stability/compare.py \
  --baseline /path/to/main-worktree \
  --candidate /path/to/cip000E-worktree \
  --suite-from /path/to/cip0011-checkout \
  --out /tmp/gpy-num-compare-main-vs-cythonless
```

Machine-readable summary: [`numerical_stability_main_vs_branch_summary.json`](numerical_stability_main_vs_branch_summary.json).

---

# Numerical stability comparison (CIP-0011)

- **Baseline:** `origin/main (c8a095cdce7a)`
- **Candidate:** `origin/cip000E-numpy-blas-primitives (533a8e9fd3fb)`
- **Suite from:** `/Users/neil/SheffieldML/GPy-cip0011-numerical-stability/numerical_stability`

## Improvements (xfail → pass) (0)

_None._

## Regressions (0)

_None._

## Known debt (still xfail) (20)

- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_all_derivatives_finite_for_confident_mislabels`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_extreme_cavity_keeps_positive_variance`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_gradient_tail_is_inverse_mills_ratio`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_logpdf_tail_accuracy`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestCountAndPositiveLikelihoods::test_gamma_logpdf_finite_for_large_shape`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestCountAndPositiveLikelihoods::test_poisson_deep_negative_latent`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestExactGPRegression::test_all_parameters_at_transform_floor`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestJitchol::test_nonfinite_input_raises[inf_diag]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::TestNonGaussianInference::test_gamma_laplace_on_ordinary_positive_data`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_K_with_X2_equal_X_is_consistent[Exponential]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_K_with_X2_equal_X_is_consistent[OU]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_cloglog_relative_accuracy_in_lower_tail`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_no_live_debugger_breakpoints_in_package`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_softplus_asymptote_for_large_f`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1000000.0-Exponential]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1000000.0-Matern32]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1000000.0-RBF]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1700000000.0-Exponential]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1700000000.0-Matern32]`: xfailed → xfailed
- `numerical_stability/test_numerical_stability.py::test_stationary_translation_invariance_large_offset[1700000000.0-RBF]`: xfailed → xfailed

## Suite drift (tests only on one side) (0)

_None._

## Stable (387)

- `numerical_stability/test_compare.py::test_classify_matrix[None-passed-suite_drift]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[failed-passed-improvement]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[passed-None-suite_drift]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[passed-error-regression]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[passed-failed-regression]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[passed-passed-stable]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[skipped-skipped-ignore]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[xfailed-error-regression]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[xfailed-failed-regression]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[xfailed-passed-improvement]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[xfailed-xfailed-known_debt]`: passed → passed
- `numerical_stability/test_compare.py::test_classify_matrix[xfailed-xpassed-improvement]`: passed → passed
- `numerical_stability/test_compare.py::test_diff_outcomes_buckets`: passed → passed
- `numerical_stability/test_compare.py::test_exit_code_regressions_and_strict_suite`: passed → passed
- `numerical_stability/test_compare.py::test_render_report_mentions_refs_and_verdict`: passed → passed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_tail_variance_asymptote`: passed → passed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_valid[0.0--1.0--2.0]`: passed → passed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_valid[0.0--1.0--4.0]`: passed → passed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_valid[0.0--1.0--6.0]`: passed → passed
- `numerical_stability/test_numerical_stability.py::TestBernoulli::test_ep_moment_match_valid[0.0--1.0--8.0]`: passed → passed
- … and 367 more

## Ignored (skipped / absent) (1)

- `numerical_stability/test_numerical_stability.py::TestSparseAndStochastic::test_svgp_extreme_lengthscale`: skipped → skipped

## Verdict

**Pass:** 0 improvement(s), no regressions.
