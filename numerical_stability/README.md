# Numerical stability suite (CIP-0011)

Adversarial / extreme-input checks for GPy, **separate** from unit tests in
`GPy/testing`. Known defects are marked `xfail(strict=True)` until
[CIP-0012](../cip/cip0012.md) removes them.

## Run

```bash
# Unit tests only (unchanged)
uv run pytest GPy/testing

# This suite only (~seconds after install)
uv run pytest numerical_stability/ -q

# Both
uv run pytest GPy/testing numerical_stability/
```

Marker: `@pytest.mark.numerical` (registered in `pyproject.toml` and
`conftest.py`).

## Cross-branch comparison

Same suite source → two installs → outcome diff (improvement vs regression):

```bash
uv run python numerical_stability/compare.py \
  --baseline main \
  --candidate HEAD \
  --suite-from candidate \
  --report numerical_stability/last-compare.md
```

| Flag | Meaning |
|------|---------|
| `--baseline REF` | Code under test A (default `origin/main`) |
| `--candidate REF` | Code under test B (default `HEAD`) |
| `--suite-from` | `baseline` \| `candidate` \| `PATH` to this directory (default `baseline` once the suite is on `main`) |
| `--strict-suite` | Fail on suite drift (tests present on only one side) |
| `--keep-work-dir` | Keep scratch worktrees / outcome JSON |

**Exit code:** non-zero on regressions (and on suite drift if `--strict-suite`).
Improvements alone exit 0.

On this branch (suite not yet on `main`), use `--suite-from candidate` or
`--suite-from numerical_stability`.

## XPASS policy

`xfail(strict=True)` means an unexpected pass **fails** the job until the
marker is removed in the same PR as the fix (CIP-0012).

## Layout

```text
numerical_stability/
  README.md
  conftest.py
  compare.py                 # cross-branch runner
  test_compare.py            # outcome-matrix unit tests
  test_numerical_stability.py
```

## See also

- [CIP-0011](../cip/cip0011.md) — integration + comparer
- [CIP-0012](../cip/cip0012.md) — defect remediation
