#!/usr/bin/env python3
"""Cross-branch numerical stability comparison (CIP-0011).

Runs the *same* suite against two installed GPy trees (baseline vs candidate)
and classifies each test nodeid as improvement / regression / stable / debt.

Exit code is non-zero iff any regression (or suite drift under --strict-suite).

Example::

    uv run python numerical_stability/compare.py \\
      --baseline main --candidate HEAD --suite-from candidate \\
      --report numerical_stability/last-compare.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

# Outcomes recorded by the embedded pytest plugin / fixtures.
VALID_OUTCOMES = frozenset(
    {"passed", "failed", "error", "skipped", "xfailed", "xpassed"}
)

# Classification labels for the markdown report.
IMPROVEMENT = "improvement"
REGRESSION = "regression"
STABLE = "stable"
KNOWN_DEBT = "known_debt"
IGNORE = "ignore"
SUITE_DRIFT = "suite_drift"


def classify(baseline: str | None, candidate: str | None) -> str:
    """Classify one nodeid from baseline vs candidate outcomes.

    ``None`` means the test was absent from that run (suite drift).
    """
    if baseline is None or candidate is None:
        if baseline == candidate:
            return IGNORE  # both absent — should not happen in practice
        return SUITE_DRIFT

    if baseline not in VALID_OUTCOMES or candidate not in VALID_OUTCOMES:
        raise ValueError(f"unknown outcomes: {baseline!r} vs {candidate!r}")

    if baseline == "skipped" and candidate == "skipped":
        return IGNORE

    # Improvements: known defect cleared (xfail -> pass / xpass).
    if baseline == "xfailed" and candidate in {"passed", "xpassed"}:
        return IMPROVEMENT

    # Regressions: was green (or expected-fail) and is now unexpectedly broken.
    if baseline == "passed" and candidate in {"failed", "error"}:
        return REGRESSION
    if baseline == "xfailed" and candidate in {"failed", "error"}:
        return REGRESSION
    if baseline == "skipped" and candidate in {"failed", "error"}:
        return REGRESSION

    if baseline == "xfailed" and candidate == "xfailed":
        return KNOWN_DEBT

    if baseline == candidate:
        return STABLE

    # Soft transitions (e.g. skipped <-> passed) are informational, not regressions.
    if {baseline, candidate} <= {"passed", "skipped", "xpassed"}:
        return STABLE if baseline == candidate else IMPROVEMENT if candidate == "passed" else STABLE

    if baseline in {"failed", "error"} and candidate in {"passed", "xfailed", "xpassed"}:
        return IMPROVEMENT

    if baseline in {"failed", "error"} and candidate in {"failed", "error"}:
        return STABLE

    return STABLE


def diff_outcomes(
    baseline: dict[str, str],
    candidate: dict[str, str],
) -> dict[str, list[tuple[str, str | None, str | None]]]:
    """Return classification -> list of (nodeid, baseline_outcome, candidate_outcome)."""
    buckets: dict[str, list[tuple[str, str | None, str | None]]] = defaultdict(list)
    for nodeid in sorted(set(baseline) | set(candidate)):
        b = baseline.get(nodeid)
        c = candidate.get(nodeid)
        buckets[classify(b, c)].append((nodeid, b, c))
    return dict(buckets)


def render_report(
    buckets: dict[str, list[tuple[str, str | None, str | None]]],
    *,
    baseline_ref: str,
    candidate_ref: str,
    suite_from: str,
) -> str:
    """Markdown summary for humans / PR artifacts."""
    lines = [
        "# Numerical stability comparison (CIP-0011)",
        "",
        f"- **Baseline:** `{baseline_ref}`",
        f"- **Candidate:** `{candidate_ref}`",
        f"- **Suite from:** `{suite_from}`",
        "",
    ]
    order = [IMPROVEMENT, REGRESSION, KNOWN_DEBT, SUITE_DRIFT, STABLE, IGNORE]
    titles = {
        IMPROVEMENT: "Improvements (xfail → pass)",
        REGRESSION: "Regressions",
        KNOWN_DEBT: "Known debt (still xfail)",
        SUITE_DRIFT: "Suite drift (tests only on one side)",
        STABLE: "Stable",
        IGNORE: "Ignored (skipped / absent)",
    }
    for key in order:
        items = buckets.get(key, [])
        lines.append(f"## {titles[key]} ({len(items)})")
        lines.append("")
        if not items:
            lines.append("_None._")
            lines.append("")
            continue
        # Keep stable/ignore lists short in the report body.
        show = items if key not in {STABLE, IGNORE} else items[:20]
        for nodeid, b, c in show:
            lines.append(f"- `{nodeid}`: {b} → {c}")
        if key in {STABLE, IGNORE} and len(items) > 20:
            lines.append(f"- … and {len(items) - 20} more")
        lines.append("")

    n_reg = len(buckets.get(REGRESSION, []))
    n_imp = len(buckets.get(IMPROVEMENT, []))
    lines.append("## Verdict")
    lines.append("")
    if n_reg:
        lines.append(f"**Fail:** {n_reg} regression(s), {n_imp} improvement(s).")
    else:
        lines.append(f"**Pass:** {n_imp} improvement(s), no regressions.")
    lines.append("")
    return "\n".join(lines)


def exit_code_for(
    buckets: dict[str, list],
    *,
    strict_suite: bool,
) -> int:
    if buckets.get(REGRESSION):
        return 1
    if strict_suite and buckets.get(SUITE_DRIFT):
        return 1
    return 0


# ---------------------------------------------------------------------------
# Runner: worktrees + uv + embedded pytest plugin
# ---------------------------------------------------------------------------

_OUTCOMES_PLUGIN = r"""
import json
import pytest

_outcomes = {}

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call" and not (report.when == "setup" and report.skipped):
        return
    nodeid = report.nodeid
    if getattr(report, "wasxfail", None):
        if report.skipped:
            _outcomes[nodeid] = "xfailed"
        elif report.passed:
            _outcomes[nodeid] = "xpassed"
        else:
            _outcomes[nodeid] = "failed"
    elif report.skipped:
        _outcomes[nodeid] = "skipped"
    elif report.failed:
        _outcomes[nodeid] = "failed" if report.when == "call" else "error"
    elif report.passed and report.when == "call":
        _outcomes[nodeid] = "passed"

def pytest_sessionfinish(session, exitstatus):
    path = session.config.getoption("--cip0011-outcomes")
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_outcomes, fh, indent=2, sort_keys=True)
"""


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _git_root() -> Path:
    out = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(out.strip())


def _resolve_ref(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", "--verify", ref], text=True).strip()


def _add_worktree(path: Path, ref: str) -> None:
    # Clear stale registrations from interrupted runs.
    subprocess.run(["git", "worktree", "prune"], check=False, capture_output=True)
    if path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(path, ignore_errors=True)
    _run(["git", "worktree", "add", "--detach", str(path), ref])


def _remove_worktree(path: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(path)],
        check=False,
        capture_output=True,
    )
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], check=False, capture_output=True)


def _suite_dir_from(
    suite_from: str,
    *,
    baseline_wt: Path,
    candidate_wt: Path,
) -> Path:
    if suite_from == "baseline":
        return baseline_wt / "numerical_stability"
    if suite_from == "candidate":
        return candidate_wt / "numerical_stability"
    path = Path(suite_from).expanduser().resolve()
    if path.is_file():
        # Allow pointing at a single test module; use its parent as suite dir.
        return path.parent
    return path


def _copy_suite(src: Path, dest: Path) -> Path:
    if dest.exists():
        shutil.rmtree(dest)
    if not src.exists():
        raise FileNotFoundError(
            f"suite directory not found: {src}\n"
            "Pass --suite-from candidate|baseline|PATH once numerical_stability/ exists on that ref."
        )
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "last-compare.md", ".pytest_cache"),
    )
    return dest


def _run_suite_on_tree(
    tree: Path,
    suite_copy: Path,
    outcomes_path: Path,
    plugin_path: Path,
    *,
    suite_root: Path,
) -> dict[str, str]:
    """Install tree with uv and run the suite; return nodeid -> outcome."""
    _run(["uv", "sync"], cwd=tree)
    env = os.environ.copy()
    # suite_root parents the copied ``numerical_stability`` package; tree supplies GPy.
    env["PYTHONPATH"] = (
        str(suite_root)
        + os.pathsep
        + str(tree)
        + os.pathsep
        + str(plugin_path.parent)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    cmd = [
        "uv",
        "run",
        "--directory",
        str(tree),
        "pytest",
        str(suite_copy),
        "-q",
        "--tb=no",
        "-p",
        "no:cacheprovider",
        "-p",
        plugin_path.stem,
        f"--cip0011-outcomes={outcomes_path}",
    ]
    print("+", " ".join(cmd), flush=True)
    # pytest may exit 1 on unexpected failures; still collect outcomes.
    proc = subprocess.run(cmd, cwd=tree, env=env)
    if not outcomes_path.exists():
        raise RuntimeError(
            f"outcomes file missing after pytest (exit {proc.returncode}): {outcomes_path}"
        )
    data = json.loads(outcomes_path.read_text(encoding="utf-8"))
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"pytest crashed with exit {proc.returncode}")
    return {k: v for k, v in data.items() if v in VALID_OUTCOMES}


def _write_plugin(dir_path: Path) -> Path:
    plugin = dir_path / "cip0011_outcomes_plugin.py"
    # Register the CLI option via a tiny conftest-style plugin module.
    plugin.write_text(
        _OUTCOMES_PLUGIN
        + """

def pytest_addoption(parser):
    parser.addoption("--cip0011-outcomes", default="", help="write CIP-0011 outcomes JSON")
""",
        encoding="utf-8",
    )
    return plugin


def run_compare(args: argparse.Namespace) -> int:
    root = _git_root()
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else Path(
        tempfile.mkdtemp(prefix="gpy-cip0011-compare-")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    baseline_wt = work_dir / "baseline"
    candidate_wt = work_dir / "candidate"
    # Preserve the package name so test_compare can `import numerical_stability.compare`.
    suite_copy = work_dir / "numerical_stability"
    plugin_path = _write_plugin(work_dir)
    keep = args.keep_work_dir

    try:
        base_sha = _resolve_ref(args.baseline)
        cand_sha = _resolve_ref(args.candidate)
        print(f"baseline {args.baseline} -> {base_sha[:12]}")
        print(f"candidate {args.candidate} -> {cand_sha[:12]}")

        _add_worktree(baseline_wt, base_sha)
        _add_worktree(candidate_wt, cand_sha)

        suite_src = _suite_dir_from(
            args.suite_from,
            baseline_wt=baseline_wt,
            candidate_wt=candidate_wt,
        )
        # If suite_from is an absolute path outside worktrees (e.g. current branch
        # draft), prefer that over missing numerical_stability/ on old refs.
        if args.suite_from not in {"baseline", "candidate"}:
            suite_src = Path(args.suite_from).expanduser().resolve()
            if suite_src.is_file():
                suite_src = suite_src.parent
        _copy_suite(suite_src, suite_copy)

        base_out = work_dir / "baseline-outcomes.json"
        cand_out = work_dir / "candidate-outcomes.json"
        baseline_outcomes = _run_suite_on_tree(
            baseline_wt, suite_copy, base_out, plugin_path, suite_root=work_dir
        )
        candidate_outcomes = _run_suite_on_tree(
            candidate_wt, suite_copy, cand_out, plugin_path, suite_root=work_dir
        )

        buckets = diff_outcomes(baseline_outcomes, candidate_outcomes)
        report = render_report(
            buckets,
            baseline_ref=f"{args.baseline} ({base_sha[:12]})",
            candidate_ref=f"{args.candidate} ({cand_sha[:12]})",
            suite_from=args.suite_from,
        )
        report_path = Path(args.report).expanduser() if args.report else work_dir / "compare-report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(report)
        print(f"wrote {report_path}")
        return exit_code_for(buckets, strict_suite=args.strict_suite)
    finally:
        if not keep:
            _remove_worktree(baseline_wt)
            _remove_worktree(candidate_wt)
            # Leave work_dir if it was user-supplied; always try to clean temp plugin debris.
            if not args.work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"kept work dir: {work_dir}")
        # Silence unused root in some paths.
        _ = root


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="origin/main", help="git ref for baseline install")
    p.add_argument("--candidate", default="HEAD", help="git ref for candidate install")
    p.add_argument(
        "--suite-from",
        default="baseline",
        help="baseline | candidate | PATH to numerical_stability/ (or a test module)",
    )
    p.add_argument("--work-dir", default="", help="scratch directory for worktrees / venvs")
    p.add_argument("--keep-work-dir", action="store_true", help="do not delete scratch trees")
    p.add_argument("--report", default="", help="markdown report path")
    p.add_argument(
        "--strict-suite",
        action="store_true",
        help="treat suite drift (tests only on one side) as failure",
    )
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return run_compare(args)


if __name__ == "__main__":
    sys.exit(main())
