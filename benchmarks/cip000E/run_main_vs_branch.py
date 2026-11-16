#!/usr/bin/env python3
"""CIP-000E: compare public defaults on ``main`` vs this branch.

Runs ``bench_suite.py`` under ``uv run`` in each checkout, then emits a
markdown report (and optional JSON) with ×branch/main ratios.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _run_suite(repo: Path, suite: Path, label: str, reps: int, seed: int, skip_models: bool) -> dict:
    # Copy the suite into the target repo so ``uv run`` resolves *that* project's
    # editable GPy (uv discovers the project from the script path, not cwd alone).
    local_suite = repo / "benchmarks" / "cip000E" / "_bench_suite_run.py"
    local_suite.parent.mkdir(parents=True, exist_ok=True)
    local_suite.write_text(suite.read_text())
    cmd = [
        "uv",
        "run",
        "--project",
        str(repo),
        "python",
        str(local_suite),
        "--label",
        label,
        "--reps",
        str(reps),
        "--seed",
        str(seed),
    ]
    if skip_models:
        cmd.append("--skip-models")
    # Drop workspace PYTHONPATH so each tree's editable install wins.
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    finally:
        try:
            local_suite.unlink()
        except OSError:
            pass
    # suite prints JSON; ignore any leading warnings by finding first '{'
    text = proc.stdout
    start = text.find("{")
    if start < 0:
        raise RuntimeError(f"No JSON from {label}:\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return json.loads(text[start:])


def _row_key(row: dict) -> tuple:
    return (
        row["bench"],
        row.get("N"),
        row.get("M"),
        row.get("Q"),
        row.get("D"),
    )


def _fmt_sec(x: float) -> str:
    if x >= 1:
        return f"{x:.3f}s"
    if x >= 1e-3:
        return f"{x * 1e3:.2f}ms"
    return f"{x * 1e6:.1f}µs"


def _markdown(main: dict, branch: dict) -> str:
    lines = []
    lines.append("# CIP-000E main vs branch benchmark")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    me, be = main["env"], branch["env"]
    lines.append(f"- **Host:** `{me.get('platform')}` (`cpu_count={me.get('cpu_count')}`)")
    lines.append(f"- **Python / NumPy / SciPy (main):** {me.get('python')} / {me.get('numpy')} / {me.get('scipy')}")
    lines.append(
        f"- **Python / NumPy / SciPy (branch):** {be.get('python')} / {be.get('numpy')} / {be.get('scipy')}"
    )
    blas = me.get("blas") or {}
    lines.append(f"- **BLAS (main probe):** `{blas.get('name', blas)}`")
    lines.append(
        f"- **Thread env:** OMP={me.get('OMP_NUM_THREADS')} "
        f"OPENBLAS={me.get('OPENBLAS_NUM_THREADS')} "
        f"VECLIB={me.get('VECLIB_MAXIMUM_THREADS')}"
    )
    lines.append("")
    lines.append("## Git / defaults")
    lines.append("")
    lines.append(
        f"| Tree | Branch | Commit | `backprop` | `flat_to_triang` | cython flags |"
    )
    lines.append("|---|---|---|---|---|---|")
    for label, payload in (("main", main), ("branch", branch)):
        g, d = payload["git"], payload["defaults"]
        flags = (
            f"chol={d.get('use_choleskies_cython')} "
            f"lin={d.get('use_linalg_cython')} "
            f"stat={d.get('use_stationary_cython')} "
            f"coreg={d.get('use_coregionalize_cython')}"
        )
        lines.append(
            f"| {label} | `{g.get('branch')}` | `{g.get('commit')}` | "
            f"`{d.get('backprop_gradient')}` | `{d.get('flat_to_triang')}` | {flags} |"
        )
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "Times are mean wall-clock over "
        f"**{main.get('reps')}** reps (same seeds). "
        "**× = branch / main** (lower means branch is faster)."
    )
    lines.append("")
    lines.append("| Bench | Shape | Main | Branch | × (branch/main) | Main impl | Branch impl |")
    lines.append("|---|---|---:|---:|---:|---|---|")

    main_map = {_row_key(r): r for r in main["rows"]}
    branch_map = {_row_key(r): r for r in branch["rows"]}
    keys = list(dict.fromkeys(list(main_map) + list(branch_map)))
    for key in keys:
        mr, br = main_map.get(key), branch_map.get(key)
        if not mr or not br:
            continue
        bench = mr["bench"]
        shape_bits = []
        for k in ("N", "M", "Q", "D"):
            if mr.get(k) is not None:
                shape_bits.append(f"{k}={mr[k]}")
        shape = ", ".join(shape_bits)
        ratio = br["seconds"] / mr["seconds"] if mr["seconds"] > 0 else float("nan")
        lines.append(
            f"| `{bench}` | {shape} | {_fmt_sec(mr['seconds'])} | {_fmt_sec(br['seconds'])} | "
            f"**{ratio:.2f}×** | `{mr.get('impl')}` | `{br.get('impl')}` |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Compares **shipped defaults**, not best-of-tree cherry-picks: "
        "`main` uses Cython where built; branch uses NumPy/BLAS primitives "
        "after CIP-000E Phase 1–2."
    )
    lines.append(
        "- Model benches (`gp_*`, `sparse_gp_*`) mix factorisation, likelihood, "
        "and kernel work — they show end-to-end impact, not a pure hotspot ratio."
    )
    lines.append(
        "- Absolute times are host-specific (Apple Accelerate on this Mac). "
        "Re-run on Linux multicore before treating ratios as universal."
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    branch_root = here.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--main-root",
        type=Path,
        default=Path("/Users/neil/SheffieldML/GPy"),
        help="Checkout of GPy on main (with Cython built)",
    )
    p.add_argument("--branch-root", type=Path, default=branch_root)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-models", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=here.parent / "results",
        help="Directory for JSON + markdown artefacts",
    )
    args = p.parse_args(argv)

    suite = here.parent / "bench_suite.py"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running suite on main ({args.main_root}) ...", file=sys.stderr)
    main_payload = _run_suite(args.main_root, suite, "main", args.reps, args.seed, args.skip_models)
    print(f"Running suite on branch ({args.branch_root}) ...", file=sys.stderr)
    branch_payload = _run_suite(
        args.branch_root, suite, "branch", args.reps, args.seed, args.skip_models
    )

    combined = {"main": main_payload, "branch": branch_payload}
    json_path = args.out_dir / "main_vs_branch.json"
    md_path = args.out_dir / "main_vs_branch.md"
    json_path.write_text(json.dumps(combined, indent=2, default=str) + "\n")
    md = _markdown(main_payload, branch_payload)
    md_path.write_text(md + "\n")
    print(md)
    print(f"\nWrote {json_path} and {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
