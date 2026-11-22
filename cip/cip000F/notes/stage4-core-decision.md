# CIP-000F Stage 4 — Core decision (Accept A)

**Date:** 2026-11-21  
**Decision:** **Accept Option A** — vendor remaining paramz core into
`GPy.core.parameterization._paramz`, drop the external `paramz` dependency,
keep `sys.modules` aliases so historical pickle paths resolve.

## Options (evidence)

| Option | Evidence after Stages 0–3 | Verdict |
|--------|---------------------------|----------|
| **A — Vendor** | Façade already owns caching, domains, ObsAr path, optimize dispatch. Remaining coupling is Param / Parameterized / Model / transformations / core observables (~26 modules). Vendoring removes the external pin without changing the user API. | **Accept (interim)** |
| **B — GPy-owned tree** | Needed for CIP-0002 tying and a static/dynamic split. Larger design; Stage 2N did not require it for GPy 1. | **Defer** — candidate follow-on CIP; do not fold CIP-0002 into this ship |
| **C — Stop after 3** | Viable if A were rejected; paramz would remain a runtime dependency behind the façade. | Rejected while A is viable |

## Accept record

- **Accepted:** Option A (vendor paramz 0.10.0 core as `_paramz`).
- **Not in this ship:** Option B redesign; CIP-0002 tying changes.
- **Pickle policy:** `paramz.*` names alias to the vendor tree via
  `_paramz_bootstrap.install_paramz_aliases()`.
- **Dependency:** `paramz>=0.10.0` removed from `pyproject.toml`; `decorator`
  added (required by vendored `Cache_this`).

## Implementation sketch

- Tree: `GPy/core/parameterization/_paramz/` (paramz 0.10.0 sources, no examples)
- Bootstrap imported first from `GPy.core.parameterization`
- Coverage / ruff omit the vendored tree; score bootstrap and façades instead
- GPy-owned `caching.py` / `domains.py` / `observable_array.py` remain the
  public Stage 1–3 surfaces; they import Observable / ObsAr through aliases
