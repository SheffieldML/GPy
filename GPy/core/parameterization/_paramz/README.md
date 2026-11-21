# Vendored paramz (CIP-000F Stage 4)

Sources from **paramz 0.10.0** (BSD 3-clause), vendored into GPy so the
external PyPI dependency can be dropped.

- Do not import this package as `GPy.core.parameterization._paramz` from
  user code; use `GPy.core.parameterization` (or the historical `paramz`
  aliases installed by `_paramz_bootstrap`).
- Upstream project: https://github.com/sods/paramz
