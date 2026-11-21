---
id: 2026-11-20_mrd-ignores-constructor-z
title: MRD constructor ignores user-supplied Z=
status: Completed
priority: Medium
created: '2026-11-20'
last_updated: '2026-11-20'
category: bugs
related_cips: []
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- mrd
- inducing-inputs
- gplvm
---


# Task: MRD ignores constructor `Z=`

## Description

`GPy.models.MRD` documents and accepts `Z` (initial inducing inputs), but
always overwrites it with `_init_Z(initz, X, input_dim)` before building
`self.Z`. Caller-supplied inducing locations are discarded.

`BayesianGPLVM` and `BayesianGPLVMMiniBatch` both guard with `if Z is None`
before sampling inducing points from `X`. The bug is specific to `MRD`.

## Acceptance Criteria

- [x] When `Z` is passed to `MRD`, `model.Z` matches the supplied array
- [x] When `Z is None`, behaviour remains `_init_Z` / `initz` as today
- [x] Regression test covers custom `Z=`
- [x] Confirm Bayesian GPLVM path already respects `Z=` (no change needed)

## Implementation Notes

- In `GPy/models/mrd.py`, mirror the `X` branch: only call `_init_Z` when
  `Z is None`.
- `self.num_inducing = self.Z.shape[0]` already runs after `Param` creation,
  so a custom `Z` with a different row count is fine.
- Bayesian GPLVM / minibatch already have `if Z is None:` — no change there.

## Related

- Files: `GPy/models/mrd.py`, `GPy/testing/test_model.py`
- Confirmed OK: `GPy/models/bayesian_gplvm.py`,
  `GPy/models/bayesian_gplvm_minibatch.py`
- Follow-up (separate): `2026-11-20_mrd-shared-latent-elbo` — after
  share-rebind of `X`/`Z`, each child’s VarDTC sits ~0.28 nats below a
  fresh MiniBatch; `kl_factr=1/V` is intentional; not a reason to change
  GPc `CMrd`. Low urgency.

## Progress Updates

### 2026-11-20

Task created after confirming MRD overwrites `Z=` unconditionally while
Bayesian GPLVM variants do not. Fix applied; `TestMisc::test_mrd` passes
with a custom-`Z` assertion.

Comments from the `CMrd` / CIP-0017 ELBO comparison filed as
`2026-11-20_mrd-shared-latent-elbo` (shared-latent wiring quirk, not
constructor `Z=`).
