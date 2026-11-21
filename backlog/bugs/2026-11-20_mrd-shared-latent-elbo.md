---
id: 2026-11-20_mrd-shared-latent-elbo
title: MRD shared-latent MiniBatch wiring ELBO slightly below fresh views
status: Proposed
priority: Low
created: '2026-11-20'
last_updated: '2026-11-20'
category: bugs
related_cips: []
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- mrd
- elbo
- vardtc
- minibatch
---

# Task: MRD shared-latent ELBO parity quirk

## Description

Worth a **small** look in GPy, not a crisis — and not a reason to change
`CMrd` (GPc / CIP-0017 reference path).

What we saw:

1. **Standalone** `BayesianGPLVMMiniBatch` with `kl_factr=1` matches
   `BayesianGPLVM` exactly (same VarDTC + KL).
2. **`kl_factr=1/V` is intentional** — that is how GPy MRD counts the KL
   once across views.
3. The odd bit is **MRD-specific**: after children share the parent’s
   `X`/`Z` (`unlink` / `del` / rebind), each child’s VarDTC term is
   ~0.28 nats below a fresh MiniBatch with the same `Y`, kernel, noise,
   and latents. That is what made raw `GPy.models.MRD` miss the `CMrd` /
   CIP-0017 path by ~0.58 (two views).

So the intended math (∑ VarDTC − KL once) is fine; the **shared-latent
MiniBatch wiring** looks slightly wrong or at least inconsistent.
Plausible causes: update order when several children own the same `X`/`Z`,
or leftover state from the parent `SparseGPMiniBatch` init. The module
even labels itself bleeding edge.

**Practical take:** good candidate for a GPy issue/PR if we care about
ELBO parity; low urgency for modelling (grads still train). GPc gating on
a BayesianGPLVM pair is the right reference — that is the stable VarDTC
path, not MiniBatch MRD’s quirk.

Related (fixed separately): constructor `Z=` was ignored —
`2026-11-20_mrd-ignores-constructor-z`.

## Acceptance Criteria

- [ ] Reproduce ~0.28 nat VarDTC gap per view after MRD share-rebind vs
      fresh `BayesianGPLVMMiniBatch` with identical parameters
- [ ] Identify cause (update order vs leftover parent MiniBatch state vs
      other)
- [ ] Either fix ELBO parity or document the known offset in MRD docs /
      docstring
- [ ] No change required to GPc `CMrd` unless the fix changes the
      intended ∑ VarDTC − KL contract

## Implementation Notes

- Hotspot: `GPy/models/mrd.py` child construction — `kl_factr = 1/V`,
  then `unlink`/`del`/`rebind` of `X`/`Z` onto the parent params.
- Compare against standalone MiniBatch and against `BayesianGPLVM` (stable
  VarDTC reference).

## Related

- Backlog: `2026-11-20_mrd-ignores-constructor-z` (constructor `Z=`)
- External: GPc `CMrd` / CIP-0017 ELBO gating
- Code comments: `GPy/models/mrd.py` (kl_factr + share-rebind)

## Progress Updates

### 2026-11-20

Comments captured from MRD vs `CMrd` ELBO comparison while investigating
constructor `Z=`. Status Proposed — low urgency; not blocking modelling.
