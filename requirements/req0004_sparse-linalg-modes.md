---
id: "0004"
title: "Sparse Gaussian inference exposes shared inducing-side factor modes with LQ default"
status: "Proposed"
priority: "High"
created: "2026-11-22"
last_updated: "2026-11-22"
related_tenets:
- "community-driven-development"
stakeholders:
- "GPy users of SparseGPRegression / VarDTC / FITC"
- "GPc / SheffieldML interop maintainers"
- "benchmark and numerical-stability maintainers"
tags:
- sparse
- linalg
- lq
- numerics
- interop
- gpc
---

# REQ-0004: Sparse Gaussian inference exposes shared inducing-side factor modes with LQ default

## Description

Users of GPy sparse Gaussian regression can choose how the inducing-side
factor of the collapsed bound is formed, using the same vocabulary as GPc
and the shared model-interchange sketch:

- **`lq`** (default once implemented): whitened features + LQ of
  \([I,\sqrt{\beta}B]\) (Seeger et al. 2017 BLR trick), for better
  conditioning than Gram+Cholesky of \(I+AA^\top\).
- **`stable`**: whitened \(C=I+\beta BB^\top\) (or FITC analogue) +
  Cholesky — GPy 1.x VarDTC/FITC arithmetic today.
- **`legacy`**: optional unwhitened / pre-stable form where GPy implements
  it for dual-path compares; otherwise documented as deferred or alias.

Approximation (which sparse model: VarDTC, FITC, DTC, …) stays on
`inference_method`. Factorisation mode is an orthogonal knob
(`sparse_linalg`), not a new inference subclass zoo.

**Why this matters**: GPc already ships `sparse_linalg` with default `lq`
and documents the shared tokens in GPc CIP-0016. Without the same surface
in GPy, benchmarks and YAML/JSON interchange cannot name the factor path,
and GPy users cannot opt into the more stable LQ route.

**Who benefits**: Sparse-GP users on ill-conditioned inducing layouts;
maintainers aligning GPy↔GPc benches and interchange; anyone comparing
modes without rewriting models.

## Acceptance Criteria

- [ ] `VarDTC`, `FITC`, and (as scoped) `DTC` accept `sparse_linalg=` at
      construction and expose a read/write `.sparse_linalg` property
- [ ] Canonical tokens are `lq` | `stable` | `legacy` (case-insensitive
      read; write canonical). Read aliases: `stablefactor`/`chol` → `stable`;
      `gram` → `legacy`
- [ ] Default is **`lq`** once the LQ path is implemented and gated;
      until then default may remain today's arithmetic labelled as
      `stable` only if the CIP documents a dated cutover
- [ ] No `VarDTC_LQ` / `FITC_LQ` subclasses — approximation stays the
      inference class
- [ ] On well-conditioned toys, `lq` and `stable` agree on log-likelihood
      (and gradients within documented FD tolerance)
- [ ] Exact / non-sparse inference ignores or rejects `sparse_linalg`
- [ ] Tokens match GPc façade / GPc CIP-0016 wire field `sparse_linalg`

## Notes (Optional)

HOW: [CIP-0013](../cip/cip0013.md). Design source of truth for the shared
vocabulary: GPc [CIP-0016](https://github.com/SheffieldML/GPc/blob/main/cip/cip0016.md)
(“Approximation vs sparse linear algebra”). LQ maths: GPc CIP-001B /
Seeger, Hetzel, Dai, Meissner & Lawrence (2017).

## Progress Updates

### 2026-11-22
Requirement drafted from the GPc CIP-0016 design note so GPy can own the
API + LQ implementation without forking inference classes.
