# Lynguine Backlog Index

This file provides an overview of all current backlog items organized by category and status.

## Documentation

### Ready

- [Latent explorer example notebook and serve notes](documentation/2026-10-26_latent-explorer-example-docs.md)

- [Sync GPy dataset resource URLs with pods / datasets_mirror](documentation/2026-10-25_sync-dataset-resource-urls.md)


### In Progress

*No tasks currently in_progress.*


### Proposed

- [Optional demos for the October 2026 maintenance batch](documentation/2026-10-07_october-2026-demos.md)


## Infrastructure

### Ready

- [CIP-000D — deprecate orphaned LFM1 kernel and schedule removal](infrastructure/2026-11-13_cip000D-deprecate-lfm1.md)

- [Design parameter tying framework for GPy multioutput kernels](infrastructure/2025-08-15_parameter-tying-framework.md)


### In Progress

- [CIP-000E — Cholesky Murray blocked soak and multicore confirmation](infrastructure/2026-11-16_cip000E-cholesky-murray-soak.md)

- [CIP-000C — migrate contributor packaging from Poetry to uv](infrastructure/2026-11-13_cip000C-uv-packaging-cutover.md)

- [Deprecate matplotlib visualize LVM callbacks](infrastructure/2026-10-26_deprecate-matplotlib-visualize.md)

- [CIP-0008 — rename default branch devel to main](infrastructure/2026-10-24_cip0008-rename-default-main.md)

- [Packaging residuals — matplotlib import coupling and arm64 Cython wheels](infrastructure/2026-10-11_packaging-residuals-matplotlib-arm64.md)


### Proposed

*No tasks currently proposed.*


## Features

### Ready

- [Port stick, oil, and faces LVM demos to explore](features/2026-10-26_port-lvm-demos-to-explore.md)

- [Plotly skeleton view and drop VPython](features/2026-10-26_plotly-skeleton-drop-vpython.md)


### In Progress

- [Optional chunked / threaded RBF (and linear) psi-statistics](features/2026-11-13_chunked-threaded-rbf-psi.md)

- [Create MATLAB comparison framework for LFM kernel validation](features/2025-08-15_matlab-comparison-framework.md)


### Proposed

- [Feature request inventory (champion before implementing)](features/2026-10-11_feature-request-inventory.md)


## Bugs

### Ready

*No tasks currently ready.*


### In Progress

*No tasks currently in_progress.*


### Proposed

- [MRD shared-latent MiniBatch wiring ELBO slightly below fresh views](bugs/2026-11-20_mrd-shared-latent-elbo.md)


---

## Recently Completed Tasks

- [CIP-000E Phase 2 — remove Cython build surface after hybrid soak](infrastructure/2026-11-16_cip000E-phase2-cython-removal.md)

- [CIP-000E — decide symmetrify default (blocked NumPy vs Cython)](infrastructure/2026-11-16_cip000E-symmetrify-numpy.md)

- [CIP-000E — replace flat/triang Cholesky maps with tril_indices NumPy](infrastructure/2026-11-16_cip000E-triang-maps-numpy.md)

- [CIP-000E — replace coregionalize gradient_reduce with one-hot BLAS form](infrastructure/2026-11-16_cip000E-coregionalize-reduce-numpy.md)

- [CIP-000E — replace stationary Cython grads with CIP-cited NumPy/BLAS forms](infrastructure/2026-11-16_cip000E-stationary-grads-numpy.md)

- [MRD constructor ignores user-supplied Z=](bugs/2026-11-20_mrd-ignores-constructor-z.md)

- [CIP-000D — implement kern coverage slices K1–K5](infrastructure/2026-11-13_cip000D-kern-coverage-slices.md)

- [CIP-0009 Slice B — next coverage hotspots after plotting](infrastructure/2026-11-13_cip0009-slice-b-hotspots.md)

- [CIP-0009 Slice A — plotly fill/bar/annotation and PCA / StateSpace plot smokes](infrastructure/2026-11-13_cip0009-slice-a-plotting-coverage.md)

- [Repo hygiene — asserts, dead util modules, old tests](infrastructure/2026-10-11_repo-hygiene-asserts-util-tests.md)


## Recently Abandoned Tasks

- [Add GPy.plotting.explore module and optional extra](features/2026-10-26_explore-module-optional-extra.md)
