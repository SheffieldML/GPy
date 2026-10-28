---
id: 2026-10-26_latent-explorer-example-docs
title: Latent explorer example notebook and serve notes
status: Ready
priority: Low
created: "2026-10-26"
last_updated: "2026-10-26"
category: documentation
related_cips:
- "000B"
owner: Neil Lawrence
dependencies:
- 2026-10-26_explore-module-optional-extra
tags:
- backlog
- documentation
- plotting
- jupyter
---

# Task: Latent explorer example docs

> Backlog task for [CIP-000B](../../cip/cip000B.md). Informal docs only;
> Sphinx compression waits until the CIP is Closed.

## Description

Add one notebook or example script that shows:

- `pip install 'gpy[explore]'` (or the extra name chosen in the module task)
- Jupyter use
- `panel serve` for a standalone window

Keep it short. Do not write a Sphinx tutorial until CIP-000B is Closed and
compressed.

The doodle notebook on `notebook-latent-doodle-space` is out of scope.

## Acceptance Criteria

- [ ] Example lives in `notebooks/` or `GPy/examples/` and imports the explore API
- [ ] README or example docstring states Jupyter vs `panel serve`
- [ ] No Sphinx page yet (compression is a later CIP-close step)

## Implementation Notes

- Prefer reusing the stick spike as the documented example.
- Optional extra must be mentioned so a default GPy install is not assumed.

## Related

- CIP: 000B
- Depends: 2026-10-26_explore-module-optional-extra

## Progress Updates

### 2026-10-26

Task created when CIP-000B was Accepted.
