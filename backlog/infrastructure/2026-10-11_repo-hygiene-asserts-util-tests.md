---
id: 2026-10-11_repo-hygiene-asserts-util-tests
title: Repo hygiene — asserts, dead util modules, old tests
status: Completed
priority: High
created: '2026-10-11'
last_updated: '2026-11-08'
category: infrastructure
related_cips:
- '0007'
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- hygiene
- packaging
---

# Task: Repo hygiene

## Description

Tracking [#1213](https://github.com/SheffieldML/GPy/issues/1213): `#922` asserts→ValueError,
`#930` dead util modules, `#1044` old tests, `#609` obsolete mpl unittest.

`#1044` became user-visible as [#1251](https://github.com/SheffieldML/GPy/issues/1251): the
1.15.0 wheel still contained Python 2 source (`old_tests`, `kern/src/todo`,
`mpi_test__.py`). Runtime support is already Python >=3.11; those files were
never imported. Do not 2to3 them.

## Acceptance Criteria

- [x] `#1044` / #1251: stop shipping (and delete) Python 2 trees; `compileall` tripwire
- [x] `#922` asserts→ValueError (GP construction + `block_matrices`; remaining internal asserts left as invariants)
- [x] `#930` dead util / symbolic AD stack removed
- [x] `#609` obsolete mpl unittest: verified obsolete (modern `test_plotting` uses pytest + `compare_images` with skipif)
- [x] Close #1213 when complete

## Related

- Issue: #1213, #1251, #1044
- CIP: 0007

## Progress Updates

### 2026-11-01
Raised to High after #1251. Deleted `GPy/old_tests`, `GPy/kern/src/todo`, and
`GPy/testing/mpi_test__.py`; Poetry `exclude` plus `python -m compileall -q GPy`
on the lint job so a Python 2 file cannot return in a wheel.

### 2026-11-08
- `#922`: `GP.__init__` / `set_XY` / `_format_input_dict` and `util/block_matrices`
  raise `ValueError` for bad caller input (trigger from #917 review).
- `#930`: deleted unused `util/ln_diff_erfs.py`, `util/functions.py`,
  `core/symbolic.py`, `kern/src/symbolic.py` (sympy AD stack; not exported;
  sympy not a declared dependency). `util/symbolic.py` was already absent.
- `#609`: no remaining matplotlib-2.0/2.2 unittest failure path; image baselines
  are exercised by modern pytest helpers in `test_plotting.py`.
