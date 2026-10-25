---
id: 2026-10-25_sync-dataset-resource-urls
title: Sync GPy dataset resource URLs with pods / datasets_mirror
status: Ready
priority: Low
created: "2026-10-25"
last_updated: "2026-10-25"
category: documentation
related_cips:
- "0007"
owner: Neil Lawrence
dependencies: []
tags:
- backlog
- datasets
- examples
- urls
---

# Task: Sync dataset resource URLs with pods

## Description

Close [#1214](https://github.com/SheffieldML/GPy/issues/1214) by updating
`GPy/util/data_resources.json` to the URLs pods 0.1.19 already uses.

GPy **examples** load data through `pods.datasets`, not `GPy.util.datasets`.
pods already has working hosts for both child tickets (`#671` Olympic /
Rogers–Girolami, `#841` ACCAD `run1TXT.ZIP`). GPy's JSON copy is still on
Dropbox, `accad.osu.edu`, and `staffwww.dcs.shef.ac.uk/.../dataset_mirror/`
(403; about 18 other GPy entries still use that host).

Live checks (2026-10-24), recorded on #1214:

- Dropbox Olympic tarball returns HTML, not a gzip.
- Glasgow `https://www.dcs.gla.ac.uk/~srogers/firstcourseml/firstcoursemldata.tar.gz`
  is a live gzip (~10.7 MB).
- `accad.osu.edu/research/mocap/data/run1TXT.ZIP` is 404.
- `https://github.com/lawrennd/datasets_mirror` serves `osu_accad/run1TXT.ZIP`
  and `stick/connections.txt` (both 200).

Do this on `main` after CIP-0008 rename work, not on the release-pipeline
branch.

## Acceptance Criteria

- [ ] `rogers_girolami_data` URL is Glasgow (not Dropbox)
- [ ] `osu_accad` / `osu_run1` URLs are `datasets_mirror` (not ACCAD / staffwww)
- [ ] Remaining `staffwww.dcs.shef.ac.uk/.../dataset_mirror/` entries in
      `GPy/util/data_resources.json` point at
      `https://github.com/lawrennd/datasets_mirror/raw/main/...` to match pods
- [ ] Spot-check the two #1214 downloads (HEAD/GET, real gzip/zip, not HTML)
- [ ] Close #1214

## Implementation Notes

Copy URL fields from
[pods `data_resources.json`](https://github.com/lawrennd/ods/blob/main/pods/data_resources.json)
for the overlapping keys. Keep GPy's file-name lists; only the hosts change.

Optional in the same PR: bump the dev extra `pods>=0.1.14` to `>=0.1.19` so
example runs pick up the pods-side fix. Soft-fail-when-missing is not needed
if the links work.

Tests that mock `download_data` will not notice. `GPy.util.datasets` is the
consumer of this JSON (`scripts/repro_param_array_save_load.py`, one example
fallback, benchmarks).

## Related

- Issue: [#1214](https://github.com/SheffieldML/GPy/issues/1214)
- Child issues: #671, #841 (already closed into the tracker)
- CIP: 0007
- Supersedes: `2026-10-11_example-datasets-links`
- pods: https://github.com/lawrennd/ods
- Mirror: https://github.com/lawrennd/datasets_mirror

## Progress Updates

### 2026-10-25

Task created Ready from the #1214 URL survey. GitHub default branch is already
`main`; implement there, not on `cip0008-rename-main`.
