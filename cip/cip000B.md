---
author: Neil Lawrence
created: "2026-10-26"
id: "000B"
last_updated: "2026-10-26"
status: Proposed
compressed: false
related_requirements: []
related_cips:
- "0007"
- "0009"
- "000A"
tags:
- cip
- plotting
- visualization
- gplvm
- jupyter
title: Modern latent-space explorer to replace matplotlib visualize
---

# CIP-000B: Modern latent-space explorer to replace matplotlib visualize

## Status

- [x] Proposed - Initial idea documented
- [ ] Accepted - Approved, ready to start work
- [ ] In Progress - Actively being implemented
- [ ] Implemented - Work complete, awaiting verification
- [ ] Closed - Verified and complete
- [ ] Rejected - Will not be implemented
- [ ] Deferred - Postponed

## Summary

Replace `GPy.plotting.matplot_dep.visualize` — the 2012–2015 latent-space
mouse explorer (`lvm`, `lvm_dimselect`, `image_show`, `vector_show`, stick
and skeleton viewers) — with a browser-based explorer whose pointer events
call `model.predict` in Python.

This is **not** a rewrite of GPy's static plotting (`plot`, `plot_latent`,
`plot_confidence`). CIP-0009 already treats that stack as out of scope for a
full rewrite. This CIP is only the interactive LVM viewers and the demos that
depend on them (`stick`, `stick_bgplvm`, oil, faces).

## Motivation

The old viewers have two weak points, and both fail on current macOS / Jupyter:

1. **Matplotlib event callbacks** (`button_press_event`, `motion_notify_event`)
   only work when a GUI event loop is integrated with the interpreter. On
   macOS that is the Cocoa backend (`macosx` in matplotlib, `osx` in IPython).
   That path has broken across non-framework builds, conda envs, Jupyter's
   default inline backend, `plt.show(block=False)` plus `input()`, and nested
   `canvas.draw()` (RecursionError in transforms). Local attempts to patch
   `visualize.py` around this are on `notebook-latent-doodle-space` and are
   not worth shipping.
2. **VPython (`visual`)** for 3D skeleton views is effectively dead in the
   form GPy imports.

The behaviour we still want is the same: move in latent space, see the
reconstruction; for Q>2, choose which two latent axes to plot. That is a
pointer stream plus `m.predict`, not a matplotlib GUI backend problem.

## Detailed Description

### What stays

- `GPy.plotting` static charts and the matplotlib / plotly backends.
- GPLVM / BayesianGPLVM / BCGPLVM model code.
- Demo *data* (OSU stick, oil, Olivetti, CMU mocap).
- The reconstruction contract: a latent point `X*` of shape `(1, Q)` in,
  `model.predict(X*)[0]` out.

### What is replaced

| Old (`visualize.py`) | Role | New |
|---|---|---|
| `lvm` | 2D latent scatter, click-to-track, reconstruct | Pointer stream on latent points |
| `lvm_dimselect` | Left/right click bars to pick y/x latent dims | Widgets selecting two axes; other dims held at mean |
| `image_show` / `vector_show` | Output views | Image / curve panes |
| `stick_show` / `skeleton_show` | 2D/3D mocap | 2D line segments; 3D Plotly bones |
| `data_play` | Frame playback | Periodic callback or slider |
| `vpython_show` | 3D via `visual` | Removed |

### Recommended stack

**HoloViews + Panel (Bokeh)** for the 2D explorer. Every mouse move needs a
Python `predict` call, so Altair (JS-only selections) is a poor fit.
HoloViews `PointerXY` / `Tap` replace `mpl_connect`. Panel serves the same
app in Jupyter, VS Code, or `panel serve`.

**Plotly** already exists as `gpy[plotting]`. Use `FigureWidget` / Plotly
3D for mocap bones, not a new 3D library (pythreejs / k3d / PyVista) unless
Plotly cannot draw the stick figure.

**ipywidgets** is already in `gpy[notebook]`. Use it for dim selectors if
Panel is too heavy for a given demo. **anywidget** is not required for v1.

**ipympl** (`%matplotlib widget`) can make *some* of the old 2D callbacks
work in Jupyter. It is a stopgap, not the rewrite: it still uses matplotlib
event handlers and does not fix VPython or the desktop demos.

### API sketch

Keep GPy models ignorant of the UI. A small explorer module owns layout and
streams; viewers are functions from a reconstructed `Y` row to a pane.

```python
# BayesianGPLVM: m.X.mean ; GPLVM: m.X
X = np.asarray(m.X.mean if hasattr(m.X, "mean") else m.X)
latent = hv.Points(X[:, [i, j]])
pointer = hv.streams.PointerXY(source=latent)

def reconstruct(x, y):
    z = latent_mean.copy()          # (1, Q)
    z[0, i], z[0, j] = x, y
    mu, _ = m.predict(z)
    return view(mu)                 # Image, curve, or stick
```

Public entry points (names provisional, decide at Accept):

- `GPy.plotting.explore.latent_explorer(model, view, ...)`
- `image_view`, `vector_view`, `stick_view`, `skeleton_view`
- Optional `panel serve` apps wrapping the stick and oil demos

Do **not** add HoloViews / Panel to the core GPy install. New extra, e.g.
`gpy[explore]`, or examples that `pytest.importorskip`. Core `predict` must
keep working without those packages.

### Relation to the doodle notebook

Baxter & Anjyo latent doodle space lives on `notebook-latent-doodle-space`
and is **out of scope** here. Once this explorer works, that notebook can
swap its matplotlib grid for `latent_explorer`. Do not merge the two
branches in this CIP.

### Tenets

Community-driven development: this is an architectural change to a public
demo API, so it goes through a CIP rather than more local matplotlib
patches. Default extra stays light; people who want the explorer opt in.

## Implementation Plan

1. **Spike (one demo, no public API yet)**
   - Panel app for `stick_bgplvm(optimize=False)`: latent scatter + stick
     reconstruction + two dim selectors.
   - Confirm it works in Jupyter and as `panel serve`, on macOS, without
     matplotlib GUI backends.

2. **Module**
   - Add `GPy/plotting/explore/` (or similar) with explorer + image/vector/stick
     views.
   - Extra in `pyproject.toml`; skip tests when the extra is absent.

3. **Port demos**
   - `stick`, `stick_bgplvm`, `bgplvm_oil` / oil dimselect, Olivetti faces.
   - `data_play` as a slider or periodic update, not `time.sleep`.

4. **3D skeleton**
   - Plotly line segments from existing `connect` / `skel` data.
   - Drop `visual` import path.

5. **Deprecate `visualize.py`**
   - Import warning pointing at the new explorer.
   - Leave the file until the demos above no longer import it, then remove
     or shrink to geometry helpers if any remain useful.

6. **Docs**
   - One notebook or example script; note Jupyter vs `panel serve`.
   - Compress into Sphinx only after this CIP is Closed.

## Backward Compatibility

- Static `plot_latent` / matplotlib plotting: unchanged.
- `GPy.plotting.matplot_dep.visualize.lvm` and friends: deprecated, then
  removed. Callers are GPy examples, not a wide public API.
- New optional extra; default `pip install gpy` does not pull Panel.
- No change to model save/load or inference.

## Testing Strategy

- Unit-test reconstruct helpers with a tiny GPLVM on `Agg` (no display):
  latent point in, `Y` shape out, viewer data structures without serving.
- Skip explorer tests if HoloViews/Panel are missing.
- Do not put macOS GUI or live browser-driver tests in CI.
- Manual check: Jupyter widget backend and `panel serve` on macOS for the
  stick spike before calling the CIP Implemented.

## Related Requirements

None yet (requirements extraction still outstanding).

## Implementation Status

- [x] Write CIP (Proposed)
- [ ] Accept CIP after review
- [ ] Spike: Panel stick explorer on macOS
- [ ] Public `explore` module + optional extra
- [ ] Port stick / oil / faces demos
- [ ] Plotly skeleton view; remove VPython path
- [ ] Deprecate `visualize.py`
- [ ] Close CIP

## References

- `GPy/plotting/matplot_dep/visualize.py` — current matplotlib/VPython viewers
- `GPy/examples/dimensionality_reduction.py` — `stick`, `stick_play`, `stick_bgplvm`
- Matplotlib backends: https://matplotlib.org/stable/users/explain/figure/backends.html
  (`macosx` backend; IPython GUI loop name `osx`)
- Local branch `notebook-latent-doodle-space` — doodle notebook plus failed
  matplotlib GUI patches (not in this CIP)
- CIP-0007 / #1210 — deprecation/plotting modernization
- CIP-0009 — coverage for plotting; explicitly not a full plotting rewrite
- HoloViews streams: `PointerXY`, `Tap`; Panel `servable()` / `panel serve`
