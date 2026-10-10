Multi-output and coregionalized regression
==========================================

GPy offers two main entry points for vector-valued / multi-task GPs. This page
summarises when to use each, how ICM/LCM kernels fit in, and the
``predict`` / ``set_XY`` conventions that trip people up (#1099, #1016, #1149).

Which model?
------------

**``GPy.models.GPCoregionalizedRegression``** (and the sparse sibling
``SparseGPCoregionalizedRegression``) — convenience wrapper for
**correlated multi-output regression**. Pass a list of ``X`` / ``Y`` arrays
(one per output) and typically an **ICM** or **LCM** kernel from
``GPy.util.multioutput``. Each output gets its own Gaussian noise term
(mixed-noise likelihood). Start here for standard multi-task regression.

**``GPy.models.MultioutputGP``** — more general **multi-likelihood /
multi-kernel** model. Pass lists of kernels and likelihoods (plus optional
cross-covariances). Use it when outputs need different likelihoods (e.g.
observation + derivative observations) or a custom multi-output kernel layout.

Rule of thumb: correlated Gaussian multi-output regression →
``GPCoregionalizedRegression`` + ICM/LCM; heterogeneous likelihoods →
``MultioutputGP``.

ICM and LCM kernels
-------------------

Helpers live in ``GPy.util.multioutput``:

* **ICM** — one base kernel times a coregionalization matrix ``B``
  (``W W^T + diag(kappa)``).
* **LCM** — sum of ICM terms, one per base kernel in ``kernels_list``.

``input_dim`` is the dimension of the **original** inputs (without the
output-index column that GPy appends internally). ``num_outputs`` must match
the length of your ``X_list`` / ``Y_list``.

Data layout
-----------

Constructors accept **lists** of arrays. Internally GPy stacks them and adds a
final column of integer **output indices**::

    X, Y, output_index = GPy.util.multioutput.build_XY(X_list, Y_list)
    # X.shape == (N_total, input_dim + 1)

So a model trained with 2-D spatial inputs has ``model.X.shape[1] == 3``
(``x``, ``y``, ``output_index``). Keep that in mind when building test points
by hand.

Prediction for one output (#1099)
---------------------------------

This does **not** work — ``output_index`` must be a column aligned with the
rows of ``Xnew``, and ``Xnew`` must include the index column::

    # Wrong
    model.predict(X_new, Y_metadata={"output_index": [0]})

Preferred: pass a **list** of per-output test arrays (empty arrays for outputs
you skip). ``predict`` builds the index column and ``Y_metadata`` for you::

    X_star = np.linspace(0, 1, 50)[:, None]          # input_dim == 1
    # Predict output 0 only (two-output model):
    mu, var = model.predict([X_star, np.empty((0, 1))])

    # Or both outputs at the same locations:
    mu, var = model.predict([X_star, X_star])

Manual stacked form (same result)::

    Xnew = np.hstack([X_star, np.zeros((X_star.shape[0], 1))])
    meta = {"output_index": np.zeros((X_star.shape[0], 1), dtype=int)}
    mu, var = model.predict(Xnew, Y_metadata=meta)

``MultioutputGP.predict`` already accepted list inputs; coregionalized models
follow the same convention.

``set_XY`` and ``normalizer`` (#1149)
------------------------------------

``set_XY`` accepts either:

* **lists** of per-output arrays (same as the constructor), or
* **stacked** arrays already in the internal layout (with the index column).

Do not mix a list for ``X`` with a stacked array for ``Y`` (or the reverse).

``normalizer=True`` standardises the **stacked** multi-output ``Y`` as a single
column (all outputs together). Re-pass lists (or stacked arrays) through
``set_XY`` when the number of observations changes so ``output_index`` stays
aligned.

Minimal example
---------------

::

    import numpy as np
    import GPy

    X1 = np.random.rand(40, 1) * 8
    X2 = np.random.rand(30, 1) * 5
    Y1 = np.sin(X1) + 0.05 * np.random.randn(*X1.shape)
    Y2 = np.sin(X2) + 0.05 * np.random.randn(*X2.shape) + 2.0

    kern = GPy.util.multioutput.LCM(
        input_dim=1,
        num_outputs=2,
        kernels_list=[GPy.kern.RBF(1), GPy.kern.RBF(1)],
        W_rank=1,
    )
    m = GPy.models.GPCoregionalizedRegression(
        [X1, X2], [Y1, Y2], kernel=kern, normalizer=False
    )
    m.optimize()

    X_star = np.linspace(0, 8, 100)[:, None]
    mu0, var0 = m.predict([X_star, np.empty((0, 1))])

See also ``GPy.examples.regression.coregionalization_toy`` and the tutorial
notebooks on the GPy homepage.
