"""Prototype: INLA-style hyperparameter integration around a GPy Laplace model.

Research material for CIP-0015. Not part of the GPy package.

For a fitted GP model ``m`` using the ``Laplace`` inference method we

1. find the MAP of the hyperparameters in the *unconstrained* (optimizer) space,
2. estimate the Hessian of the negative log posterior there,
3. build a design of points in standardised coordinates ``z`` (``u = u* + L z``,
   ``L L^T = H^{-1}``),
4. evaluate the unnormalised log posterior ``log p(y | u) + log p(u)`` at each point
   (``-m.objective_function()``), and
5. mix the per-point Laplace predictions with the normalised weights.

Density convention: GPy's ``log_prior`` adds the transform Jacobian only for
parameters that carry a prior. Parameters without a prior therefore get a flat
density in the unconstrained space. Giving every free parameter a prior makes the
target density well defined (see commit dae930d4).
"""

import itertools

import numpy as np


def _logsumexp_weights(logp):
    logp = np.asarray(logp, dtype=float)
    w = np.exp(logp - logp.max())
    return w / w.sum()


class INLAHyperparameters:
    """Hyperparameter-marginalised predictions for a GPy model with Laplace inference."""

    def __init__(self, model, design="auto", grid_step=1.0, grid_cutoff=3.0, ccd_radius=None):
        """
        :param model: a GPy ``GP`` model (typically Laplace inference).
        :param design: ``"grid"``, ``"ccd"`` (central composite) or ``"auto"``
            (grid for <= 3 hyperparameters, otherwise ccd).
        :param grid_step: spacing in standardised z units for the grid design.
        :param grid_cutoff: drop grid points whose log density is more than this
            below the maximum (in nats) -- R-INLA uses 'diff.logdens'.
        :param ccd_radius: radius of the star points for ccd (default sqrt(d) * 1.1).
        """
        self.model = model
        self.design = design
        self.grid_step = grid_step
        self.grid_cutoff = grid_cutoff
        self.ccd_radius = ccd_radius
        self.fitted = False

    # ---- hyperparameter posterior -------------------------------------------------

    def _log_post(self, u):
        """Unnormalised log p(u | y) in the unconstrained space."""
        try:
            return -self.model._objective(u)
        except Exception:  # noqa: BLE001 -- treat any numerical failure as zero mass
            return -np.inf

    def _hessian(self, u, h=1e-3):
        """Hessian of the negative log posterior by central differences of the gradient."""
        d = u.size
        H = np.zeros((d, d))
        for i in range(d):
            up, um = u.copy(), u.copy()
            up[i] += h
            um[i] -= h
            gp = self.model._objective_grads(up)[1].copy()
            gm = self.model._objective_grads(um)[1].copy()
            H[:, i] = (gp - gm) / (2 * h)
        self.model.optimizer_array = u
        return 0.5 * (H + H.T)

    def fit(self, optimize=True, **optimize_kwargs):
        m = self.model
        if optimize:
            m.optimize(**optimize_kwargs)
        self.u_star = np.array(m.optimizer_array, dtype=float)
        d = self.u_star.size
        H = self._hessian(self.u_star)
        evals, evecs = np.linalg.eigh(H)
        if evals.min() <= 1e-8:
            # Not a proper mode (or a flat direction): regularise so the design stays finite.
            evals = np.maximum(evals, 1e-2)
            self.hessian_regularised = True
        else:
            self.hessian_regularised = False
        self.hessian = H
        # u = u* + L z with L L^T = H^{-1}
        self.L = evecs / np.sqrt(evals)
        self._build_design(d)
        self.fitted = True
        return self

    def _build_design(self, d):
        design = self.design
        if design == "auto":
            design = "grid" if d <= 3 else "ccd"
        if design == "grid":
            Z = self._grid_design(d)
        elif design == "ccd":
            Z = self._ccd_design(d)
        else:
            raise ValueError("unknown design %r" % design)
        U = self.u_star + Z @ self.L.T
        logp = np.array([self._log_post(u) for u in U])
        keep = np.isfinite(logp)
        if design == "grid":
            keep &= logp > logp[keep].max() - self.grid_cutoff
        self.design_used = design
        self.Z, self.U, self.logp = Z[keep], U[keep], logp[keep]
        self.weights = _logsumexp_weights(self.logp)
        self.model.optimizer_array = self.u_star

    def _grid_design(self, d):
        """Greedy outward exploration of a z-grid, stopping where density drops off."""
        step = self.grid_step
        seen = {}
        frontier = [tuple([0] * d)]
        best = -np.inf
        while frontier:
            idx = frontier.pop()
            if idx in seen:
                continue
            lp = self._log_post(self.u_star + (np.array(idx) * step) @ self.L.T)
            seen[idx] = lp
            if np.isfinite(lp):
                best = max(best, lp)
            if np.isfinite(lp) and lp > best - self.grid_cutoff:
                for i in range(d):
                    for s in (-1, 1):
                        nb = list(idx)
                        nb[i] += s
                        if tuple(nb) not in seen:
                            frontier.append(tuple(nb))
        return np.array([np.array(k) * step for k in seen])

    def _ccd_design(self, d):
        """Central composite design in z: centre, 2d star points, and 2^d (or fractional) corners."""
        f0 = self.ccd_radius if self.ccd_radius is not None else 1.1 * np.sqrt(d)
        pts = [np.zeros(d)]
        for i in range(d):
            for s in (-1, 1):
                e = np.zeros(d)
                e[i] = s * f0
                pts.append(e)
        if d <= 5:
            corners = itertools.product((-1, 1), repeat=d)
            pts.extend(np.array(c) * 1.0 for c in corners)
        else:
            rng = np.random.default_rng(0)
            pts.extend(rng.choice((-1, 1), size=(2**5, d)) * 1.0)
        return np.array(pts)

    # ---- prediction -----------------------------------------------------------------

    def predict_noiseless(self, Xnew):
        """Mixture mean and variance of the latent function f(Xnew)."""
        return self._mix(lambda m: m.predict_noiseless(Xnew))

    def predict(self, Xnew, **kwargs):
        """Mixture of the likelihood-space predictive mean and variance (e.g. P(y=1))."""
        return self._mix(lambda m: m.predict(Xnew, **kwargs))

    def _mix(self, fn):
        assert self.fitted, "call fit() first"
        means, variances = [], []
        for u in self.U:
            self.model.optimizer_array = u
            mu, var = fn(self.model)
            means.append(np.asarray(mu))
            variances.append(np.asarray(var))
        self.model.optimizer_array = self.u_star
        means, variances = np.array(means), np.array(variances)
        w = self.weights.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (w * means).sum(0)
        # law of total variance
        var = (w * (variances + means**2)).sum(0) - mean**2
        return mean, var

    def hyperparameter_marginals(self):
        """Weighted mean and standard deviation of each (constrained) hyperparameter."""
        assert self.fitted
        vals = []
        for u in self.U:
            self.model.optimizer_array = u
            vals.append(np.array(self.model.param_array, dtype=float))
        self.model.optimizer_array = self.u_star
        vals = np.array(vals)
        mean = (self.weights[:, None] * vals).sum(0)
        sd = np.sqrt((self.weights[:, None] * (vals - mean) ** 2).sum(0))
        return dict(zip(self.model.parameter_names_flat(), zip(mean, sd), strict=False))


if __name__ == "__main__":
    import GPy

    rng = np.random.default_rng(1)
    n = 25
    X = np.sort(rng.uniform(-3, 3, (n, 1)), axis=0)
    p = 1 / (1 + np.exp(-3 * np.sin(1.5 * X)))
    Y = (rng.uniform(size=(n, 1)) < p).astype(float)

    kern = GPy.kern.RBF(1)
    kern.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
    kern.variance.set_prior(GPy.priors.LogGaussian(1.0, 1.0), warning=False)
    m = GPy.core.GP(
        X,
        Y,
        kernel=kern,
        likelihood=GPy.likelihoods.Bernoulli(),
        inference_method=GPy.inference.latent_function_inference.Laplace(),
    )

    inla = INLAHyperparameters(m).fit()
    print("design:", inla.design_used, "points:", len(inla.weights), "regularised:", inla.hessian_regularised)
    print("hyperparameter posterior (mean, sd):", inla.hyperparameter_marginals())

    Xs = np.linspace(-3.5, 3.5, 8)[:, None]
    mu_map, var_map = m.predict_noiseless(Xs)
    mu_i, var_i = inla.predict_noiseless(Xs)
    print("latent mean MAP  :", np.round(mu_map.ravel(), 3))
    print("latent mean INLA :", np.round(mu_i.ravel(), 3))
    print("latent var  MAP  :", np.round(var_map.ravel(), 3))
    print("latent var  INLA :", np.round(var_i.ravel(), 3))
    print("P(y=1) MAP :", np.round(m.predict(Xs)[0].ravel(), 3))
    print("P(y=1) INLA:", np.round(inla.predict(Xs)[0].ravel(), 3))
