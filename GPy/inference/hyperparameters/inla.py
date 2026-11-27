"""INLA-style integration over GP hyperparameters (CIP-0015).

Given a GPy model whose likelihood and inference method give a (Laplace) posterior
for fixed hyperparameters ``theta``, :class:`INLAHyperparameters` approximates

    p(f* | y) ~= sum_k w_k p(f* | theta_k, y),      w_k ~ p(theta_k | y),

by exploring ``p(theta | y)`` around its mode and mixing the per-point predictions.

All exploration happens in the unconstrained optimiser space ``u`` (GPy's
``optimizer_array``), where positivity constraints are smooth transforms and a
Gaussian approximation to ``p(u | y)`` is reasonable. The target density in ``u`` is
``log p(y | u) + log p(u)``, where GPy's ``log_prior`` includes the transform
Jacobian for parameters that carry a prior. Every free hyperparameter must
therefore have a prior; otherwise the density is improper and depends on the
parameterisation, and we raise.
"""

import itertools
import warnings

import numpy as np

_NUMERICAL_ERRORS = (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError, ValueError)


def _normalised_weights(log_p):
    log_p = np.asarray(log_p, dtype=float)
    w = np.exp(log_p - log_p.max())
    return w / w.sum()


class INLAHyperparameters:
    """Hyperparameter-marginalised predictions for a GPy model.

    The wrapper leaves the model untouched except while evaluating design points; the
    model is restored to the hyperparameter mode afterwards. It is not safe to use the
    model from other threads during :meth:`fit` or the ``predict*`` methods.

    :param model: a GPy ``GP`` model, typically with ``Laplace`` inference.
    :param design: ``"grid"``, ``"ccd"`` (central composite design), ``"importance"`` or
        ``"auto"`` (grid for at most 3 hyperparameters, otherwise importance). The grid is
        accurate but its size grows exponentially with the number of hyperparameters; ccd is
        cheap but coarse; importance has a fixed budget of ``n_samples`` Laplace fits.
    :param grid_step: grid spacing in standardised ``z`` units.
    :param grid_cutoff: grid points more than this many nats below the maximum
        log density are dropped. Weakly identified hyperparameters have heavy tails in
        the constrained scale, so a tight cutoff (we tried 3) understates their spread.
    :param ccd_radius: radius of the ccd star points (default ``1.1 * sqrt(d)``).
    :param n_samples: number of importance draws (antithetic pairs; rounded up to even).
    :param proposal_dof: degrees of freedom of the multivariate-t proposal in ``z`` space.
        Heavy tails guard against the weakly identified hyperparameters' long tails.
    :param proposal_scale: scale of the t proposal in ``z`` units.
    :param skew: scale each standardised axis separately on its positive and negative
        sides for the importance proposal, matched to the log-density drop at
        ``skew_probe``. Real posteriors are often steep on one side and heavy-tailed on
        the other; a symmetric proposal wastes draws on the steep side.
    :param skew_probe: probe distance in ``z`` units for the one-sided scales.
    :param seed: seed for the importance draws, so fits are reproducible.
    :param hessian_step: finite-difference step in ``u`` for the Hessian.

    After :meth:`fit`, ``effective_sample_size`` is ``1 / sum(weights**2)``. It is the
    reliability diagnostic for the importance design (a warning is raised below 20). For
    the grid and ccd designs it only describes how concentrated the weights are. The
    importance proposal is symmetric in ``z`` while real posteriors can be strongly skewed
    (steep on one side, heavy-tailed on the other), so check the effective sample size and
    raise ``n_samples`` when it is small.
    """

    def __init__(
        self,
        model,
        design="auto",
        grid_step=1.5,
        grid_cutoff=9.0,
        ccd_radius=None,
        n_samples=500,
        proposal_dof=5,
        proposal_scale=1.2,
        skew=True,
        skew_probe=2.0,
        seed=0,
        hessian_step=1e-3,
    ):
        if design not in ("auto", "grid", "ccd", "importance"):
            raise ValueError(f"design must be 'auto', 'grid', 'ccd' or 'importance', got {design!r}")
        self.model = model
        self.design = design
        self.grid_step = grid_step
        self.grid_cutoff = grid_cutoff
        self.ccd_radius = ccd_radius
        self.n_samples = n_samples
        self.proposal_dof = proposal_dof
        self.proposal_scale = proposal_scale
        self.skew = skew
        self.skew_probe = skew_probe
        self.seed = seed
        self.hessian_step = hessian_step
        self.fitted = False

    # ---- checks -------------------------------------------------------------------

    def _check_priors(self):
        """Raise if any free hyperparameter has no prior."""
        m = self.model
        fixes = m._fixes_
        free = np.arange(m.param_array.size) if fixes is None else np.flatnonzero(fixes)
        if free.size != m.optimizer_array.size:
            return  # tied parameters: the index bookkeeping below would be unreliable
        priored = [np.asarray(i).ravel() for _, i in m.priors.items()]
        priored = np.concatenate(priored) if priored else np.array([], dtype=int)
        missing = np.setdiff1d(free, priored)
        if missing.size:
            names = np.asarray(m.parameter_names_flat())[missing]
            raise ValueError(
                "INLAHyperparameters needs a prior on every free hyperparameter so that the target "
                f"density (including transform Jacobians) is proper. Missing priors for: {', '.join(map(str, names))}. "
                "Set them with e.g. `param.set_prior(GPy.priors.LogGaussian(0., 1.))` or fix the "
                "parameter."
            )

    # ---- hyperparameter posterior ---------------------------------------------------

    def _log_post(self, u):
        """Unnormalised log p(u | y); ``-inf`` where the model cannot be evaluated."""
        try:
            value = -self.model._objective(u)
        except _NUMERICAL_ERRORS:
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    def _hessian(self, u):
        """Hessian of -log p(u | y) by central differences of the model gradient."""
        h = self.hessian_step
        d = u.size
        H = np.empty((d, d))
        for i in range(d):
            up, um = u.copy(), u.copy()
            up[i] += h
            um[i] -= h
            gp = np.array(self.model._objective_grads(up)[1], dtype=float)
            gm = np.array(self.model._objective_grads(um)[1], dtype=float)
            H[:, i] = (gp - gm) / (2 * h)
        self.model.optimizer_array = u
        return 0.5 * (H + H.T)

    def fit(self, optimize=True, **optimize_kwargs):
        """Find the hyperparameter mode and build the weighted design.

        :param optimize: run ``model.optimize`` first. Pass ``False`` if the model is
            already at its MAP.
        """
        self._check_priors()
        m = self.model
        if optimize:
            m.optimize(**optimize_kwargs)
        self.u_star = np.array(m.optimizer_array, dtype=float)
        grad = np.asarray(m._objective_grads(self.u_star)[1], dtype=float)
        m.optimizer_array = self.u_star
        if np.abs(grad).max() > 1e-2:
            warnings.warn(
                f"Hyperparameters are not at a stationary point (max |gradient| = {np.abs(grad).max():.3g}); "
                "the Gaussian approximation to p(theta | y) may be poor.",
                stacklevel=2,
            )
        H = self._hessian(self.u_star)
        evals, evecs = np.linalg.eigh(H)
        if evals.min() <= 0:
            raise np.linalg.LinAlgError(
                "Hessian of the negative log posterior at the mode is not positive definite "
                f"(smallest eigenvalue {evals.min():.3g}): the hyperparameters are not identified or the "
                "optimiser did not reach a maximum."
            )
        self.hessian = H
        # u = u* + L z, with L L^T = H^{-1}
        self.L = evecs / np.sqrt(evals)
        self._build_design(self.u_star.size)
        self.fitted = True
        return self

    # ---- designs --------------------------------------------------------------------

    def _build_design(self, d):
        design = self.design if self.design != "auto" else ("grid" if d <= 3 else "importance")
        log_q = None
        if design == "grid":
            Z = self._grid_design(d)
        elif design == "ccd":
            Z = self._ccd_design(d)
        else:
            Z, log_q = self._importance_design(d)
        U = self.u_star + Z @ self.L.T
        log_p = np.array([self._log_post(u) for u in U])
        keep = np.isfinite(log_p)
        if design == "grid":
            keep &= log_p > log_p[keep].max() - self.grid_cutoff
        self.design_used = design
        self.Z, self.U, self.log_post = Z[keep], U[keep], log_p[keep]
        log_w = self.log_post if log_q is None else self.log_post - log_q[keep]
        self.weights = _normalised_weights(log_w)
        self.effective_sample_size = float(1.0 / np.sum(self.weights**2))
        self.model.optimizer_array = self.u_star
        if design == "importance" and self.effective_sample_size < 20:
            warnings.warn(
                f"Importance design has an effective sample size of only {self.effective_sample_size:.1f} "
                f"from {len(self.weights)} points; increase n_samples or proposal_scale.",
                stacklevel=3,
            )

    def _skew_scales(self, d):
        """One-sided scales (plus, minus) per standardised axis from the log-density drop at +-skew_probe.

        A Gaussian with scale ``sigma`` loses ``t**2 / (2 sigma**2)`` nats at distance ``t``, so we
        set ``sigma = t / sqrt(2 * drop)``. Scales are clipped to [0.25, 4] and fall back to 1 where
        the probe cannot be evaluated or the density does not drop.
        """
        t = self.skew_probe
        centre = self._log_post(self.u_star)
        scales = np.ones((2, d))
        for side, sign in enumerate((1.0, -1.0)):
            for j in range(d):
                z = np.zeros(d)
                z[j] = sign * t
                drop = centre - self._log_post(self.u_star + z @ self.L.T)
                if np.isfinite(drop) and drop > 1e-8:
                    scales[side, j] = np.clip(t / np.sqrt(2.0 * drop), 0.25, 4.0)
                elif not np.isfinite(drop):
                    scales[side, j] = 0.25  # density vanishes there: stay close to the mode
        self.model.optimizer_array = self.u_star
        return scales[0], scales[1]

    def _importance_design(self, d):
        """Antithetic multivariate-t draws, optionally skewed per axis, with log proposal density.

        Draws are ``z = z0 * s`` with ``z0`` multivariate-t and ``s`` the plus or minus scale of
        each coordinate's sign, so ``log q(z) = log t(z0) - sum(log s)``. Constants are dropped.
        """
        rng = np.random.default_rng(self.seed)
        half = (self.n_samples + 1) // 2
        dof = self.proposal_dof
        g = rng.standard_normal((half, d))
        chi = rng.chisquare(dof, size=(half, 1))
        base = self.proposal_scale * g / np.sqrt(chi / dof)
        Z0 = np.vstack([base, -base])
        log_q = -0.5 * (dof + d) * np.log1p(np.sum((Z0 / self.proposal_scale) ** 2, axis=1) / dof)
        if not self.skew:
            return Z0, log_q
        plus, minus = self._skew_scales(d)
        self.skew_scales = (plus, minus)
        scale = np.where(Z0 >= 0, plus, minus)
        return Z0 * scale, log_q - np.sum(np.log(scale), axis=1)

    def _grid_design(self, d):
        """Explore a z-grid outward from the mode while the density stays within the cutoff."""
        step = self.grid_step
        seen = {}
        frontier = [(0,) * d]
        best = -np.inf
        while frontier:
            idx = frontier.pop()
            if idx in seen:
                continue
            lp = self._log_post(self.u_star + (np.array(idx) * step) @ self.L.T)
            seen[idx] = lp
            if not np.isfinite(lp):
                continue
            best = max(best, lp)
            if lp > best - self.grid_cutoff:
                for i in range(d):
                    for s in (-1, 1):
                        nb = list(idx)
                        nb[i] += s
                        if tuple(nb) not in seen:
                            frontier.append(tuple(nb))
        return np.array([np.array(k) * step for k in seen])

    def _ccd_design(self, d):
        """Centre, 2d star points, and 2^d corners (a fixed random subset of 32 for d > 5)."""
        radius = self.ccd_radius if self.ccd_radius is not None else 1.1 * np.sqrt(d)
        pts = [np.zeros(d)]
        for i in range(d):
            for s in (-1, 1):
                e = np.zeros(d)
                e[i] = s * radius
                pts.append(e)
        if d <= 5:
            pts.extend(np.array(c, dtype=float) for c in itertools.product((-1, 1), repeat=d))
        else:
            rng = np.random.default_rng(0)
            pts.extend(rng.choice((-1.0, 1.0), size=(32, d)))
        return np.array(pts)

    # ---- prediction -----------------------------------------------------------------

    def predict_noiseless(self, Xnew, **kwargs):
        """Mixture mean and variance of the latent function at ``Xnew``."""
        return self._mix(lambda m: m.predict_noiseless(Xnew, **kwargs))

    def predict(self, Xnew, **kwargs):
        """Mixture mean and variance in observation space (e.g. ``P(y=1)`` for Bernoulli)."""
        return self._mix(lambda m: m.predict(Xnew, **kwargs))

    def _mix(self, predict):
        if not self.fitted:
            raise RuntimeError("call fit() before predicting")
        means, variances = [], []
        try:
            for u in self.U:
                self.model.optimizer_array = u
                mu, var = predict(self.model)
                means.append(np.asarray(mu))
                variances.append(np.asarray(var))
        finally:
            self.model.optimizer_array = self.u_star
        means, variances = np.array(means), np.array(variances)
        w = self.weights.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (w * means).sum(0)
        var = (w * (variances + means**2)).sum(0) - mean**2  # law of total variance
        return mean, var

    def hyperparameter_summary(self):
        """Weighted mean and standard deviation of each hyperparameter (constrained scale)."""
        if not self.fitted:
            raise RuntimeError("call fit() first")
        values = []
        try:
            for u in self.U:
                self.model.optimizer_array = u
                values.append(np.array(self.model.unfixed_param_array, dtype=float))
        finally:
            self.model.optimizer_array = self.u_star
        values = np.array(values)
        mean = (self.weights[:, None] * values).sum(0)
        sd = np.sqrt((self.weights[:, None] * (values - mean) ** 2).sum(0))
        return {name: (mu, s) for name, mu, s in zip(self.model.parameter_names_flat(), mean, sd, strict=True)}
