***************************************
Integrating over GP hyperparameters
***************************************

A GP model fitted by ``model.optimize()`` uses a single value of the hyperparameters
(kernel lengthscales and variances, noise): the maximum likelihood or MAP estimate. Predictive
uncertainty then ignores how uncertain those hyperparameters are. For non-Gaussian likelihoods
fitted with the Laplace approximation this can understate uncertainty noticeably, particularly
away from the data where the lengthscale matters most.

:py:class:`GPy.inference.hyperparameters.INLAHyperparameters` integrates over the
hyperparameters in the style of INLA (Rue, Martino and Chopin, 2009). For a fitted model it

1. finds the mode of the hyperparameter posterior :math:`p(\theta \mid y)` in GPy's unconstrained
   parameter space,
2. builds a small weighted set of hyperparameter values around the mode (the *design*),
3. makes the usual prediction at each design point, and
4. combines the predictions with the weights, using the law of total variance.

It is a wrapper: it does not change the model, the inference method or ``model.predict``.
It works with any GPy model that provides ``predict``, and is most useful with the
:py:class:`~GPy.inference.latent_function_inference.laplace.Laplace` inference method.

Example
=======

Classification with a Laplace approximation, integrating over the lengthscale and variance::

 import numpy as np
 import GPy
 from GPy.inference.hyperparameters import INLAHyperparameters

 rng = np.random.default_rng(1)
 X = np.sort(rng.uniform(-3, 3, (25, 1)), axis=0)
 Y = (rng.uniform(size=(25, 1)) < 1 / (1 + np.exp(-3 * np.sin(1.5 * X)))).astype(float)

 kern = GPy.kern.RBF(1)
 kern.lengthscale.set_prior(GPy.priors.LogGaussian(0.0, 1.0), warning=False)
 kern.variance.set_prior(GPy.priors.LogGaussian(1.0, 1.0), warning=False)
 m = GPy.core.GP(X, Y, kernel=kern, likelihood=GPy.likelihoods.Bernoulli(),
                 inference_method=GPy.inference.latent_function_inference.Laplace())

 inla = INLAHyperparameters(m).fit()          # optimises m, then builds the design
 Xnew = np.linspace(-3.5, 3.5, 100)[:, None]
 mean, var = inla.predict_noiseless(Xnew)     # latent function mixture moments
 prob, _ = inla.predict(Xnew)                 # mixture of P(y = 1)
 print(inla.hyperparameter_summary())         # weighted mean and sd of each hyperparameter

``fit`` optimises the model first; pass ``optimize=False`` if it is already at its MAP.
After the predictions the model is left at the hyperparameter mode.

Requirements
============

* **Every free hyperparameter needs a prior.** GPy's ``log_prior`` adds the transform Jacobian
  only for parameters that have a prior. Without priors the density is flat in the unconstrained
  space, which is improper and depends on the parameterisation, so ``fit`` raises a
  ``ValueError`` naming the parameters. Fix a parameter instead of giving it a prior if you do
  not want to integrate over it.
* **The mode must be a proper maximum.** If the Hessian of the negative log posterior at the mode
  is not positive definite (a hyperparameter is not identified, or the optimiser stopped early),
  ``fit`` raises ``numpy.linalg.LinAlgError``. A warning is issued if the gradient at the mode is
  not close to zero.

Choosing a design
=================

The design is set with ``design=``. All designs work in standardised coordinates
:math:`z`, where :math:`u = u^* + L z` and :math:`L L^\top` is the inverse Hessian at the mode.

``"grid"``
  Explores a regular grid outward from the mode until the log density falls ``grid_cutoff``
  nats (default 9) below the maximum, with spacing ``grid_step`` (default 1.5). The most accurate
  choice, but the number of points grows exponentially with the number of hyperparameters
  (about 150 at two hyperparameters, 1300 at four and 6300 at five in our tests).

``"importance"``
  Draws ``n_samples`` (default 500) antithetic points from a heavy-tailed multivariate-t
  proposal and weights them by the ratio of posterior to proposal density. The cost is fixed and
  does not grow with the number of hyperparameters. With ``skew=True`` (the default) each
  axis is scaled separately on its two sides, which helps because real posteriors are often
  steep on one side and heavy-tailed on the other.

``"ccd"``
  A central composite design: the centre, two points per axis and the corners. Very cheap,
  but coarse. Predictive means improve on the MAP only modestly and the spread of the
  hyperparameters is substantially underestimated. Use it only as a quick hedge.

``"auto"`` (the default) uses the grid for up to three hyperparameters and ``"importance"``
above that.

After fitting, ``inla.weights``, ``inla.U`` (the design points in the unconstrained space) and
``inla.effective_sample_size`` are available. For the importance design the effective sample
size is the reliability diagnostic: a warning is raised if it falls below 20. On strongly skewed,
concentrated posteriors it can be a small fraction of ``n_samples``; increase ``n_samples`` when
it is small.

How accurate is it?
===================

We compared the designs with dense quadrature in one hyperparameter, with HMC over the
hyperparameters on a two-hyperparameter classification problem, and with an importance-sampling
reference (cross-checked with HMC) at four and five hyperparameters. Because HMC over
:py:class:`GPy.inference.mcmc.HMC` targets the same Laplace-based density, these comparisons
measure the design approximation, not the Laplace approximation itself.

* With two hyperparameters (25 Bernoulli observations) the grid design agreed with HMC to within
  0.04 on the latent mean and variance and 0.01 on :math:`P(y=1)`. The MAP plug-in was off by
  0.25 and 1.30 on the latent mean and variance. The design needed 155 Laplace fits (3 s)
  against 55 s for HMC.
* With four and five hyperparameters, 1000 skew-aware importance points reached a maximum latent
  mean error of 0.07 and 0.06 reference predictive standard deviations, close to the grid
  (0.09 and 0.05) with up to six times fewer fits. The MAP plug-in was at 0.74 and 0.49, and the
  central composite design at 0.45 and 0.32.
* Estimates of the spread of weakly identified lengthscales are noisier than the predictive
  means, because their posteriors have long tails on the constrained scale.

Limitations
===========

* The result is hyperparameter-marginalised *Laplace* with dense covariances. It does not have the
  sparse-precision scalability that makes INLA fast on large latent Gaussian Markov random fields.
  Each design point costs one full Laplace fit.
* Only the predictive mean and variance are mixed. The mixture density and quantiles are not
  available, and INLA's simplified-Laplace and skewness corrections to the latent marginals are
  not implemented.
* The importance proposal is a scaled multivariate-t. It does not adapt to the weighted draws,
  so on strongly skewed posteriors the effective sample size can remain low.
* The wrapper evaluates the model at each design point by setting its parameters, so the model
  must not be used from other threads while ``fit`` or a ``predict`` method is running. It relies
  on the model's private ``_objective`` methods.

References
==========

* H. Rue, S. Martino and N. Chopin, *Approximate Bayesian inference for latent Gaussian models by
  using integrated nested Laplace approximations*, Journal of the Royal Statistical Society B,
  71(2), 319-392, 2009.
* Design record, measurements and notebooks: ``cip/cip0015.md`` and ``cip/cip0015/`` in the
  repository.
