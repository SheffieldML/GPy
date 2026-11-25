# Copyright (c) 2026 GPy authors.
# Licensed under the BSD 3-clause license (see LICENSE.txt)
"""Shared helpers for MCMC samplers."""

from __future__ import annotations

import numpy as np
from GPy.core.parameterization.transformations import Transformation, __fixed__


def unfixed_indices_without_priors(model):
    """Return sorted unfixed ``param_array`` indices that have no prior.

    Fixed parameters are ignored. Indices refer to the model's flat
    ``param_array`` / ``parameter_names_flat()`` ordering.
    """
    fixed = set()
    for constraint, ind in model.constraints.items():
        if constraint is __fixed__ or constraint == __fixed__:
            fixed.update(int(i) for i in np.atleast_1d(ind))

    priored = set()
    if getattr(model, "priors", None) is not None and model.priors.size > 0:
        for _prior, ind in model.priors.items():
            priored.update(int(i) for i in np.atleast_1d(ind))

    n_params = model.param_array.size
    return sorted(i for i in range(n_params) if i not in fixed and i not in priored)


def unfixed_transformed_indices_without_priors(model):
    """Return sorted unfixed transformed indices that have no prior.

    ``log_prior()`` only adds the transform log-Jacobian for parameters that
    carry a prior. Sampling those parameters in unconstrained space without a
    prior therefore targets a flat density in unconstrained coordinates, not
    the constrained-space posterior implied by the transform alone.
    """
    missing = unfixed_indices_without_priors(model)
    if not missing:
        return []

    missing_set = set(missing)
    transformed_missing = set()
    for constraint, ind in model.constraints.items():
        if not isinstance(constraint, Transformation):
            continue
        for jj in np.atleast_1d(ind):
            jj = int(jj)
            if jj in missing_set:
                transformed_missing.add(jj)
    return sorted(transformed_missing)


def assert_mcmc_priors(model, require_priors=True):
    """Validate that MCMC on ``model`` has a well-defined posterior target.

    When ``require_priors`` is True (the default), every unfixed parameter must
    have a prior. For transformed parameters this also ensures the change-of-
    variables Jacobian in :meth:`~GPy.core.parameterization.priorizable.Priorizable.log_prior`
    is included in the sampling target.

    :param model: GPy model whose unfixed parameters will be sampled
    :param require_priors: if False, skip the check (escape hatch for
        intentional flat-in-unconstrained sampling)
    :raises ValueError: if unfixed parameters lack priors
    """
    if not require_priors:
        return

    missing = unfixed_indices_without_priors(model)
    if not missing:
        return

    names = list(model.parameter_names_flat())
    missing_names = [str(names[i]) for i in missing]
    transformed_missing = set(unfixed_transformed_indices_without_priors(model))

    detail_parts = []
    for i, name in zip(missing, missing_names, strict=True):
        if i in transformed_missing:
            detail_parts.append(f"{name} (transformed: Jacobian omitted without a prior)")
        else:
            detail_parts.append(f"{name} (no prior: improper flat target)")

    raise ValueError(
        "MCMC requires a prior on every unfixed parameter so the sampling "
        "target is a proper posterior in unconstrained space (including "
        "transform Jacobians where constraints apply). Missing priors for: "
        + ", ".join(detail_parts)
        + ". Set priors before sampling, or pass require_priors=False if you "
        "intentionally want a flat density in unconstrained coordinates."
    )
