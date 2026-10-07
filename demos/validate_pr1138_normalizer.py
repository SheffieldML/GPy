#!/usr/bin/env python
"""
Minimal check for SheffieldML/GPy#1138.

Claim (normalizer=True):
  1. posterior_samples variance is too small vs predict (noise added on
     normalized scale after f was already inverse-transformed).
  2. log_predictive_density compares Y on the original scale with
     (_raw_predict) moments on the normalized scale.

On pre-#1138 devel both checks failed for normalizer=True.
After #1138 they should pass. Without a normalizer, the same checks
already passed (control).
"""
from __future__ import print_function

import numpy as np

import GPy


def _gaussian_logpdf(y, mu, var):
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var


def check(normalizer, size=40000, seed=0):
    np.random.seed(seed)
    X = np.random.uniform(0, 10, (30, 1))
    Y = 50 + 10 * np.sin(X) + np.random.randn(30, 1)
    X_new = np.linspace(0, 10, 4)[:, None]
    Y_new = 50 + 10 * np.sin(X_new)

    m = GPy.models.GPRegression(X, Y, normalizer=normalizer)
    m.optimize()
    mu, var = m.predict(X_new)

    samples = m.posterior_samples(X_new, size=size)
    sample_mean = samples.mean(-1)
    sample_var = samples.var(-1)

    lpd = m.log_predictive_density(X_new, Y_new)
    expected = _gaussian_logpdf(Y_new, mu, var)

    mean_ok = np.allclose(sample_mean, mu, atol=0.1)
    var_ok = np.allclose(sample_var, var, rtol=0.05)
    lpd_ok = np.allclose(lpd, expected)

    label = "normalizer={}".format(normalizer)
    print("---", label, "---")
    print("predict var[0]:     {:.4f}".format(float(var[0, 0])))
    print("sample  var[0]:     {:.4f}".format(float(sample_var[0, 0])))
    print("sample/predict[0]:  {:.4f}".format(float(sample_var[0, 0] / var[0, 0])))
    print("max |sample mean - mu|: {:.4e}".format(float(np.max(np.abs(sample_mean - mu)))))
    print("max |lpd - N(mu,var)|:  {:.4e}".format(float(np.max(np.abs(lpd - expected)))))
    print("lpd[0]:      {:.4f}".format(float(lpd[0, 0])))
    print("expected[0]: {:.4f}".format(float(expected[0, 0])))
    print(
        "checks: mean={}  var={}  lpd={}".format(
            "OK" if mean_ok else "FAIL",
            "OK" if var_ok else "FAIL",
            "OK" if lpd_ok else "FAIL",
        )
    )
    return mean_ok and var_ok and lpd_ok


if __name__ == "__main__":
    ok_off = check(normalizer=False)
    ok_on = check(normalizer=True)
    print()
    if ok_off and not ok_on:
        print("VALIDATES #1138: broken with normalizer=True, fine without.")
    elif ok_off and ok_on:
        print("Both modes pass — #1138 fix appears to be present.")
    else:
        print("Unexpected result (control failed or mixed). Inspect output above.")
