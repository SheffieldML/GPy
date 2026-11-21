"""Pure NumPy Gaussian likelihood primitives (CIP-000F Stage 2)."""

from __future__ import annotations

import numpy as np

log_2_pi = np.log(2.0 * np.pi)


def logpdf_link(link_f, y, variance):
    """Independent Gaussian log density of ``y`` given mean ``link_f``."""
    link_f = np.asarray(link_f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    variance = float(np.asarray(variance).reshape(-1)[0])
    return -(1.0 / (2.0 * variance)) * ((y - link_f) ** 2) - 0.5 * np.log(variance) - 0.5 * log_2_pi


def dlogpdf_dlink(link_f, y, variance):
    """Gradient of ``logpdf_link`` wrt ``link_f``."""
    link_f = np.asarray(link_f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    variance = float(np.asarray(variance).reshape(-1)[0])
    s2_i = 1.0 / variance
    return s2_i * y - s2_i * link_f
