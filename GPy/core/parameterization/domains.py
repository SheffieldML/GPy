# Copyright (c) 2012-2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Re-export paramz domain constants for GPy callers (CIP-000F Stage 0)."""

from paramz.domains import _NEGATIVE, _POSITIVE, _REAL  # noqa: F401

__all__ = ["_NEGATIVE", "_POSITIVE", "_REAL"]
