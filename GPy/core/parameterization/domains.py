# Copyright (c) 2012-2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
(Hyper-)Parameter domains for priors and transformations (CIP-000F Stage 1).

These match the string values historically defined in ``paramz.domains`` so
identity comparisons against paramz transformation domains remain valid under
CPython string interning.
"""

_REAL = "real"
_POSITIVE = "positive"
_NEGATIVE = "negative"
_BOUNDED = "bounded"

__all__ = ["_REAL", "_POSITIVE", "_NEGATIVE", "_BOUNDED"]
