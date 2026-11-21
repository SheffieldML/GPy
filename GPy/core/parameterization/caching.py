# Copyright (c) 2012-2026, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""Re-export paramz caching helpers for GPy callers (CIP-000F Stage 0)."""

from paramz.caching import Cache_this, Cacher  # noqa: F401

__all__ = ["Cache_this", "Cacher"]
