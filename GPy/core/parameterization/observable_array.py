# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#
# CIP-000F Stage 3: GPy module path for ObsAr.
#
# Model X/Y are stored as plain ``ndarray`` (see ``GPy.core.gp._as_data_array``).
# ``ObsAr`` remains importable for expectation propagation and for unpickling
# models saved when X/Y were still ``paramz.ObsAr``. Stage 4 may replace the
# body of this module and keep ``sys.modules`` aliases so old pickles load.

from paramz.core.observable_array import ObsAr

__all__ = ["ObsAr"]
