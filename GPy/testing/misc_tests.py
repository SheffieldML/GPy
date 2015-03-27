import numpy as np
import scipy as sp
import GPy

class MiscTests(np.testing.TestCase):
    """
    Testing some utilities of misc
    """
    def setUp(self):
        self._lim_val = np.finfo(np.float64).max
        self._lim_val_exp = np.log(self._lim_val)

    def test_safe_exp_upper(self):
        assert np.exp(self._lim_val_exp + 1) == np.inf
        assert GPy.util.misc.safe_exp(self._lim_val_exp + 1) < np.inf

    def test_safe_exp_lower(self):
        assert GPy.util.misc.safe_exp(1e-10) < np.inf
