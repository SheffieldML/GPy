import numpy as np
import GPy
import warnings


class TestMisc:
    """
    Testing some utilities of misc
    """

    def setup(self):
        self._lim_val = np.finfo(np.float64).max
        self._lim_val_exp = np.log(self._lim_val)

    def test_safe_exp_upper(self):
        self.setup()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # always print
            assert np.isfinite(np.exp(self._lim_val_exp))
            assert np.isinf(np.exp(self._lim_val_exp + 1))
            assert np.isfinite(GPy.util.misc.safe_exp(self._lim_val_exp + 1))

            print(w)
            print(len(w))
            assert len(w) <= 1  # should have one overflow warning

    def test_safe_exp_lower(self):
        assert GPy.util.misc.safe_exp(1e-10) < np.inf
