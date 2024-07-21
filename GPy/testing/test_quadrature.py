from __future__ import print_function, division
import numpy as np
from ..util.quad_integrate import quadgk_int, quadvgk


class TestQuad:
    """
    test file for checking implementation of gaussian-kronrod quadrature.
    we will take a function which can be integrated analytically and check if quadgk result is similar or not!
    through this file we can test how numerically accurate quadrature implementation in native numpy or manual code is.
    """

    def test_infinite_quad(self):
        def f(x):
            return np.exp(-0.5 * x**2) * np.power(x, np.arange(3)[:, None])

        quad_int_val = quadgk_int(f)
        real_val = np.sqrt(np.pi * 2)
        np.testing.assert_almost_equal(real_val, quad_int_val[0], decimal=7)

    def test_finite_quad(self):
        def f2(x):
            return x**2

        quad_int_val = quadvgk(f2, 1.0, 2.0)
        real_val = 7 / 3.0
        np.testing.assert_almost_equal(real_val, quad_int_val, decimal=5)


if __name__ == "__main__":

    def f(x):
        return np.exp(-0.5 * x**2) * np.power(x, np.arange(3)[:, None])

    quad_int_val = quadgk_int(f)
    real_val = np.sqrt(np.pi * 2)
    np.testing.assert_almost_equal(real_val, quad_int_val[0], decimal=7)
    print(quadgk_int(f))
