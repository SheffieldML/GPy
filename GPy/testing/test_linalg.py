import numpy as np
import scipy as sp
import pytest
from ..util.linalg import jitchol, trace_dot, ijk_jlk_to_il, ijk_ljk_to_ilk


class TestLinalg:
    def setup_method(self):
        # Create PD matrix
        A = np.random.randn(20, 100)
        self.A = A.dot(A.T)
        # compute Eigdecomp
        vals, vectors = np.linalg.eigh(self.A)
        # Set smallest eigenval to be negative with 5 rounds worth of jitter
        vals[vals.argmin()] = 0
        default_jitter = 1e-6 * np.mean(vals)
        vals[vals.argmin()] = -default_jitter * (10**3.5)
        self.A_corrupt = (vectors * vals).dot(vectors.T)

    def test_jitchol_success(self):
        """
        Expect 5 rounds of jitter to be added and for the recovered matrix to be
        identical to the corrupted matrix apart from the jitter added to the diagonal
        """
        self.setup_method()
        L = jitchol(self.A_corrupt, maxtries=5)
        A_new = L.dot(L.T)
        diff = A_new - self.A_corrupt
        np.testing.assert_allclose(
            diff, np.eye(A_new.shape[0]) * np.diag(diff).mean(), atol=1e-13
        )

    def test_jitchol_failure(self):
        self.setup_method()
        # Five rounds of jitter are required to make this matrix positive definite.
        with pytest.raises(sp.linalg.LinAlgError):
            jitchol(self.A_corrupt, maxtries=4)

    def test_trace_dot(self):
        self.setup_method()
        N = 5
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        trace = np.trace(A.dot(B))
        test_trace = trace_dot(A, B)
        np.testing.assert_allclose(trace, test_trace, atol=1e-13)

    def test_einsum_ij_jlk_to_ilk(self):
        self.setup_method()
        A = np.random.randn(15, 150, 5)
        B = np.random.randn(150, 50, 5)
        pure = np.einsum("ijk,jlk->il", A, B)
        quick = ijk_jlk_to_il(A, B)
        np.testing.assert_allclose(pure, quick)

    def test_einsum_ijk_ljk_to_ilk(self):
        self.setup_method()
        A = np.random.randn(150, 20, 5)
        B = np.random.randn(150, 20, 5)
        # B = A.copy()
        pure = np.einsum("ijk,ljk->ilk", A, B)
        quick = ijk_ljk_to_ilk(A, B)
        np.testing.assert_allclose(pure, quick)
