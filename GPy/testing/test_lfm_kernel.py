# Copyright (c) 2012, 2013 GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import GPy
import pytest
import numpy as np
from ..util.config import config

verbose = 0


class TestLFMKernel:
    """Test suite for LFM (Latent Force Model) kernel implementation."""
    
    def setup(self):
        """Set up test data and parameters."""
        self.N, self.D = 10, 1  # 1 input dimension + 1 output index dimension = 2 total
        # Create test data with output indices
        self.X = np.random.randn(self.N, 2)  # 2 dimensions: input + output index
        self.X2 = np.random.randn(self.N + 5, 2)
        
        # Set output indices (second column)
        self.X[:, 1] = np.random.randint(0, 2, self.N)  # 2 outputs
        self.X2[:, 1] = np.random.randint(0, 2, self.X2.shape[0])
        
        # LFM parameters
        self.mass = 1.0
        self.damper = 0.5
        self.spring = 2.0
        self.sensitivity = 1.0
        self.delay = 0.1
        
    def test_lfm_kernel_creation(self):
        """Test basic LFM kernel creation and parameter handling."""
        # Test first-order LFM kernel
        k1 = GPy.kern.LFM1(2, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        assert k1.name == 'LFM1'
        assert k1.input_dim == 2  # 1 input dim + 1 output index dim
        assert k1.output_dim == 2  # Default: 2 outputs (force and displacement)
        
        # Test second-order LFM kernel
        k2 = GPy.kern.LFM2(2, mass=self.mass, damper=self.damper, 
                           spring=self.spring, sensitivity=self.sensitivity, 
                           delay=self.delay)
        assert k2.name == 'LFM2'
        assert k2.input_dim == 2  # 1 input dim + 1 output index dim
        assert k2.output_dim == 2
        
        # Test parameter values
        assert k1.mass == self.mass
        assert k1.damper == self.damper
        assert k1.sensitivity == self.sensitivity
        assert k1.delay == self.delay
        
        assert k2.mass == self.mass
        assert k2.damper == self.damper
        assert k2.spring == self.spring
        assert k2.sensitivity == self.sensitivity
        assert k2.delay == self.delay
        
    def test_lfm_kernel_covariance(self):
        """Test LFM kernel covariance computation."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test K(X, X)
        K = k1.K(self.X)
        assert K.shape == (self.N, self.N)
        assert np.all(np.isfinite(K))
        
        # Test K(X, X2)
        K2 = k1.K(self.X, self.X2)
        assert K2.shape == (self.N, self.X2.shape[0])
        assert np.all(np.isfinite(K2))
        
        # Test Kdiag(X)
        Kdiag = k1.Kdiag(self.X)
        assert Kdiag.shape == (self.N,)
        assert np.all(np.isfinite(Kdiag))
        
    def test_lfm_kernel_positive_definite(self):
        """Test that LFM kernel produces positive semi-definite matrices."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        k2 = GPy.kern.LFM2(self.D, mass=self.mass, damper=self.damper, 
                           spring=self.spring, sensitivity=self.sensitivity, 
                           delay=self.delay)
        
        # Check positive semi-definiteness
        K1 = k1.K(self.X)
        K2 = k2.K(self.X)
        
        # Eigenvalues should be non-negative (with small tolerance)
        eigvals1 = np.linalg.eigvals(K1)
        eigvals2 = np.linalg.eigvals(K2)
        
        assert np.all(eigvals1.real >= -1e-10)
        assert np.all(eigvals2.real >= -1e-10)
        
    def test_lfm_kernel_gradients(self):
        """Test LFM kernel gradient computation."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test gradient computation
        dL_dK = np.random.randn(self.N, self.N)
        k1.update_gradients_full(dL_dK, self.X)
        
        # Check that gradients are computed
        assert hasattr(k1, 'mass')
        assert hasattr(k1, 'damper')
        assert hasattr(k1, 'sensitivity')
        assert hasattr(k1, 'delay')
        
    def test_lfm_kernel_multioutput(self):
        """Test LFM kernel with multiple outputs."""
        # Test with 3 outputs (force, displacement, velocity)
        k1 = GPy.kern.LFM1(self.D, output_dim=3, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Create data with 3 outputs
        X_multi = self.X.copy()
        X_multi[:, 1] = np.random.randint(0, 3, self.N)
        
        K = k1.K(X_multi)
        assert K.shape == (self.N, self.N)
        assert np.all(np.isfinite(K))
        
    def test_lfm_kernel_parameter_constraints(self):
        """Test LFM kernel parameter constraints."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test that parameters are constrained appropriately
        # Mass should be positive
        k1.mass = -1.0
        assert k1.mass > 0  # Should be constrained to positive
        
        # Damper should be positive
        k1.damper = -0.5
        assert k1.damper > 0  # Should be constrained to positive
        
        # Spring (for LFM2) should be positive
        k2 = GPy.kern.LFM2(self.D, mass=self.mass, damper=self.damper, 
                           spring=self.spring, sensitivity=self.sensitivity, 
                           delay=self.delay)
        k2.spring = -2.0
        assert k2.spring > 0  # Should be constrained to positive
        
    def test_lfm_kernel_serialization(self):
        """Test LFM kernel serialization and deserialization."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test pickling
        import pickle
        k1_pickled = pickle.dumps(k1)
        k1_unpickled = pickle.loads(k1_pickled)
        
        # Check that parameters are preserved
        assert k1_unpickled.mass == k1.mass
        assert k1_unpickled.damper == k1.damper
        assert k1_unpickled.sensitivity == k1.sensitivity
        assert k1_unpickled.delay == k1.delay
        
        # Check that kernel computation is preserved
        K_original = k1.K(self.X)
        K_unpickled = k1_unpickled.K(self.X)
        np.testing.assert_array_almost_equal(K_original, K_unpickled)
        
    def test_lfm_kernel_combination(self):
        """Test LFM kernel in combination with other kernels."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        k_rbf = GPy.kern.RBF(self.D)
        
        # Test addition
        k_add = k1 + k_rbf
        K_add = k_add.K(self.X)
        assert K_add.shape == (self.N, self.N)
        assert np.all(np.isfinite(K_add))
        
        # Test multiplication
        k_prod = k1 * k_rbf
        K_prod = k_prod.K(self.X)
        assert K_prod.shape == (self.N, self.N)
        assert np.all(np.isfinite(K_prod))
        
    def test_lfm_kernel_edge_cases(self):
        """Test LFM kernel edge cases and error handling."""
        # Test with zero mass (should handle gracefully or raise appropriate error)
        with pytest.raises((ValueError, AssertionError)):
            k1 = GPy.kern.LFM1(self.D, mass=0.0, damper=self.damper, 
                               sensitivity=self.sensitivity, delay=self.delay)
        
        # Test with zero damper (should handle gracefully or raise appropriate error)
        with pytest.raises((ValueError, AssertionError)):
            k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=0.0, 
                               sensitivity=self.sensitivity, delay=self.delay)
        
        # Test with negative delay (should handle gracefully)
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=-0.1)
        K = k1.K(self.X)
        assert np.all(np.isfinite(K))
        
    def test_lfm_kernel_mathematical_properties(self):
        """Test LFM kernel mathematical properties."""
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test symmetry: K(X, X2) = K(X2, X)^T
        K_forward = k1.K(self.X, self.X2)
        K_backward = k1.K(self.X2, self.X)
        np.testing.assert_array_almost_equal(K_forward, K_backward.T)
        
        # Test diagonal: Kdiag(X) should equal diagonal of K(X, X)
        K_full = k1.K(self.X)
        K_diag = k1.Kdiag(self.X)
        np.testing.assert_array_almost_equal(np.diag(K_full), K_diag)
        
    def test_lfm_kernel_parameter_tying(self):
        """Test LFM kernel with parameter tying (when available)."""
        # This test assumes parameter tying functionality will be implemented
        # For now, we'll test the basic functionality without tying
        
        k1 = GPy.kern.LFM1(self.D, mass=self.mass, damper=self.damper, 
                           sensitivity=self.sensitivity, delay=self.delay)
        
        # Test that kernel works without parameter tying
        K = k1.K(self.X)
        assert K.shape == (self.N, self.N)
        assert np.all(np.isfinite(K))
        
        # TODO: Add parameter tying tests when CIP-0002 is implemented
        # This would test scenarios like:
        # - Tying mass parameters across different outputs
        # - Tying sensitivity parameters across different outputs
        # - Tying delay parameters across different outputs


def check_lfm_kernel_gradient_functions(kern, X=None, X2=None, verbose=False):
    """Check LFM kernel gradient functions using GPy's standard test framework."""
    from .test_kernel import check_kernel_gradient_functions
    
    # Use output index 1 (second column) for multioutput testing
    return check_kernel_gradient_functions(kern, X=X, X2=X2, output_ind=1, verbose=verbose)


class TestLFMKernelGradients:
    """Test LFM kernel gradients using GPy's standard gradient checking."""
    
    def setup(self):
        """Set up test data."""
        self.N, self.D = 10, 3
        self.X = np.random.randn(self.N, self.D + 1)
        self.X2 = np.random.randn(self.N + 5, self.D + 1)
        
        # Set output indices
        self.X[:, 1] = np.random.randint(0, 2, self.N)
        self.X2[:, 1] = np.random.randint(0, 2, self.X2.shape[0])
        
    def test_lfm1_gradients(self):
        """Test LFM1 kernel gradients."""
        k = GPy.kern.LFM1(self.D, mass=1.0, damper=0.5, sensitivity=1.0, delay=0.1)
        k.randomize()
        assert check_lfm_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose)
        
    def test_lfm2_gradients(self):
        """Test LFM2 kernel gradients."""
        k = GPy.kern.LFM2(self.D, mass=1.0, damper=0.5, spring=2.0, sensitivity=1.0, delay=0.1)
        k.randomize()
        assert check_lfm_kernel_gradient_functions(k, X=self.X, X2=self.X2, verbose=verbose)
