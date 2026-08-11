# Copyright (c) 2025, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import erf, erfcx
from .kern import Kern
from ...core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


class LFM1(Kern):
    """
    Latent Force Model kernel for first-order differential equation (LFM1).
    
    This kernel implements the Single Input Motif (SIM) kernel for first-order
    differential equations of the form:
    
    .. math::
       \\frac{dx(t)}{dt} = B + S f(t-\\delta) - D x(t)
    
    where:
    - B is the initial level (initVal)
    - S is the sensitivity to the latent force
    - D is the decay rate
    - δ is the time delay
    - f(t) is the latent force with RBF covariance
    
    The kernel is designed to work with GPy's multioutput framework where
    the second input dimension is used as the output index.
    
    :param input_dim: Input dimension (should be 2: 1 for time + 1 for output index)
    :type input_dim: int
    :param output_dim: Number of outputs (default: 2 for force and displacement)
    :type output_dim: int
    :param mass: Mass parameter (default: 1.0)
    :type mass: float
    :param damper: Damping parameter (decay rate, default: 1.0)
    :type damper: float
    :param sensitivity: Sensitivity to latent force (default: 1.0)
    :type sensitivity: float
    :param delay: Time delay (default: 0.0)
    :type delay: float
    :param variance: Kernel variance (default: 1.0)
    :type variance: float
    :param lengthscale: Lengthscale of latent RBF kernel (default: 1.0)
    :type lengthscale: float
    :param active_dims: Active dimensions for the kernel
    :type active_dims: list
    :param name: Kernel name
    :type name: str
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        mass=1.0,
        damper=1.0,
        sensitivity=1.0,
        delay=0.0,
        variance=1.0,
        lengthscale=1.0,
        active_dims=None,
        name="LFM1",
    ):
        # Validate input dimension (should be 2: time + output index)
        assert input_dim == 2, "LFM1 kernel requires exactly 2 input dimensions (time + output index)"
        
        super(LFM1, self).__init__(
            input_dim=input_dim, active_dims=active_dims, name=name
        )
        
        self.output_dim = output_dim
        
        # Initialize parameters with constraints
        self.mass = Param("mass", mass, Logexp())  # Must be positive
        self.damper = Param("damper", damper, Logexp())  # Must be positive (decay rate)
        self.sensitivity = Param("sensitivity", sensitivity, Logexp())  # Must be positive
        self.delay = Param("delay", delay)  # Can be negative
        self.variance = Param("variance", variance, Logexp())  # Must be positive
        self.lengthscale = Param("lengthscale", lengthscale, Logexp())  # Must be positive
        
        # Link parameters for optimization
        self.link_parameters(
            self.mass, self.damper, self.sensitivity, 
            self.delay, self.variance, self.lengthscale
        )
        
        # Kernel properties
        self.is_stationary = False
        self.is_normalized = False
        self.positive_time = True

    @Cache_this(limit=3)
    def K(self, X, X2=None):
        """
        Compute the kernel matrix.
        
        :param X: Input array of shape (n, 2) where first column is time, second is output index
        :param X2: Second input array (optional)
        :return: Kernel matrix
        """
        if X2 is None:
            X2 = X
            
        # Extract time and output indices
        t1 = X[:, 0:1]  # Time points
        t2 = X2[:, 0:1]  # Time points for X2
        idx1 = X[:, 1:2].astype(int)  # Output indices
        idx2 = X2[:, 1:2].astype(int)  # Output indices for X2
        
        # Apply time delay
        t1_delayed = t1 - self.delay
        t2_delayed = t2 - self.delay
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(t1_delayed, t2_delayed, idx1, idx2)
        
        return K

    def _compute_kernel_matrix(self, t1, t2, idx1, idx2):
        """
        Compute the kernel matrix using the analytical solution.
        
        This implements the SIM kernel computation based on the MATLAB implementation.
        """
        n1, n2 = t1.shape[0], t2.shape[0]
        K = np.zeros((n1, n2))
        
        # Parameters
        D = self.damper  # Decay rate
        sigma = self.lengthscale * np.sqrt(2)  # Lengthscale for RBF
        S = self.sensitivity  # Sensitivity
        
        # Compute kernel for each pair of points
        for i in range(n1):
            for j in range(n2):
                # Only compute kernel if output indices match (same output)
                if idx1[i] == idx2[j]:
                    K[i, j] = self._compute_kernel_element(t1[i, 0], t2[j, 0], D, sigma, S)
        
        # Apply variance scaling
        K = self.variance * K
        
        return K

    def _compute_kernel_element(self, t1, t2, D, sigma, S):
        """
        Compute a single kernel element using the analytical solution.
        
        Based on the MATLAB simComputeH function.
        """
        # Apply time delay (already done in K method)
        
        # Compute the kernel using error functions
        # This is the analytical solution for the first-order ODE kernel
        
        # For now, implement a simplified version
        # TODO: Implement full analytical solution with error functions
        
        # Simplified implementation based on exponential decay
        diff_t = t1 - t2
        abs_diff_t = np.abs(diff_t)
        
        # Basic exponential decay kernel
        # This is a placeholder - need to implement the full analytical solution
        kernel_val = np.exp(-D * abs_diff_t) * np.exp(-0.5 * (diff_t / sigma) ** 2)
        
        return kernel_val

    @Cache_this(limit=3)
    def Kdiag(self, X):
        """
        Compute the diagonal of the kernel matrix.
        
        :param X: Input array
        :return: Diagonal of kernel matrix
        """
        # Extract time and output indices
        t = X[:, 0:1]
        idx = X[:, 1:2].astype(int)
        
        # Apply time delay
        t_delayed = t - self.delay
        
        # Compute diagonal elements
        diag = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            diag[i] = self._compute_kernel_element(t_delayed[i, 0], t_delayed[i, 0], 
                                                 self.damper, self.lengthscale * np.sqrt(2), 
                                                 self.sensitivity)
        
        # Apply variance scaling
        diag = self.variance * diag
        
        return diag

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Update gradients with respect to parameters.
        
        :param dL_dK: Gradient of objective with respect to kernel matrix
        :param X: Input array
        :param X2: Second input array (optional)
        """
        if X2 is None:
            X2 = X
            
        # Extract time and output indices
        t1 = X[:, 0:1]
        t2 = X2[:, 0:1]
        idx1 = X[:, 1:2].astype(int)
        idx2 = X2[:, 1:2].astype(int)
        
        # Apply time delay
        t1_delayed = t1 - self.delay
        t2_delayed = t2 - self.delay
        
        # Initialize gradients
        self.mass.gradient = 0.0
        self.damper.gradient = 0.0
        self.sensitivity.gradient = 0.0
        self.delay.gradient = 0.0
        self.variance.gradient = 0.0
        self.lengthscale.gradient = 0.0
        
        # Compute gradients
        # TODO: Implement gradient computation
        # For now, use finite differences or analytical gradients
        
        # Simplified gradient computation
        n1, n2 = t1.shape[0], t2.shape[0]
        
        for i in range(n1):
            for j in range(n2):
                if idx1[i] == idx2[j]:
                    # Compute gradients for each parameter
                    # This is a placeholder - need to implement proper gradients
                    pass

    def update_gradients_diag(self, dL_dKdiag, X):
        """
        Update gradients with respect to parameters for diagonal computation.
        
        :param dL_dKdiag: Gradient of objective with respect to diagonal
        :param X: Input array
        """
        # TODO: Implement diagonal gradient computation
        pass

    def parameters_changed(self):
        """
        Called when parameters have changed.
        """
        # Clear any cached computations
        pass

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.
        """
        input_dict = super(LFM1, self)._save_to_input_dict()
        input_dict["class"] = "GPy.kern.LFM1"
        input_dict["output_dim"] = self.output_dim
        input_dict["mass"] = self.mass
        input_dict["damper"] = self.damper
        input_dict["sensitivity"] = self.sensitivity
        input_dict["delay"] = self.delay
        input_dict["variance"] = self.variance
        input_dict["lengthscale"] = self.lengthscale
        return input_dict

    @staticmethod
    def from_dict(input_dict):
        """
        Create a kernel from a dictionary.
        """
        return LFM1(
            input_dim=input_dict["input_dim"],
            output_dim=input_dict["output_dim"],
            mass=input_dict["mass"],
            damper=input_dict["damper"],
            sensitivity=input_dict["sensitivity"],
            delay=input_dict["delay"],
            variance=input_dict["variance"],
            lengthscale=input_dict["lengthscale"],
            active_dims=input_dict["active_dims"],
            name=input_dict["name"]
        )
