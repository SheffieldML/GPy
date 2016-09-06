# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Kurt Cutajar

import numpy as np
from .stationary import Stationary
from paramz.caching import Cache_this


class GridKern(Stationary):

	def __init__(self, input_dim, variance, lengthscale, ARD, active_dims, name, originalDimensions, useGPU=False):
		super(GridKern, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, useGPU=useGPU)
		self.originalDimensions = originalDimensions

	@Cache_this(limit=3, ignore_args=())
	def dKd_dVar(self, X, X2=None):
		"""
		Derivative of Kernel function wrt variance applied on inputs X and X2.
		In the stationary case there is an inner function depending on the
		distances from X to X2, called r.

		dKd_dVar(X, X2) = dKdVar_of_r((X-X2)**2)
		"""
		r = self._scaled_dist(X, X2)
		return self.dKdVar_of_r(r)

	@Cache_this(limit=3, ignore_args=())
	def dKd_dLen(self, X, dimension, lengthscale, X2=None):
		"""
		Derivate of Kernel function wrt lengthscale applied on inputs X and X2.
		In the stationary case there is an inner function depending on the
		distances from X to X2, called r.

		dKd_dLen(X, X2) = dKdLen_of_r((X-X2)**2)
		"""
		r = self._scaled_dist(X, X2)
		return self.dKdLen_of_r(r, dimension, lengthscale)

class GridRBF(GridKern):
	"""
	Similar to regular RBF but supplemented with methods required for Gaussian grid regression
	Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

	.. math::

	   k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg)

	"""
	_support_GPU = True
	def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='gridRBF', originalDimensions=1, useGPU=False):
		super(GridRBF, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name, originalDimensions, useGPU=useGPU)

	def K_of_r(self, r):
		return (self.variance**(float(1)/self.originalDimensions)) * np.exp(-0.5 *  r**2)

	def dKdVar_of_r(self, r):
		"""
		Compute derivative of kernel wrt variance
		"""
		return np.exp(-0.5 * r**2)

	def dKdLen_of_r(self, r, dimCheck, lengthscale):
		"""
		Compute derivative of kernel for dimension wrt lengthscale
		Computation of derivative changes when lengthscale corresponds to
		the dimension of the kernel whose derivate is being computed. 
		"""
		if (dimCheck == True):
			return (self.variance**(float(1)/self.originalDimensions)) * np.exp(-0.5 * r**2) * (r**2) / (lengthscale**(float(1)/self.originalDimensions))
		else:
			return (self.variance**(float(1)/self.originalDimensions)) * np.exp(-0.5 * r**2) / (lengthscale**(float(1)/self.originalDimensions))

	def dK_dr(self, r):
		return -r*self.K_of_r(r)