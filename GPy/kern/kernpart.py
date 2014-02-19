# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
#from ...core.parameterized.Parameterized import set_as_parameter
from ...core.parameterization import Parameterized

class Kernpart(Parameterized):
    def __init__(self,input_dim,name):
        """
        The base class for a kernpart: a positive definite function 
        which forms part of a covariance function (kernel).

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        super(Kernpart, self).__init__(name)
        # the input dimensionality for the covariance
        self.input_dim = input_dim
        # the number of optimisable parameters
        # the name of the covariance function.
        # link to parameterized objects
        #self._X = None
    
    def connect_input(self, X):
        X.add_observer(self, self.on_input_change)
        #self._X = X
        
    def on_input_change(self, X):
        """
        During optimization this function will be called when
        the inputs X changed. Use this to update caches dependent
        on the inputs X.
        """
        # overwrite this to update kernel when inputs X change
        pass
    
        
#     def set_as_parameter_named(self, name, gradient, index=None, *args, **kwargs):
#         """
#         :param names:        name of parameter to set as parameter
#         :param gradient:     gradient method to get the gradient of this parameter
#         :param index:        index of where to place parameter in printing
#         :param args, kwargs: additional arguments to gradient
#     
#         Convenience method to connect Kernpart parameters:
#         parameter with name (attribute of this Kernpart) will be set as parameter with following name:
#         
#             kernel_name + _ + parameter_name
#     
#         To add the kernels name to the parameter name use this method to 
#         add parameters.
#         """
#         self.set_as_parameter(name, getattr(self, name), gradient, index, *args, **kwargs)
#     def set_as_parameter(self, name, array, gradient, index=None, *args, **kwargs):
#         """
#         See :py:func:`GPy.core.parameterized.Parameterized.set_as_parameter`
#         
#         Note: this method adds the kernels name in front of the parameter.
#         """
#         p = Param(self.name+"_"+name, array, gradient, *args, **kwargs)
#         if index is None:
#             self._parameters_.append(p)
#         else:
#             self._parameters_.insert(index, p)
#         self.__dict__[name] = p
    #set_as_parameter.__doc__ += set_as_parameter.__doc__  # @UndefinedVariable
#     def _get_params(self):
#         raise NotImplementedError
#     def _set_params(self,x):
#         raise NotImplementedError
#     def _get_param_names(self):
#         raise NotImplementedError
    def K(self,X,X2,target):
        raise NotImplementedError
    def Kdiag(self,X,target):
        raise NotImplementedError
    def _param_grad_helper(self,dL_dK,X,X2,target):
        raise NotImplementedError
    def dKdiag_dtheta(self,dL_dKdiag,X,target):
        # In the base case compute this by calling _param_grad_helper. Need to
        # override for stationary covariances (for example) to save
        # time.
        for i in range(X.shape[0]):
            self._param_grad_helper(dL_dKdiag[i], X[i, :][None, :], X2=None, target=target)
    def psi0(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dtheta(self,dL_dpsi0,Z,mu,S,target):
        raise NotImplementedError
    def dpsi0_dmuS(self,dL_dpsi0,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi1(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dtheta(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dZ(self,dL_dpsi1,Z,mu,S,target):
        raise NotImplementedError
    def dpsi1_dmuS(self,dL_dpsi1,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def psi2(self,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dZ(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dtheta(self,dL_dpsi2,Z,mu,S,target):
        raise NotImplementedError
    def dpsi2_dmuS(self,dL_dpsi2,Z,mu,S,target_mu,target_S):
        raise NotImplementedError
    def gradients_X(self, dL_dK, X, X2, target):
        raise NotImplementedError
    def dKdiag_dX(self, dL_dK, X, target):
        raise NotImplementedError
    def update_gradients_full(self, dL_dK, X):
        """Set the gradients of all parameters when doing full (N) inference."""
        raise NotImplementedError
    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        """Set the gradients of all parameters when doing sparse (M) inference."""
        raise NotImplementedError
    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        """Set the gradients of all parameters when doing variational (M) inference with uncertain inputs."""
        raise NotImplementedError

class Kernpart_stationary(Kernpart):
    def __init__(self, input_dim, lengthscale=None, ARD=False):
        self.input_dim = input_dim
        self.ARD = ARD
        if not ARD:
            self.num_params = 2
            if lengthscale is not None:
                self.lengthscale = np.asarray(lengthscale)
                assert self.lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                self.lengthscale = np.ones(1)
        else:
            self.num_params = self.input_dim + 1
            if lengthscale is not None:
                self.lengthscale = np.asarray(lengthscale)
                assert self.lengthscale.size == self.input_dim, "bad number of lengthscales"
            else:
                self.lengthscale = np.ones(self.input_dim)

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))

    def _set_params(self, x):
        self.lengthscale = x
        self.lengthscale2 = np.square(self.lengthscale)
        # reset cached results
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))
        self._Z, self._mu, self._S = np.empty(shape=(3, 1)) # cached versions of Z,mu,S


    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        # For stationary covariances, derivative of diagonal elements
        # wrt lengthscale is 0.
        target[0] += np.sum(dL_dKdiag)

    def dKdiag_dX(self, dL_dK, X, target):
        pass # true for all stationary kernels


class Kernpart_inner(Kernpart):
    def __init__(self,input_dim):
        """
        The base class for a kernpart_inner: a positive definite function which forms part of a kernel that is based on the inner product between inputs.

        :param input_dim: the number of input dimensions to the function
        :type input_dim: int

        Do not instantiate.
        """
        Kernpart.__init__(self, input_dim)

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._parameters_ = np.empty(shape=(3, 1))
