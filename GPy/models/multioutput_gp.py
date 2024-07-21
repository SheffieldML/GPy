# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from ..core.model import Model
from ..core.parameterization.variational import VariationalPosterior
from ..core.mapping import Mapping
from .. import likelihoods
from ..likelihoods.gaussian import Gaussian
from .. import kern
from ..kern import DiffKern
from ..inference.latent_function_inference import exact_gaussian_inference, expectation_propagation
from ..util.normalizer import Standardize
from .. import util
from paramz import ObsAr
from ..core.gp import GP

from ..util.multioutput import index_to_slices
import logging
import warnings
logger = logging.getLogger("GP")

class MultioutputGP(GP):
    """
    Gaussian process model for using observations from multiple likelihoods and different kernels
    :param X_list: input observations in a list for each likelihood
    :param Y: output observations in a list for each likelihood
    :param kernel_list: kernels in a list for each likelihood
    :param likelihood_list: likelihoods in a list
    :param kernel_cross_covariances: Cross covariances between different likelihoods. See class MultioutputKern for more
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    """
    def __init__(self, X_list, Y_list, kernel_list, likelihood_list, name='multioutputgp', kernel_cross_covariances={}, inference_method=None):
        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)
        
        assert isinstance(kernel_list, list)
        kernel = kern.MultioutputDerivativeKern(kernels=kernel_list, cross_covariances=kernel_cross_covariances)

        assert isinstance(likelihood_list, list)
        likelihood = likelihoods.MultioutputLikelihood(likelihood_list)
        
        if inference_method is None:
            if all([isinstance(l, Gaussian) for l in likelihood_list]):
                inference_method = exact_gaussian_inference.ExactGaussianInference()
            else:
                inference_method = expectation_propagation.EP() 
        
        super(MultioutputGP, self).__init__(X,Y,kernel,likelihood, Y_metadata={'output_index':self.output_index, 'trials':np.ones(self.output_index.shape)}, inference_method = inference_method)

    def predict_noiseless(self,  Xnew, full_cov=False, Y_metadata=None, kern=None):
        if isinstance(Xnew, list):
            Xnew, _, ind  = util.multioutput.build_XY(Xnew,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind, 'trials': np.ones(ind.shape)}
        return super(MultioutputGP, self).predict_noiseless(Xnew, full_cov, Y_metadata, kern)
    
    def predict(self, Xnew, full_cov=False, Y_metadata=None, kern=None, likelihood=None, include_likelihood=True):
        if isinstance(Xnew, list):
            Xnew, _, ind = util.multioutput.build_XY(Xnew,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind, 'trials': np.ones(ind.shape)}
        return super(MultioutputGP, self).predict(Xnew, full_cov, Y_metadata, kern, likelihood, include_likelihood)
    
    def predict_quantiles(self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None):
        if isinstance(X, list):
            X, _, ind  = util.multioutput.build_XY(X,None)
            if Y_metadata is None:
                Y_metadata={'output_index': ind, 'trials': np.ones(ind.shape)}
        return super(MultioutputGP, self).predict_quantiles(X, quantiles, Y_metadata, kern, likelihood)

    def predictive_gradients(self, Xnew, kern=None):
        """
        Compute the derivatives of the predicted latent function with respect
        to X*
        
        Given a set of points at which to predict X* (size [N*,Q]), compute the
        derivatives of the mean and variance. Resulting arrays are sized:
            dmu_dX* -- [N*, Q ,D], where D is the number of output in this GP
            (usually one).
        
        Note that this is not the same as computing the mean and variance of
        the derivative of the function!

         dv_dX*  -- [N*, Q],    (since all outputs have the same variance)
        :param X: The points at which to get the predictive gradients
        :type X: np.ndarray (Xnew x self.input_dim)
        :returns: dmu_dX, dv_dX
        :rtype: [np.ndarray (N*, Q ,D), np.ndarray (N*,Q) ]

        """
        
        if isinstance(Xnew, list):
            Xnew, _, ind  = util.multioutput.build_XY(Xnew, None)
        
        slices = index_to_slices(Xnew[:,-1])

        if kern is None:
            kern = self.kern

        if all([(isinstance(k, DiffKern)) for k in self.kern.kern[1:]]):
            """
            Compute the gradients of the predicted latent function and predicted
            partial derivatives with respect to X*.

            This works only for models that observe the gradient of the latent function.

            Xnew is given as a list of arrays, where each array X*_i (size [N_i*, Q])
            contains points at which to compute gradients for each predicted latent
            function or partial derivative.

            Resulting arrays are sized [sum_i^D : N_i*, Q]

            Passing a list of only one array [X*] returns only gradients of
            the predicted latent function and does not compute gradients of
            predicted partial derivatives.

            In this case the resulting arrays are sized [N*, Q].
            
            :param Xnew: points at which to compute predictive gradients
            :type Xnew: list
            :type Xnew[i]: np.darray (sum_i^D : N_i*, Q)
            :returns: dmu_dX, dv_dX
            :rtype: (np.ndarray (sum_i^D : N_i*, Q), np.ndarray (sum_i^D : N_i*, Q))
            """

            dims = Xnew.shape[1] - 1

            mean_jac = np.empty((Xnew.shape[0], dims))
            var_jac = np.empty((Xnew.shape[0], dims))

            X = self._predictive_variable
            alpha = self.posterior.woodbury_vector
            Wi = self.posterior.woodbury_inv

            k = kern.K(Xnew, X)
            for dimX in range(dims):
                dk_dx = kern.dK_dX(Xnew, X, dimX)
                dk_dxdiag = kern.dK_dXdiag(Xnew, dimX)

                mean_jac[:,dimX] = np.dot(dk_dx, alpha).flatten()
                var_jac[:,dimX] = dk_dxdiag - 2*(np.dot(k, Wi)*dk_dx).sum(-1)
            return mean_jac, var_jac

        mean_jac = np.empty((Xnew.shape[0],Xnew.shape[1]-1,self.output_dim))
        for i in range(self.output_dim):
            mean_jac[:,:,i] = kern.gradients_X(self.posterior.woodbury_vector[:,i:i+1].T, Xnew, self._predictive_variable)[:,0:-1]

        # gradients wrt the diagonal part k_{xx}
        dv_dX = kern.gradients_X(np.eye(Xnew.shape[0]), Xnew)[:,0:-1]
        #grads wrt 'Schur' part K_{xf}K_{ff}^{-1}K_{fx}
        if self.posterior.woodbury_inv.ndim == 3:
            tmp = np.empty(dv_dX.shape + (self.posterior.woodbury_inv.shape[2],))
            tmp[:] = dv_dX[:,:,None]
            for i in range(self.posterior.woodbury_inv.shape[2]):
                alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable), self.posterior.woodbury_inv[:, :, i])
                tmp[:, :, i] += kern.gradients_X(alpha, Xnew, self._predictive_variable)
        else:
            tmp = dv_dX
            alpha = -2.*np.dot(kern.K(Xnew, self._predictive_variable), self.posterior.woodbury_inv)
            tmp += kern.gradients_X(alpha, Xnew, self._predictive_variable)[:,0:-1]
        return mean_jac, tmp
    
    def log_predictive_density(self, x_test, y_test, Y_metadata=None):
        if isinstance(x_test, list):
            x_test, y_test, ind  = util.multioutput.build_XY(x_test, y_test)
            if Y_metadata is None:
                Y_metadata={'output_index': ind, 'trials': np.ones(ind.shape)}
        return super(MultioutputGP, self).log_predictive_density(x_test, y_test, Y_metadata)
    
    def set_XY(self, X=None, Y=None):
        if isinstance(X, list):
            X, _, self.output_index  = util.multioutput.build_XY(X, None)
        if isinstance(Y, list):
            _, Y, self.output_index  = util.multioutput.build_XY(Y, Y)      
                
        self.update_model(False)
        if Y is not None:
            self.Y = ObsAr(Y)
            self.Y_normalized = self.Y
        if X is not None:
            self.X = ObsAr(X)
            
        self.Y_metadata={'output_index': self.output_index, 'trials': np.ones(self.output_index.shape)}
        if isinstance(self.inference_method, expectation_propagation.EP):      
            self.inference_method.reset()
        self.update_model(True)