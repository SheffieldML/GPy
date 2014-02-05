import __builtin__
import numpy as np
import pylab as pb
from GPy import kern
from GPy.core import priors
from GPy.core import GP
from GPy.models import GPLVM
from GPy.likelihoods import Gaussian
from GPy import util
from GPy.util.linalg import pca
from GPy.inference import optimization
from GPy.util.linalg import pdinv, mdot, tdot, dpotrs, dtrtrs
from GPy.likelihoods import EP, Laplace

def initialise_latent(init, input_dim, Y):
    Xr = np.random.randn(Y.shape[0], input_dim)
    if init.lower() == 'pca':
        PC = pca(Y, input_dim)[0]
        Xr[:PC.shape[0], :PC.shape[1]] = PC
    return Xr

class ScaledGPLVM(GPLVM):
    """
    Scaled Gaussian Process Latent Variable Model

    Incorporates scaling parameter on each output dimension (column of Y).

    :param Y: observed data
    :type Y: np.ndarray
    :param input_dim: latent dimensionality
    :type input_dim: int
    :param init: initialisation method for the latent space
    :type init: 'PCA'|'random'

    """
    def __init__(self, Y, input_dim, init='PCA', X=None, kernel=None, normalize_Y=False):
        self.w=np.ones(Y.shape[1])
        if X is None:
            X = initialise_latent(init, input_dim, Y)
        if kernel is None:
            kernel = kern.rbf(input_dim, ARD=input_dim > 1) + kern.bias(input_dim, np.exp(-2))
        likelihood = Gaussian(Y, normalize=normalize_Y, variance=np.exp(-2.))
        GP.__init__(self, X, likelihood, kernel, normalize_X=False)
        self.set_prior('.*X', priors.Gaussian(0, 1))
        #self.constrain_fixed('.*w', 1) #Collapses to normal GPLVM
        self.constrain_positive('.*w')
        self.ensure_default_constraints()

    def _get_param_names(self):
        return __builtin__.sum([['X_%i_%i' % (n, q) for q in range(self.input_dim)] for n in range(self.num_data)], []) + __builtin__.sum([['w_%i' % (i) for i in range(self.likelihood.Y.shape[1])]], []) + GP._get_param_names(self)

    def _get_params(self):
        return np.hstack((self.X.flatten(), self.w.flatten(), GP._get_params(self)))        
        
    def _set_params(self, x):
        self.X = x[:self.num_data * self.input_dim].reshape(self.num_data, self.input_dim).copy()
        self.w=x[self.num_data * self.input_dim:(self.num_data * self.input_dim)+self.w.size]
        self._newset_GPparams(x[self.X.size+self.w.size:])

    def _newset_GPparams(self, p):
        new_kern_params = p[:self.kern.num_params_transformed()]
        new_likelihood_params = p[self.kern.num_params_transformed():]
        old_likelihood_params = self.likelihood._get_params()

        self.kern._set_params_transformed(new_kern_params)
        self.likelihood._set_params_transformed(new_likelihood_params)

        self.K = self.kern.K(self.X)

        #Re fit likelihood approximation (if it is an approx), as parameters have changed
        if isinstance(self.likelihood, Laplace):
            self.likelihood.fit_full(self.K)

        self.K += self.likelihood.covariance_matrix

        self.Ki, self.L, self.Li, self.K_logdet = pdinv(self.K)

        # the gradient of the likelihood wrt the covariance matrix
        if self.likelihood.YYT is None:
            # alpha = np.dot(self.Ki, self.likelihood.Y)
            #alpha, _ = dpotrs(self.L, self.likelihood.Y, lower=1)
            #Replacing Y with Y having its columns multiplied by values in w
            alpha, _ = dpotrs(self.L, np.transpose(np.vstack([self.likelihood.Y[:,i]*self.w[i] for i in range(self.w.shape[0])])), lower=1)

            self.dL_dK = 0.5 * (tdot(alpha) - self.output_dim * self.Ki)
        else:
            # tmp = mdot(self.Ki, self.likelihood.YYT, self.Ki)
            tmp, _ = dpotrs(self.L, np.asfortranarray(np.transpose(np.vstack([self.likelihood.YYT[:,i]*(self.w[i]**2) for i in range(self.w.shape[0])]))), lower=1)
            tmp, _ = dpotrs(self.L, np.asfortranarray(tmp.T), lower=1)
            self.dL_dK = 0.5 * (tmp - self.output_dim * self.Ki)

        #Adding dZ_dK (0 for a non-approximate likelihood, compensates for
        #additional gradients of K when log-likelihood has non-zero Z term)
        self.dL_dK += self.likelihood.dZ_dK

   
    def log_likelihood(self):
        """
        The log marginal likelihood of the GP.

        For an EP model,  can be written as the log likelihood of a regression
        model for a new variable Y* = v_tilde/tau_tilde, with a covariance
        matrix K* = K + diag(1./tau_tilde) plus a normalization term.
        """
        return ( self.num_data*np.sum(np.log(self.w)) -0.5 * self.num_data * self.output_dim * np.log(2.*np.pi) - 0.5 * self.output_dim * self.K_logdet + self._model_fit_term() + self.likelihood.Z)
        #return ( (-1.0*np.sum(np.log(self.w))) -(0.5 * self.num_data * self.output_dim * np.log(2.*np.pi)) - (0.5 * self.output_dim * self.K_logdet) + self._model_fit_term() + self.likelihood.Z)

    def _log_likelihood_gradients(self):
        dL_dX = self.kern.dK_dX(self.dL_dK, self.X)
        
        if self.likelihood.YYT is not None:
            #dL_dw = [(-1.0/self.w[i]) - np.multiply(self.Ki, np.transpose(np.vstack([self.likelihood.YYT[:,i]*(self.w[i]**2) for i in range(self.w.size)]))) for i in range(self.w.shape[0])]
            dL_dw = [(self.num_data/self.w[i]) - np.multiply(self.Ki, np.transpose(np.vstack([self.likelihood.YYT[:,i]*(self.w[i]**2) for i in range(self.w.size)]))) for i in range(self.w.shape[0])]
        else:
            #dL_dw=np.array([(-1.0/self.w[i]) -self.w[i]*np.dot(self.likelihood.Y[:,i], np.dot(self.Ki, self.likelihood.Y[:,i])) for i in range(self.w.size)])
            dL_dw=np.array([(self.num_data/self.w[i]) -self.w[i]*np.dot(self.likelihood.Y[:,i], np.dot(self.Ki, self.likelihood.Y[:,i])) for i in range(self.w.shape[0])])
        return np.hstack((dL_dX.flatten(), dL_dw.flatten(), GP._log_likelihood_gradients(self)))

    def _model_fit_term(self):
        """
        Computes the model fit using YYT if it's available
        """
        if self.likelihood.YYT is None:
            tmp, _ = dtrtrs(self.L, np.asfortranarray(np.transpose(np.vstack([self.likelihood.Y[:,i]*self.w[i] for i in range(self.w.shape[0])]))), lower=1)
            return -0.5 * np.sum(np.square(tmp))
            # return -0.5 * np.sum(np.square(np.dot(self.Li, self.likelihood.Y)))
        else:
            return -0.5 * np.sum(np.multiply(self.Ki, np.transpose(np.vstack([self.likelihood.YYT[:,i]*(self.w[i]**2) for i in range(self.w.shape[0])]))))
