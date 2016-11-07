from .posterior import Posterior
from ...util.linalg import jitchol, tdot, dtrtrs, dtrtri, pdinv
from ...util import diag
import numpy as np
from . import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class PEP(LatentFunctionInference):
    '''
    Sparse Gaussian processes using Power-Expectation Propagation
    for regression: alpha \approx 0 gives VarDTC and alpha = 1 gives FITC
    
    Reference: A Unifying Framework for Sparse Gaussian Process Approximation using 
    Power Expectation Propagation, https://arxiv.org/abs/1605.07066
    
    '''
    const_jitter = 1e-6
    
    def __init__(self, alpha):
        super(PEP, self).__init__()
        self.alpha = alpha

    def inference(self, kern, X, Z, likelihood, Y, mean_function=None, Y_metadata=None):
        assert mean_function is None, "inference with a mean function not implemented"

        num_inducing, _ = Z.shape
        num_data, output_dim = Y.shape

        #make sure the noise is not hetero
        sigma_n = likelihood.gaussian_variance(Y_metadata)
        if sigma_n.size >1:
            raise NotImplementedError("no hetero noise with this implementation of PEP")

        Kmm = kern.K(Z)
        Knn = kern.Kdiag(X)
        Knm = kern.K(X, Z)
        U = Knm

        #factor Kmm
        diag.add(Kmm, self.const_jitter)
        Kmmi, L, Li, _ = pdinv(Kmm)

        #compute beta_star, the effective noise precision
        LiUT = np.dot(Li, U.T)
        sigma_star = sigma_n + self.alpha * (Knn - np.sum(np.square(LiUT),0))
        beta_star = 1./sigma_star

        # Compute and factor A
        A = tdot(LiUT*np.sqrt(beta_star)) + np.eye(num_inducing)
        LA = jitchol(A)

        # back substitute to get b, P, v
        URiy = np.dot(U.T*beta_star,Y)
        tmp, _ = dtrtrs(L, URiy, lower=1)
        b, _ = dtrtrs(LA, tmp, lower=1)
        tmp, _ = dtrtrs(LA, b, lower=1, trans=1)
        v, _ = dtrtrs(L, tmp, lower=1, trans=1)
        tmp, _ = dtrtrs(LA, Li, lower=1, trans=0)
        P = tdot(tmp.T)

        alpha_const_term = (1.0-self.alpha) / self.alpha

        #compute log marginal
        log_marginal = -0.5*num_data*output_dim*np.log(2*np.pi) + \
                       -np.sum(np.log(np.diag(LA)))*output_dim + \
                       0.5*output_dim*(1+alpha_const_term)*np.sum(np.log(beta_star)) + \
                       -0.5*np.sum(np.square(Y.T*np.sqrt(beta_star))) + \
                       0.5*np.sum(np.square(b)) + 0.5*alpha_const_term*num_data*np.log(sigma_n)
        #compute dL_dR
        Uv = np.dot(U, v)
        dL_dR = 0.5*(np.sum(U*np.dot(U,P), 1) - (1.0+alpha_const_term)/beta_star + np.sum(np.square(Y), 1) - 2.*np.sum(Uv*Y, 1) \
            + np.sum(np.square(Uv), 1))*beta_star**2 

        # Compute dL_dKmm
        vvT_P = tdot(v.reshape(-1,1)) + P
        dL_dK = 0.5*(Kmmi - vvT_P)
        KiU = np.dot(Kmmi, U.T)
        dL_dK += self.alpha * np.dot(KiU*dL_dR, KiU.T)

        # Compute dL_dU
        vY = np.dot(v.reshape(-1,1),Y.T)
        dL_dU = vY - np.dot(vvT_P, U.T)
        dL_dU *= beta_star
        dL_dU -= self.alpha * 2.*KiU*dL_dR

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)
        dL_dthetaL += 0.5*alpha_const_term*num_data / sigma_n
        grad_dict = {'dL_dKmm': dL_dK, 'dL_dKdiag':dL_dR * self.alpha, 'dL_dKnm':dL_dU.T, 'dL_dthetaL':dL_dthetaL}

        #construct a posterior object
        post = Posterior(woodbury_inv=Kmmi-P, woodbury_vector=v, K=Kmm, mean=None, cov=None, K_chol=L)

        return post, log_marginal, grad_dict
