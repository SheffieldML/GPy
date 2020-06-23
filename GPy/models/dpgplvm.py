# Copyright (c) 2015 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .bayesian_gplvm import BayesianGPLVM

class DPBayesianGPLVM(BayesianGPLVM):
    """
    Bayesian Gaussian Process Latent Variable Model with Descriminative prior
    """
    def __init__(self, Y, input_dim, X_prior, X=None, X_variance=None, init='PCA', num_inducing=10,
                 Z=None, kernel=None, inference_method=None, likelihood=None,
                 name='bayesian gplvm', mpi_comm=None, normalizer=None,
                 missing_data=False, stochastic=False, batchsize=1):
        super(DPBayesianGPLVM,self).__init__(Y=Y, input_dim=input_dim, X=X, X_variance=X_variance,
                                             init=init, num_inducing=num_inducing, Z=Z, kernel=kernel,
                                             inference_method=inference_method, likelihood=likelihood,
                                             mpi_comm=mpi_comm, normalizer=normalizer,
                                             missing_data=missing_data, stochastic=stochastic,
                                             batchsize=batchsize, name='dp bayesian gplvm')
        self.X.mean.set_prior(X_prior)
        self.link_parameter(X_prior)
