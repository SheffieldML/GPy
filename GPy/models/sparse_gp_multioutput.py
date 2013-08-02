# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import SparseGP
from .. import likelihoods
from .. import kern
from ..util import multioutput


import pylab as pb

class SparseGPMultioutput(SparseGP):
    """
    Multiple output Gaussian process

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: input observations
    :param Y_list: observed values
    :param L_list: a GPy likelihood, defaults to Binomial with probit link_function
    :param kernel_list: a GPy kernel, defaults to rbf
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(self,X_list,Y_list,kernel_list=None,normalize_X=False,normalize_Y=False,Z_list=None,num_inducing_list=10,X_variance=None,W=1,mixed_noise_list=[]): #TODO W

        assert len(X_list) == len(Y_list)
        index = []
        for x,y,j in zip(X_list,Y_list,range(len(X_list))):
            assert x.shape[0] == y.shape[0]
            index.append(np.repeat(j,y.size)[:,None])
        index = np.vstack(index)


        self.likelihood_list = []
        if mixed_noise_list == []:
            for Y in Y_list:
                self.likelihood_list.append(likelihoods.Gaussian(Y,normalize = normalize_Y))

            Y = np.vstack([l_.Y for l_ in self.likelihood_list])
            likelihood = likelihoods.Gaussian(Y,normalize=False)
            likelihood.index = index

        else:
            assert len(Y_list) == len(mixed_noise_list)
            for noise,Y in zip(mixed_noise_list,Y_list):
                self.likelihood_list.append(likelihoods.EP(Y,noise))
            likelihood = likelihoods.EP_Mixed_Noise(Y_list, mixed_noise_list)

        """
        if noise_list == []:
            self.likelihood_list = []
            for Y in Y_list:
                self.likelihood_list.append(likelihoods.Gaussian(Y,normalize = normalize_Y))

        Y = np.vstack([l_.Y for l_ in self.likelihood_list])
        likelihood = likelihoods.Gaussian(Y,normalize=False)
        likelihood.index = index
        """
        X = np.hstack([np.vstack(X_list),index])
        original_dim = X.shape[1] - 1

        if kernel_list is None:
            kernel_list = [[kern.rbf(original_dim)],[kern.white(original_dim+1)]]

        mkernel = multioutput.build_cor_kernel(input_dim=original_dim, Nout=len(X_list), CK = kernel_list[0], NC = kernel_list[1], W=1)

        z_index = []
        if Z_list is None:
            if isinstance(num_inducing_list,int):
                num_inducing_list = [num_inducing_list for Xj in X_list]
            Z_list = []
            for Xj,nj,j in zip(X_list,num_inducing_list,range(len(X_list))):
                i = np.random.permutation(Xj.shape[0])[:nj]
                z_index.append(np.repeat(j,nj)[:,None])
                Z_list.append(Xj[i].copy())
        else:
            assert len(Z_list) == len(X_list)
            for Zj,Xj,j in zip(Z_list,X_list,range(len(Z_list))):
                assert Zj.shape[1] == Xj.shape[1]
                z_index.append(np.repeat(j,Zj.shape[0])[:,None])

        Z = np.hstack([np.vstack(Z_list),np.vstack(z_index)])


        self.multioutput = True
        SparseGP.__init__(self, X, likelihood, mkernel, Z=Z, normalize_X=normalize_X, X_variance=X_variance)
        self.constrain_fixed('.*iip_\d+_1')
        self.ensure_default_constraints()
