# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from ..util import multioutput


import pylab as pb

class GPMultioutput(GP):
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

    def __init__(self,X_list,Y_list,kernel_list=None,normalize_X=False,normalize_Y=False,W=1,mixed_noise_list=[]): #TODO W
        #TODO: split into 2 models gp_mixed_noise and ep_mixed_noise

        assert len(X_list) == len(Y_list)
        index = []
        i = 0
        for x,y in zip(X_list,Y_list):
            assert x.shape[0] == y.shape[0]
            index.append(np.repeat(i,y.size)[:,None])
            i += 1
        index = np.vstack(index)

        """

        if mixed_noise_list == []:
            for Y in Y_list:
                self.likelihood_list.append(likelihoods.Gaussian(Y,normalize = normalize_Y))

            Y = np.vstack([l_.Y for l_ in self.likelihood_list])
            likelihood = likelihoods.Gaussian(Y,normalize=False)
            likelihood.index = index
        """
        if mixed_noise_list == []:
            likelihood = likelihoods.Gaussian_Mixed_Noise(Y_list,normalize=normalize_Y)
             #TODO: allow passing the variance parameter into the model
        else:
            self.likelihood_list = [] #TODO this is not needed
            assert len(Y_list) == len(mixed_noise_list)
            for noise,Y in zip(mixed_noise_list,Y_list):
                self.likelihood_list.append(likelihoods.EP(Y,noise))
             #TODO: allow normalization
            likelihood = likelihoods.EP_Mixed_Noise(Y_list, mixed_noise_list)

        X = np.hstack([np.vstack(X_list),index])
        original_dim = X.shape[1] - 1

        if kernel_list is None:
            kernel_list = [[kern.rbf(original_dim)],[kern.white(original_dim+1)]]

        mkernel = multioutput.build_cor_kernel(input_dim=original_dim, Nout=len(X_list), CK = kernel_list[0], NC = kernel_list[1], W=1)

        self.multioutput = True
        GP.__init__(self, X, likelihood, mkernel, normalize_X=normalize_X)
        self.ensure_default_constraints()
