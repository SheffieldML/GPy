# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPMultioutputRegression(GP):
    """
    Multiple output Gaussian process with Gaussian noise

    This is a wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: input observations
    :type X_list: list of numpy arrays (num_data_output_i x input_dim), one array per output
    :param Y_list: observed values
    :type Y_list: list of numpy arrays (num_data_output_i x 1), one array per output
    :param kernel_list: GPy kernels, defaults to rbf
    :type kernel_list: list of GPy kernels
    :param noise_variance_list: noise parameters per output, defaults to 1.0 for every output
    :type noise_variance_list: list of floats
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param rank: number tuples of the corregionalization parameters 'coregion_W' (see coregionalize kernel documentation)
    :type rank: integer
    """

    def __init__(self,X_list,Y_list,kernel_list=None,noise_variance_list=None,normalize_X=False,normalize_Y=False,rank=1):

        self.output_dim = len(Y_list)
        assert len(X_list) == self.output_dim, 'Number of outputs do not match length of inputs list.'

        #Inputs indexing
        i = 0
        index = []
        for x,y in zip(X_list,Y_list):
            assert x.shape[0] == y.shape[0]
            index.append(np.repeat(i,x.size)[:,None])
            i += 1
        index = np.vstack(index)
        X = np.hstack([np.vstack(X_list),index])
        original_dim = X.shape[1] - 1

        #Mixed noise likelihood definition
        likelihood = likelihoods.Gaussian_Mixed_Noise(Y_list,noise_params=noise_variance_list,normalize=normalize_Y)

        #Coregionalization kernel definition
        if kernel_list is None:
            kernel_list = [kern.rbf(original_dim)]
        mkernel = kern.build_lcm(input_dim=original_dim, output_dim=self.output_dim, kernel_list = kernel_list, rank=rank)

        self.multioutput = True
        GP.__init__(self, X, likelihood, mkernel, normalize_X=normalize_X)
        self.ensure_default_constraints()
