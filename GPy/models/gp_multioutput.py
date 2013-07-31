# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern


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

    def __init__(self,X_list,Y_list,noise_list=[],kernel_list=None,normalize_X=False,normalize_Y=False,W=1): #TODO W

        assert len(X_list) == len(Y_list)
        index = []
        i = 0
        for x,y in zip(X_list,Y_list):
            assert x.shape[0] == y.shape[0]
            index.append(np.repeat(i,y.size)[:,None])
            i += 1
        index = np.vstack(index)

        if noise_list == []:
            likelihood_list = []
            for Y in Y_list:
                likelihood_list.append(likelihoods.Gaussian(Y,normalize = normalize_Y))

        Y = np.vstack([l_.Y for l_ in likelihood_list])
        likelihood = likelihoods.Gaussian(Y,normalize=False)
        likelihood.index = index

        X = np.hstack([np.vstack(X_list),index])

        if kernel_list is None:
            original_dim = X.shape[1]-1
            kernel_list = [kern.rbf(original_dim) + kern.white(original_dim)]

        mkernel = kernel_list[0].prod(kern.coregionalise(len(X_list),W),tensor=True)
        for k in kernel_list[1:]:
            mkernel += k.prod(kern.coregionalise(len(X_list),W),tensor=True)
        self.multioutput = True
        GP.__init__(self, X, likelihood, mkernel, normalize_X=normalize_X)
        self.ensure_default_constraints()


"""
if likelihood is None:
noise_model_list = []
for Y in Y_list:
noise_model_list.append(likelihoods.Gaussian(Y,normalize = normalize_Y))
#noise_model_list = [likelihoods.gaussian(variance=1.) for Y in Y_list]
#likelihood = likelihoods.EP_Mixed_Noise(Y_list, noise_model_list)

elif Y_list is not None:
if not all(np.vstack(Y_list).flatten() == likelihood.data.flatten()):
raise Warning, 'likelihood.data and Y_list values are different.'

X = np.hstack([np.vstack(X_list),likelihood.index])

if kernel_list is None:
original_dim = X.shape[1]-1
kernel_list = [kern.rbf(original_dim) + kern.white(original_dim)]

mkernel = kernel_list[0].prod(kern.coregionalise(len(X_list),W),tensor=True) #TODO W
for k in kernel_list[1:]:
mkernel += k.prod(kern.coregionalise(len(X_list),W),tensor=True) #TODO W

#kern1 = kern.rbf(1) + kern.white(1)
#kern2 = kern.coregionalise(2,1)
#kern3 = kern1.prod(kern2,tensor=True)
"""

