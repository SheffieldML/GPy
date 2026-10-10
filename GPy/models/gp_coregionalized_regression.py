# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern
from .. import util

class GPCoregionalizedRegression(GP):
    """
    Gaussian Process model for heteroscedastic multioutput regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param name: model name
    :type name: string
    :param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
    :type W_rank: integer
    :param kernel_name: name of the kernel
    :type kernel_name: string
    :param normalizer: normalize ``Y``. ``False`` disables; ``True`` uses
        ``Standardize``. Note this standardizes the stacked multi-output ``Y``
        as a single column (all outputs together).
    """
    def __init__(self, X_list, Y_list, kernel=None, likelihoods_list=None, name='GPCR',W_rank=1,kernel_name='coreg', normalizer=False):

        #Input and Output
        X,Y,self.output_index = util.multioutput.build_XY(X_list,Y_list)
        Ny = len(Y_list)

        #Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1]-1)
            
            kernel = util.multioutput.ICM(input_dim=X.shape[1]-1, num_outputs=Ny, kernel=kernel, W_rank=W_rank,name=kernel_name)

        #Likelihood
        likelihood = util.multioutput.build_likelihood(Y_list,self.output_index,likelihoods_list)

        super(GPCoregionalizedRegression, self).__init__(
            X,
            Y,
            kernel,
            likelihood,
            Y_metadata={"output_index": self.output_index},
            normalizer=normalizer,
        )

    def set_XY(self, X=None, Y=None):
        """
        Set the input / output data of the model.

        ``X`` and ``Y`` may be stacked arrays (as stored on the model) or
        lists of per-output arrays, matching the constructor.
        """
        X, Y, self.output_index = util.multioutput.coerce_coregionalized_XY(
            X, Y, self.output_index
        )
        if X is not None and Y is None and X.shape[0] != self.Y.shape[0]:
            raise ValueError(
                "set_XY with X only requires the same number of rows as current Y; "
                "pass Y as well when changing the number of observations"
            )
        if Y is not None and X is None and Y.shape[0] != self.X.shape[0]:
            raise ValueError(
                "set_XY with Y only requires the same number of rows as current X; "
                "pass X as well when changing the number of observations"
            )
        if self.Y_metadata is None:
            self.Y_metadata = {}
        self.Y_metadata["output_index"] = self.output_index
        super(GPCoregionalizedRegression, self).set_XY(X=X, Y=Y)
