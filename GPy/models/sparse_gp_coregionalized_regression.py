# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core import SparseGP
from ..inference.latent_function_inference import VarDTC
from .. import kern
from .. import util


class SparseGPCoregionalizedRegression(SparseGP):
    """
    Sparse Gaussian Process model for heteroscedastic multioutput regression

    This is a thin wrapper around the SparseGP class, with a set of sensible defaults

    :param X_list: list of input observations corresponding to each output
    :type X_list: list of numpy arrays
    :param Y_list: list of observed values related to the different noise models
    :type Y_list: list of numpy arrays
    :param Z_list: list of inducing inputs (optional)
    :type Z_list: empty list | list of numpy arrays
    :param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
    :type kernel: None | GPy.kernel defaults
    :likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
    :type likelihoods_list: None | a list GPy.likelihoods
    :param num_inducing: number of inducing inputs, defaults to 10 per output (ignored if Z_list is not empty)
    :type num_inducing: integer | list of integers

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

    def __init__(
        self,
        X_list,
        Y_list,
        Z_list=[],
        kernel=None,
        likelihoods_list=None,
        num_inducing=10,
        X_variance=None,
        name="SGPCR",
        W_rank=1,
        kernel_name="coreg",
        normalizer=False,
    ):
        # Input and Output
        X, Y, self.output_index = util.multioutput.build_XY(X_list, Y_list)
        Ny = len(Y_list)

        # Kernel
        if kernel is None:
            kernel = kern.RBF(X.shape[1] - 1)

            kernel = util.multioutput.ICM(
                input_dim=X.shape[1] - 1,
                num_outputs=Ny,
                kernel=kernel,
                W_rank=W_rank,
                name=kernel_name,
            )

        # Likelihood
        likelihood = util.multioutput.build_likelihood(
            Y_list, self.output_index, likelihoods_list
        )

        # Inducing inputs list
        if len(Z_list):
            assert (
                len(Z_list) == Ny
            ), "Number of outputs do not match length of inducing inputs list."
        else:
            if isinstance(num_inducing, int):
                num_inducing = [num_inducing] * Ny
            num_inducing = np.asarray(num_inducing)
            assert (
                num_inducing.size == Ny
            ), "Number of outputs do not match length of inducing inputs list."
            for ni, Xi in zip(num_inducing, X_list):
                i = np.random.permutation(Xi.shape[0])[:ni]
                Z_list.append(Xi[i].copy())

        Z, _, Iz = util.multioutput.build_XY(Z_list)

        super(SparseGPCoregionalizedRegression, self).__init__(
            X,
            Y,
            Z,
            kernel,
            likelihood,
            inference_method=VarDTC(),
            Y_metadata={"output_index": self.output_index},
            normalizer=normalizer,
        )
        self[".*inducing"][:, -1].fix()

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
        super(SparseGPCoregionalizedRegression, self).set_XY(X=X, Y=Y)

    def predict(
        self,
        Xnew,
        full_cov=False,
        Y_metadata=None,
        kern=None,
        likelihood=None,
        include_likelihood=True,
    ):
        """``Xnew`` may be stacked or a list of per-output arrays (#1099)."""
        Xnew, Y_metadata = util.multioutput.prepare_Xnew(Xnew, Y_metadata)
        return super(SparseGPCoregionalizedRegression, self).predict(
            Xnew,
            full_cov=full_cov,
            Y_metadata=Y_metadata,
            kern=kern,
            likelihood=likelihood,
            include_likelihood=include_likelihood,
        )

    def predict_noiseless(self, Xnew, full_cov=False, Y_metadata=None, kern=None):
        Xnew, Y_metadata = util.multioutput.prepare_Xnew(Xnew, Y_metadata)
        return super(SparseGPCoregionalizedRegression, self).predict_noiseless(
            Xnew, full_cov=full_cov, Y_metadata=Y_metadata, kern=kern
        )

    def predict_quantiles(
        self, X, quantiles=(2.5, 97.5), Y_metadata=None, kern=None, likelihood=None
    ):
        X, Y_metadata = util.multioutput.prepare_Xnew(X, Y_metadata)
        return super(SparseGPCoregionalizedRegression, self).predict_quantiles(
            X, quantiles=quantiles, Y_metadata=Y_metadata, kern=kern, likelihood=likelihood
        )
