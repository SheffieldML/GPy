# Copyright (c) 2012, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.sparse_gp_mpi import SparseGP_MPI
from .. import likelihoods
from .. import kern
from ..inference.latent_function_inference import VarDTC
from GPy.core.parameterization.variational import NormalPosterior


class SparseGPRegression(SparseGP_MPI):
    """
    Gaussian Process model for regression

    This is a thin wrapper around the SparseGP class, with a set of sensible defalts

    :param X: input observations
    :param X_variance: input uncertainties, one per input X
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf+white
    :param Z: inducing inputs (optional, see note)
    :type Z: np.ndarray (num_inducing x input_dim) | None
    :param num_inducing: number of inducing points (ignored if Z is passed, see note)
    :type num_inducing: int
    :rtype: model object

    .. Note:: If no Z array is passed, num_inducing (default 10) points are selected from the data. Other wise num_inducing is ignored
    .. Note:: Multiple independent outputs are allowed using columns of Y

    """

    def __init__(
        self,
        X,
        Y,
        kernel=None,
        Z=None,
        num_inducing=10,
        X_variance=None,
        mean_function=None,
        normalizer=None,
        mpi_comm=None,
        name="sparse_gp",
    ):
        num_data, input_dim = X.shape

        # kern defaults to rbf (plus white for stability)
        if kernel is None:
            kernel = kern.RBF(input_dim)  #  + kern.white(input_dim, variance=1e-3)

        # Z defaults to a subset of the data
        if Z is None:
            i = np.random.permutation(num_data)[: min(num_inducing, num_data)]
            Z = X.view(np.ndarray)[i].copy()
        else:
            assert Z.shape[1] == input_dim

        likelihood = likelihoods.Gaussian()

        if not (X_variance is None):
            X = NormalPosterior(X, X_variance)

        if mpi_comm is not None:
            from ..inference.latent_function_inference.var_dtc_parallel import VarDTC_minibatch

            infr = VarDTC_minibatch(mpi_comm=mpi_comm)
        else:
            infr = VarDTC()

        super(SparseGPRegression, self).__init__(
            X,
            Y,
            Z,
            kernel,
            likelihood,
            mean_function=mean_function,
            inference_method=infr,
            normalizer=normalizer,
            mpi_comm=mpi_comm,
            name=name,
        )

    def to_dict(self, save_data=True):
        """
        Convert the object into a json serializable dictionary.

        :param boolean save_data: if true, it adds the training data self.X and self.Y to the dictionary
        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """
        if self.has_uncertain_inputs():
            raise NotImplementedError(
                "Saving a SparseGPRegression with uncertain inputs (X_variance) is not supported."
            )
        input_dict = super(SparseGPRegression, self).to_dict(save_data)
        input_dict["class"] = "GPy.models.SparseGPRegression"
        return input_dict

    @staticmethod
    def _build_from_input_dict(input_dict, data=None):
        from ..core import SparseGP

        input_dict = SparseGP._format_input_dict(input_dict, data)
        m = SparseGPRegression(
            input_dict["X"],
            input_dict["Y"],
            kernel=input_dict["kernel"],
            Z=input_dict["Z"],
            mean_function=input_dict["mean_function"],
            normalizer=input_dict["normalizer"],
            name=input_dict["name"],
        )
        m.likelihood.variance[:] = input_dict["likelihood"].variance.values
        return m

    def parameters_changed(self):
        from ..inference.latent_function_inference.var_dtc_parallel import update_gradients_sparsegp, VarDTC_minibatch

        if isinstance(self.inference_method, VarDTC_minibatch):
            update_gradients_sparsegp(self, mpi_comm=self.mpi_comm)
        else:
            super(SparseGPRegression, self).parameters_changed()
