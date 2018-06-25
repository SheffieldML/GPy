# Copyright (c) 2012-2014, Max Zwiessele, James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

__doc__ = """
Inference over Gaussian process latent functions

In all our GP models, the consistency propery means that we have a Gaussian
prior over a finite set of points f. This prior is

  math:: N(f | 0, K)

where K is the kernel matrix.

We also have a likelihood (see GPy.likelihoods) which defines how the data are
related to the latent function: p(y | f).  If the likelihood is also a Gaussian,
the inference over f is tractable (see exact_gaussian_inference.py).

If the likelihood object is something other than Gaussian, then exact inference
is not tractable. We then resort to a Laplace approximation (laplace.py) or
expectation propagation (ep.py).

The inference methods return a
:class:`~GPy.inference.latent_function_inference.posterior.Posterior`
instance, which is a simple
structure which contains a summary of the posterior. The model classes can then
use this posterior object for making predictions, optimizing hyper-parameters,
etc.

"""

class LatentFunctionInference(object):
    def on_optimization_start(self):
        """
        This function gets called, just before the optimization loop to start.
        """
        pass

    def on_optimization_end(self):
        """
        This function gets called, just after the optimization loop ended.
        """
        pass

    def _save_to_input_dict(self):
        input_dict = {}
        return input_dict

    def to_dict(self):
        raise NotImplementedError

    @staticmethod
    def from_dict(input_dict):
        """
        Instantiate an object of a derived class using the information
        in input_dict (built by the to_dict method of the derived class).
        More specifically, after reading the derived class from input_dict,
        it calls the method _build_from_input_dict of the derived class.
        Note: This method should not be overrided in the derived class. In case
        it is needed, please override _build_from_input_dict instate.

        :param dict input_dict: Dictionary with all the information needed to
           instantiate the object.
        """

        import copy
        input_dict = copy.deepcopy(input_dict)
        inference_class = input_dict.pop('class')
        import GPy
        inference_class = eval(inference_class)
        return inference_class._build_from_input_dict(inference_class, input_dict)

    @staticmethod
    def _build_from_input_dict(inference_class, input_dict):
        return inference_class(**input_dict)

class InferenceMethodList(LatentFunctionInference, list):

    def on_optimization_start(self):
        for inf in self:
            inf.on_optimization_start()

    def on_optimization_end(self):
        for inf in self:
            inf.on_optimization_end()

    def __getstate__(self):
        state = []
        for inf in self:
            state.append(inf)
        return state

    def __setstate__(self, state):
        for inf in state:
            self.append(inf)

from .exact_gaussian_inference import ExactGaussianInference
from .laplace import Laplace,LaplaceBlock
from GPy.inference.latent_function_inference.var_dtc import VarDTC
from .expectation_propagation import EP, EPDTC
from .dtc import DTC
from .fitc import FITC
from .pep import PEP
from .var_dtc_parallel import VarDTC_minibatch
from .var_gauss import VarGauss
from .gaussian_grid_inference import GaussianGridInference
from .vardtc_svi_multiout import VarDTC_SVI_Multiout
from .vardtc_svi_multiout_miss import VarDTC_SVI_Multiout_Miss


# class FullLatentFunctionData(object):
#
#

# class EMLikeLatentFunctionInference(LatentFunctionInference):
#     def update_approximation(self):
#         """
#         This function gets called when the
#         """
#
#     def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
#         """
#         Do inference on the latent functions given a covariance function `kern`,
#         inputs and outputs `X` and `Y`, inducing_inputs `Z`, and a likelihood `likelihood`.
#         Additional metadata for the outputs `Y` can be given in `Y_metadata`.
#         """
#         raise NotImplementedError, "Abstract base class for full inference"
#
# class VariationalLatentFunctionInference(LatentFunctionInference):
#     def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
#         """
#         Do inference on the latent functions given a covariance function `kern`,
#         inputs and outputs `X` and `Y`, inducing_inputs `Z`, and a likelihood `likelihood`.
#         Additional metadata for the outputs `Y` can be given in `Y_metadata`.
#         """
#         raise NotImplementedError, "Abstract base class for full inference"
