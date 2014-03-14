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

from exact_gaussian_inference import ExactGaussianInference
from laplace import Laplace
from GPy.inference.latent_function_inference.var_dtc import VarDTC
from expectation_propagation import EP
from dtc import DTC
from fitc import FITC

# class FullLatentFunctionData(object):
#
#
# class LatentFunctionInference(object):
#     def inference(self, kern, X, likelihood, Y, Y_metadata=None):
#         """
#         Do inference on the latent functions given a covariance function `kern`,
#         inputs and outputs `X` and `Y`, and a likelihood `likelihood`.
#         Additional metadata for the outputs `Y` can be given in `Y_metadata`.
#         """
#         raise NotImplementedError, "Abstract base class for full inference"
