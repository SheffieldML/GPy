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

The inference methods return a "Posterior" instance, which is a simple
structure which contains a summary of the posterior. The model classes can then
use this posterior object for making predictions, optimizing hyper-parameters,
etc.

"""

from exact_gaussian_inference import ExactGaussianInference
from laplace import LaplaceInference
expectation_propagation = 'foo' # TODO
from dtc import DTC
from fitc import FITC
