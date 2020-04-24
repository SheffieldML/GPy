"""
Introduction
^^^^^^^^^^^^

'Likelihood' in this context is a measure of how well a model *f* predicts a dataset *y*. The importance of likelihoods in Gaussian Processes is in determining the 'best' values of kernel and noise hyperparamters to relate known, observed and unobserved data. The purpose of optimizing a model (e.g. :py:class:`GPy.models.GPRegression`) is to determine the 'best' hyperparameters i.e. those that minimize negative log marginal likelihood.

.. inheritance-diagram:: GPy.likelihoods.likelihood GPy.likelihoods.mixed_noise.MixedNoise
   :top-classes: GPy.core.parameterization.parameterized.Parameterized

Most likelihood classes inherit directly from :py:class:`GPy.likelihoods.likelihood`, although an intermediary class :py:class:`GPy.likelihoods.mixed_noise.MixedNoise` is used by :py:class:`GPy.likelihoods.multioutput_likelihood`.

"""

from .bernoulli import Bernoulli
from .exponential import Exponential
from .gaussian import Gaussian, HeteroscedasticGaussian
from .gamma import Gamma
from .poisson import Poisson
from .student_t import StudentT
from .likelihood import Likelihood
from .mixed_noise import MixedNoise
from .binomial import Binomial
from .weibull import Weibull
from .loglogistic import LogLogistic
from .multioutput_likelihood import MultioutputLikelihood