"""
Introduction
^^^^^^^^^^^^



"""

from . import optimization
from . import latent_function_inference
from . import mcmc
from . import hyperparameters

import sys

sys.modules["GPy.inference.optimization"] = optimization
