"""
Introduction
^^^^^^^^^^^^



"""

from . import optimization
from . import latent_function_inference
from . import mcmc

import sys
sys.modules['GPy.inference.optimization'] = optimization
sys.modules['GPy.inference.optimization.optimization'] = optimization
