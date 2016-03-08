from paramz.optimization import Optimizer
from . import stochastics

from paramz.optimization import *
import sys

sys.modules['GPy.inference.optimization.stochastics'] = stochastics
sys.modules['GPy.inference.optimization.Optimizer'] = Optimizer
