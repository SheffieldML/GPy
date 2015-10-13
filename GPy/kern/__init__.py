"""
Kernel module the kernels to sit in.

.. automodule:: .src
   :members:
   :private-members:
"""
from .src.kern import Kern
from .src.add import Add
from .src.prod import Prod
from .src.rbf import RBF
from .src.linear import Linear, LinearFull
from .src.static import Bias, White, Fixed
from .src.brownian import Brownian
from .src.stationary import Exponential, OU, Matern32, Matern52, ExpQuad, RatQuad, Cosine
from .src.mlp import MLP
from .src.periodic import PeriodicExponential, PeriodicMatern32, PeriodicMatern52
from .src.standard_periodic import StdPeriodic
from .src.independent_outputs import IndependentOutputs, Hierarchical
from .src.coregionalize import Coregionalize
from .src.ODE_UY import ODE_UY
from .src.ODE_UYC import ODE_UYC
from .src.ODE_st import ODE_st
from .src.ODE_t import ODE_t
from .src.poly import Poly
from .src.eq_ode2 import EQ_ODE2
from .src.trunclinear import TruncLinear,TruncLinear_inf
from .src.splitKern import SplitKern,DEtime
from .src.splitKern import DEtime as DiffGenomeKern
from .src.spline import Spline
from .src.basis_funcs import LinearSlopeBasisFuncKernel, BasisFuncKernel, ChangePointBasisFuncKernel, DomainKernel