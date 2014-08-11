from _src.kern import Kern
from _src.rbf import RBF
from _src.linear import Linear, LinearFull
from _src.static import Bias, White
from _src.brownian import Brownian
from _src.stationary import Exponential, OU, Matern32, Matern52, ExpQuad, RatQuad, Cosine
from _src.mlp import MLP
from _src.periodic import PeriodicExponential, PeriodicMatern32, PeriodicMatern52
from _src.independent_outputs import IndependentOutputs, Hierarchical
from _src.coregionalize import Coregionalize
from _src.ODE_UY import ODE_UY
from _src.ODE_UYC import ODE_UYC
from _src.ODE_st import ODE_st
from _src.ODE_t import ODE_t
from _src.poly import Poly

from _src.trunclinear import TruncLinear,TruncLinear_inf
from _src.splitKern import SplitKern,DiffGenomeKern

# TODO: put this in an init file somewhere
#I'm commenting this out because the files were not added. JH. Remember to add the files before commiting
try:
    import sympy as sym
    sympy_available=True
except ImportError:
    sympy_available=False

if sympy_available:
    from _src.symbolic import Symbolic
    from _src.eq import Eq
    from _src.heat_eqinit import Heat_eqinit
    #from _src.ode1_eq_lfm import Ode1_eq_lfm

