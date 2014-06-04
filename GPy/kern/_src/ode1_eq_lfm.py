try:
    import sympy as sym
    sympy_available=True
except ImportError:
    sympy_available=False

import numpy as np
from GPy.util.symbolic import differfln
from symbolic import Symbolic
from sympy import Function, S, oo, I, cos, sin, asin, log, erf, pi, exp, sqrt, sign, gamma, polygamma

class Ode1_eq_lfm(Symbolic):
    """
    A symbolic covariance based on a first order differential equation being driven by a latent force that is an exponentiated quadratic. 

    """
    def __init__(self, output_dim=1, param=None, name='Ode1_eq_lfm'):

        input_dim = 2
        x_0, z_0, decay_i, decay_j, lengthscale = sym.symbols('x_0, z_0, decay_i, decay_j, lengthscale', positive=True)
        scale_i, scale_j = sym.symbols('scale_i, scale_j')
        # note that covariance only valid for positive time.
        
        class sim_h(Function):
            nargs = 5
            @classmethod
            def eval(cls, t, tprime, d_i, d_j, l):
                half_l_di = 0.5*l*d_i
                arg_1 = half_l_di + tprime/l
                arg_2 = half_l_di - (t-tprime)/l
                ln_part_1 = differfln(arg_1, arg_2)
                arg_1 = half_l_di 
                arg_2 = half_l_di - t/l
                ln_part_2 = differfln(half_l_di, half_l_di - t/l)

                
                return (exp(half_l_di*half_l_di
                                     - d_i*(t-tprime)
                                     + ln_part_1
                                     - log(d_i + d_j))
                        - exp(half_l_di*half_l_di
                              - d_i*t - d_j*tprime
                              + ln_part_2
                              - log(d_i + d_j)))
            

        f = scale_i*scale_j*(sim_h(x_0, z_0, decay_i, decay_j, lengthscale)
                             + sim_h(z_0, x_0, decay_j, decay_i, lengthscale))
        # extra input dim is to signify the output dimension. 
        super(Ode1_eq_lfm, self).__init__(input_dim, k=f, output_dim=output_dim, name=name)
        self.lengthscale.constrain_positive()
        self.decay.constrain_positive()
