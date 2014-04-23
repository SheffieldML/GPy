try:
    import sympy as sym
    sympy_available=True
except ImportError:
    sympy_available=False

import numpy as np
from symbolic import Symbolic

class Heat_eqinit(Symbolic):
    """
    A symbolic covariance based on laying down an initial condition of the heat equation with an exponentiated quadratic covariance. The covariance then has multiple outputs which are interpreted as observations of the diffused process with different diffusion coefficients (or at different times). 

    """
    def __init__(self, input_dim, output_dim=1, param=None, name='Heat_eqinit'):

        x = sym.symbols('x_:' + str(input_dim))
        z = sym.symbols('z_:' + str(input_dim))
        scale = sym.var('scale_i scale_j',positive=True)
        lengthscale = sym.var('lengthscale_i lengthscale_j', positive=True)
        shared_lengthscale = sym.var('shared_lengthscale', positive=True)
        dist_string = ' + '.join(['(x_%i - z_%i)**2' %(i, i) for i in range(input_dim)])
        from sympy.parsing.sympy_parser import parse_expr
        dist = parse_expr(dist_string)

        # this is the covariance function               
        f = scale_i*scale_j*sym.exp(-dist/(2*(shared_lengthscale**2 + lengthscale_i*lengthscale_j)))
        # extra input dim is to signify the output dimension. 
        super(Heat_eqinit, self).__init__(input_dim=input_dim+1, k=f, output_dim=output_dim, name=name)
                                          
