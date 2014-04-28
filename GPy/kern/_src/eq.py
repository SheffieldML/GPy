try:
    import sympy as sym
    sympy_available=True
except ImportError:
    sympy_available=False

import numpy as np
from symbolic import Symbolic

class Eq(Symbolic):
    """
    The exponentiated quadratic covariance as a symbolic function. 

    """
    def __init__(self, input_dim, output_dim=1, variance=1.0, lengthscale=1.0, name='Eq'):

        parameters = {'variance' : variance, 'lengthscale' : lengthscale}
        x = sym.symbols('x_:' + str(input_dim))
        z = sym.symbols('z_:' + str(input_dim))
        variance = sym.var('variance',positive=True)
        lengthscale = sym.var('lengthscale', positive=True)
        dist_string = ' + '.join(['(x_%i - z_%i)**2' %(i, i) for i in range(input_dim)])
        from sympy.parsing.sympy_parser import parse_expr
        dist = parse_expr(dist_string)

        # this is the covariance function               
        f = variance*sym.exp(-dist/(2*lengthscale**2))
        # extra input dim is to signify the output dimension. 
        super(Eq, self).__init__(input_dim=input_dim, k=f, output_dim=output_dim, parameters=parameters, name=name)
                                          
