# Check Matthew Rocklin's blog post.
import sympy as sym
import numpy as np
from GPy.kern.src import Kern
from ...core.symbolic import Symbolic_core


class Symbolic(Kern, Symbolic_core):
    """
    """
    def __init__(self, input_dim, k=None, output_dim=1, name='symbolic', parameters=None, active_dims=None, operators=None, func_modules=[]):

        if k is None:
            raise ValueError("You must provide an argument for the covariance function.")

        Kern.__init__(self, input_dim, active_dims, name=name)
        kdiag = k
        self.cacheable = ['X', 'Z']
        Symbolic_core.__init__(self, {'k':k,'kdiag':kdiag}, cacheable=self.cacheable, derivatives = ['X', 'theta'], parameters=parameters, func_modules=func_modules)        
        self.output_dim = output_dim

    def __add__(self,other):
        return spkern(self._sym_k+other._sym_k)

    def _set_expressions(self, expressions):
        """This method is overwritten because we need to modify kdiag by substituting z for x. We do this by calling the parent expression method to extract variables from expressions, then subsitute the z variables that are present with x."""
        Symbolic_core._set_expressions(self, expressions)
        Symbolic_core._set_variables(self, self.cacheable)
        # Substitute z with x to obtain kdiag.
        for x, z in zip(self.variables['X'], self.variables['Z']):
            expressions['kdiag'] = expressions['kdiag'].subs(z, x)
        Symbolic_core._set_expressions(self, expressions)
            
        
    def K(self,X,X2=None):
        if X2 is None:
            return self.eval_function('k', X=X, Z=X)
        else:
            return self.eval_function('k', X=X, Z=X2)


    def Kdiag(self,X):
        d = self.eval_function('kdiag', X=X)
        if not d.shape[0] == X.shape[0]:
            d = np.tile(d, (X.shape[0], 1))
        return d


    def gradients_X(self, dL_dK, X, X2=None):
        #if self._X is None or X.base is not self._X.base or X2 is not None:
        g = self.eval_gradients_X('k', dL_dK, X=X, Z=X2)
        if X2 is None:
            g *= 2
        return g

    def gradients_X_diag(self, dL_dK, X):
        return self.eval_gradients_X('kdiag', dL_dK, X=X)

    def update_gradients_full(self, dL_dK, X, X2=None):
        # Need to extract parameters to local variables first
        if X2 is None:
            # need to double this inside ...
            gradients = self.eval_update_gradients('k', dL_dK, X=X)
        else:
            gradients = self.eval_update_gradients('k', dL_dK, X=X, Z=X2)

        for name, val in gradients:
            setattr(getattr(self, name), 'gradient', val)


    def update_gradients_diag(self, dL_dKdiag, X):
        gradients = self.eval_update_gradients('kdiag', dL_dKdiag, X)
        for name, val in gradients:
            setattr(getattr(self, name), 'gradient', val)

