from .kern import CombinationKernel
import numpy as np
from paramz.caching import Cache_this

# A thin wrapper around the base kernel to tell that we are dealing with a partial derivative of a Kernel
class DiffKern(CombinationKernel):
    def __init__(self, base_kern, dimension):
        super(DiffKern, self).__init__([base_kern], 'DiffKern')
        self.base_kern = base_kern
        self.dimension = dimension
        self._gradient_array_ = self.base_kern._gradient_array_
        self.gradient = self.base_kern.gradient

    def parameters_changed(self):
        self.base_kern.parameters_changed()

    @Cache_this(limit=3, ignore_args=())
    def K(self, X, X2=None, dimX2 = None): #X in dimension self.dimension
        if X2 is None:
            X2 = X
        if dimX2 is None:
            dimX2 = self.dimension
        return self.base_kern.dK2_dXdX2(X,X2, self.dimension, dimX2)
 
    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self, X):
        return np.diag(self.base_kern.dK2_dXdX2(X,X, self.dimension, self.dimension))
    
    @Cache_this(limit=3, ignore_args=())
    def dK_dX_wrap(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX(X,X2, self.dimension)

    @Cache_this(limit=3, ignore_args=())
    def dK_dX2_wrap(self, X, X2): #X in dimension self.dimension
        return self.base_kern.dK_dX2(X,X2, self.dimension)

    def reset_gradients(self):
        self.base_kern.reset_gradients()

    def get_gradient(self):
        return self.base_kern.gradient.copy()

    def append_gradient(self, gradient):
        self.base_kern.gradient += gradient

    def update_gradients_full(self, dL_dK, X, X2=None, dimX2=None):
        if dimX2 is None:
            dimX2 = self.dimension
        gradients = self.base_kern.dgradients2_dXdX2(X,X2,self.dimension,dimX2)
        self.base_kern.update_gradients_direct(*[self._convert_gradients(dL_dK, gradient) for gradient in gradients])

    def update_gradients_diag(self, dL_dK_diag, X):
        gradients = self.base_kern.dgradients2_dXdX2(X,X, self.dimension, self.dimension)
        self.base_kern.update_gradients_direct(*[self._convert_gradients(dL_dK_diag, gradient, f=np.diag) for gradient in gradients])

    def update_gradients_dK_dX(self, dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        gradients = self.base_kern.dgradients_dX(X,X2, self.dimension)
        self.base_kern.append_gradients_direct(*[self._convert_gradients(dL_dK, gradient) for gradient in gradients])

    def update_gradients_dK_dX2(self, dL_dK, X, X2=None):
        gradients = self.base_kern.dgradients_dX2(X,X2, self.dimension)
        self.base_kern.append_gradients_direct(*[self._convert_gradients(dL_dK, gradient) for gradient in gradients])

    def _convert_gradients(self, l,g, f = lambda x:x):
        if type(g) is np.ndarray:
            return np.sum(f(l)*f(g))
        else:
            return np.array([np.sum(f(l)*f(gi)) for gi in g])