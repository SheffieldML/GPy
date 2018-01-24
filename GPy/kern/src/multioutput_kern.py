from .kern import Kern, CombinationKernel
import numpy as np
from functools import reduce, partial
from .independent_outputs import index_to_slices
from paramz.caching import Cache_this

class ZeroKern(Kern):
    def __init__(self):
        super(ZeroKern, self).__init__(1, None, name='ZeroKern',useGPU=False)

    def K(self, X ,X2=None):
        if X2 is None:
            X2 = X
        return np.zeros((X.shape[0],X2.shape[0]))
    
    def update_gradients_full(self,dL_dK, X, X2=None):
        return np.zeros(dL_dK.shape)
    
    def gradients_X(self,dL_dK, X, X2=None):
        return np.zeros((X.shape[0],X.shape[1]))
        
class MultioutputKern(CombinationKernel):
    """
    Multioutput kernel is a meta class for combining different kernels for multioutput GPs. 

    As an example let us have inputs x1 for output 1 with covariance k1 and x2 for output 2 with covariance k2.
    In addition, we need to define the cross covariances k12(x1,x2) and k21(x2,x1). Then the kernel becomes:
    k([x1,x2],[x1,x2]) = [k1(x1,x1) k12(x1, x2); k21(x2, x1), k2(x2,x2)]
    
    For  the kernel, the kernels of outputs are given as list in param "kernels" and cross covariances are
    given in param "cross_covariances" as a dictionary of tuples (i,j) as keys. If no cross covariance is given,
    it defaults to zero, as in k12(x1,x2)=0.
    
    In the cross covariance dictionary, the value needs to be a struct with elements 
    -'kernel': a member of Kernel class that stores the hyper parameters to be updated when optimizing the GP
    -'K': function defining the cross covariance
    -'update_gradients_full': a function to be used for updating gradients
    -'gradients_X': gives a gradient of the cross covariance with respect to the first input
    """
    def __init__(self, kernels, cross_covariances={}, name='MultioutputKern'):
        #kernels contains a list of kernels as input, 
        if not isinstance(kernels, list):
            self.single_kern = True
            self.kern = kernels
            kernels = [kernels]
        else:
            self.single_kern = False
            self.kern = kernels
            
        # The combination kernel ALLWAYS puts the extra dimension last.
        # Thus, the index dimension of this kernel is always the last dimension
        # after slicing. This is why the index_dim is just the last column:
        self.index_dim = -1
        
        super(MultioutputKern, self).__init__(kernels=kernels, extra_dims=[self.index_dim], name=name, link_parameters=False)

        nl = len(kernels)
        #build covariance structure
        covariance = [[None for i in range(nl)] for j in range(nl)]
        linked = []
        for i in range(0,nl):
            unique=True
            for j in range(0,nl):
                if i==j or (kernels[i] is kernels[j]):
                    covariance[i][j] = {'kern': kernels[i], 'K': kernels[i].K, 'update_gradients_full': kernels[i].update_gradients_full, 'gradients_X': kernels[i].gradients_X}
                    if i>j:
                        unique=False
                elif cross_covariances.get((i,j)) is not None: #cross covariance is given
                    covariance[i][j] = cross_covariances.get((i,j))
                else: # zero covariance structure
                    kern = ZeroKern()
                    covariance[i][j] = {'kern': kern, 'K': kern.K, 'update_gradients_full': kern.update_gradients_full, 'gradients_X': kern.gradients_X}       
            if unique is True:
                linked.append(i)
        self.covariance = covariance
        self.link_parameters(*[kernels[i] for i in linked])
        
    @Cache_this(limit=3, ignore_args=())
    def K(self, X ,X2=None):
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:,self.index_dim])
        slices2 = index_to_slices(X2[:,self.index_dim])
        target =  np.zeros((X.shape[0], X2.shape[0]))
        [[[[ target.__setitem__((slices[i][k],slices2[j][l]), self.covariance[i][j]['K'](X[slices[i][k],:],X2[slices2[j][l],:])) for k in range( len(slices[i]))] for l in range(len(slices2[j])) ] for i in range(len(slices))] for j in range(len(slices2))]  
        return target

    @Cache_this(limit=3, ignore_args=())
    def Kdiag(self,X):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        target = np.zeros(X.shape[0])
        [[np.copyto(target[s], kern.Kdiag(X[s])) for s in slices_i] for kern, slices_i in zip(kerns, slices)]
        return target
    
    def _update_gradients_full_wrapper(self, cov_struct, dL_dK, X, X2):
        gradient = cov_struct['kern'].gradient.copy()
        cov_struct['update_gradients_full'](dL_dK, X, X2)
        cov_struct['kern'].gradient += gradient
    
    def _update_gradients_diag_wrapper(self, kern, dL_dKdiag, X):
        gradient = kern.gradient.copy()
        kern.update_gradients_diag(dL_dKdiag, X)
        kern.gradient += gradient
        
    def reset_gradients(self):
        for kern in self.kern: kern.reset_gradients()

    def update_gradients_full(self,dL_dK, X, X2=None):
        self.reset_gradients()
        slices = index_to_slices(X[:,self.index_dim])
        if X2 is not None:
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[[[ self._update_gradients_full_wrapper(self.covariance[i][j], dL_dK[slices[i][k],slices2[j][l]], X[slices[i][k],:], X2[slices2[j][l],:]) for k in range(len(slices[i]))] for l in range(len(slices2[j]))] for i in range(len(slices))] for j in range(len(slices2))]
        else:
            [[[[ self._update_gradients_full_wrapper(self.covariance[i][j], dL_dK[slices[i][k],slices[j][l]], X[slices[i][k],:], X[slices[j][l],:]) for k in range(len(slices[i]))] for l in range(len(slices[j]))] for i in range(len(slices))] for j in range(len(slices))]
            
    def update_gradients_diag(self, dL_dKdiag, X):
        self.reset_gradients()
        slices = index_to_slices(X[:,self.index_dim])
        [[ self._update_gradients_diag_wrapper(self.covariance[i][i]['kern'], dL_dKdiag[slices[i][k]], X[slices[i][k],:]) for k in range(len(slices[i]))] for i in range(len(slices))]
    
    def gradients_X(self,dL_dK, X, X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        target = np.zeros((X.shape[0], X.shape[1]) )
        if X2 is not None:
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[[[ target.__setitem__((slices[i][k]), target[slices[i][k],:] + self.covariance[i][j]['gradients_X'](dL_dK[slices[i][k],slices2[j][l]], X[slices[i][k],:], X2[slices2[j][l],:]) ) for k in range(len(slices[i]))] for l in range(len(slices2[j]))] for i in range(len(slices))] for j in range(len(slices2))]
        else:
            [[[[ target.__setitem__((slices[i][k]), target[slices[i][k],:] + self.covariance[i][j]['gradients_X'](dL_dK[slices[i][k],slices[j][l]], X[slices[i][k],:], (None if (i==j and k==l) else X[slices[j][l],:] )) ) for k in range(len(slices[i]))] for l in range(len(slices[j]))] for i in range(len(slices))] for j in range(len(slices))]
        return target