from .kern import Kern, CombinationKernel
import numpy as np
from functools import reduce, partial
from GPy.util.multioutput import index_to_slices
from paramz.caching import Cache_this

class MultioutputKern(CombinationKernel):
    def __init__(self, kernels, cross_covariances, name='MultioutputKern'):
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
        
        super(MultioutputKern, self).__init__(kernels=kernels, extra_dims=[self.index_dim], name=name, link_params=False)

        nl = len(kernels)
        #build covariance structure
        covariance = [[None for i in range(nl)] for j in range(nl)]
        linked = []
        for i in range(0,nl):
            unique=True
            for j in range(0,nl):
                if i==j or (kernels[i] is kernels[j]):
                    covariance[i][j] = {'K': kernels[i].K, 'update_gradients_full': kernels[i].update_gradients_full, 'gradients_X': kernels[i].gradients_X}
                    if i>j:
                        unique=False
                elif cross_covariances.get((i,j)) is not None: #cross covariance is given
                    covariance[i][j] = cross_covariances.get((i,j))
                else: # zero matrix
                    covariance[i][j] = {'K': lambda x, x2: np.zeros((x.shape[0],x2.shape[0])), 'update_gradients_full': lambda x, x2: np.zeros((x.shape[0],x2.shape[0])), 'gradients_X': lambda x, x2: np.zeros((x.shape[0],x.shape[1]))}       
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

    def reset_gradients(self):
        for kern in self.kern: kern.reset_gradients()

    def update_gradients_full(self,dL_dK,X,X2=None, reset=True):
        if reset:
            self.reset_gradients()
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:,self.index_dim])
        slices2 = index_to_slices(X2[:,self.index_dim])                
        [[[[ self.covariance[i][j]['update_gradients_full'](dL_dK[slices[i][k],slices2[j][l]], X[slices[i][k],:], X2[slices2[j][l],:], False) for k in range(len(slices[i]))] for l in range(len(slices2[j]))] for i in range(len(slices))] for j in range(len(slices2))]
        
    def update_gradients_diag(self, dL_dKdiag, X):
        for kern in self.kerns: kern.reset_gradients()
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        [[ self.kerns[i].update_gradients_diag(dL_dKdiag[slices[i][k]], X[slices[i][k],:], False) for k in range(len(slices[i]))] for i in range(len(slices))]
    
    def gradients_X(self,dL_dK, X, X2=None):
        if X2 is None:
            X2 = X
        slices = index_to_slices(X[:,self.index_dim])
        slices2 = index_to_slices(X2[:,self.index_dim])
        target = np.zeros((X.shape[0], X.shape[1]) )
        [[[[ target.__setitem__((slices[i][k]), target[slices[i][k],:] + self.covariance[i][j]['gradients_X'](dL_dK[slices[i][k],slices2[j][l]], X[slices[i][k],:], X2[slices2[j][l],:]) ) for k in range(len(slices[i]))] for l in range(len(slices2[j]))] for i in range(len(slices))] for j in range(len(slices2))] 
        return target