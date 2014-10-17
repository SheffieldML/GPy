'''
Created on 24 Feb 2014

@author: maxz
'''

import numpy as np
from GPy.util.pca import PCA

def initialize_latent(init, input_dim, Y):
    Xr = np.asfortranarray(np.random.randn(Y.shape[0], input_dim))
    if init == 'PCA':
        p = PCA(Y)
        PC = p.project(Y, min(input_dim, Y.shape[1]))
        Xr[:PC.shape[0], :PC.shape[1]] = PC
        var = .1*p.fracs[:input_dim]
    else:
        var = Xr.var(0)

    Xr -= Xr.mean(0)
    Xr /= Xr.std(0)

    return Xr, var/var.max()
