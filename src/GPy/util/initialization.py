'''
Created on 24 Feb 2014

@author: maxz
'''

import numpy as np
from ..util.pca import PCA

def initialize_latent(init, input_dim, Y):
    Xr = np.asfortranarray(np.random.normal(0, 1, (Y.shape[0], input_dim)))
    if 'PCA' in init:
        p = PCA(Y)
        PC = p.project(Y, min(input_dim, Y.shape[1]))
        Xr[:PC.shape[0], :PC.shape[1]] = PC
        var = .1*p.fracs[:input_dim]
    elif init in 'empirical_samples':
        from ..util.linalg import tdot
        from ..util import diag
        YYT = tdot(Y)
        diag.add(YYT, 1e-6)
        EMP = np.asfortranarray(np.random.multivariate_normal(np.zeros(Y.shape[0]), YYT, min(input_dim, Y.shape[1])).T)
        Xr[:EMP.shape[0], :EMP.shape[1]] = EMP
        var = np.random.uniform(0.5, 1.5, input_dim)
    else:
        var = Xr.var(0)

    Xr -= Xr.mean(0)
    Xr /= Xr.std(0)

    return Xr, var/var.max()
