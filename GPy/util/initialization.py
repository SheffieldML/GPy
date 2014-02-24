'''
Created on 24 Feb 2014

@author: maxz
'''

import numpy as np
from linalg import PCA

def initialize_latent(init, input_dim, Y):
    Xr = np.random.randn(Y.shape[0], input_dim)
    if init == 'PCA':
        PC = PCA(Y, input_dim)[0]
        Xr[:PC.shape[0], :PC.shape[1]] = PC
    else:
        pass
    return Xr