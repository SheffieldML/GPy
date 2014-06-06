# Copyright (c) 2013, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# Construct covariance functions from matlab saves.
import numpy as np
from kern import kern
import parts
import scipy.io

def read_matlab(mat_data)
    mat_data = scipy.io.loadmat(os.path.join(data_path, data_set, 'frey_rawface.mat'))
    if mat_data['type']=='cmpnd':
        # cmpnd kernel
        types = []
        for i in range(mat_data['comp'][0][0]):
            types.append(mat_data['comp'][0][0][i]
        
