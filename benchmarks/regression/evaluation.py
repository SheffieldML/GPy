# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import abc
import numpy as np

class Evaluation(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def evaluate(self, gt, pred):
        """Compute a scalar for access the performance"""
        return None

class RMSE(Evaluation):
    "Rooted Mean Square Error"
    name = 'RMSE'
    
    def evaluate(self, gt, pred):
        return np.sqrt(np.square(gt-pred).astype(np.float).mean())
    
