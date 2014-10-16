'''
Created on 9 Oct 2014

@author: maxz
'''

class StochasticStorage(object):
    '''
    This is a container for holding the stochastic parameters,
    such as subset indices or step length and so on.
    '''
    def __init__(self, model):
        """
        Initialize this stochastic container using the given model
        """

    def do_stochastics(self):
        """
        Update the internal state to the next batch of the stochastic
        descent algorithm.
        """
        pass

class SparseGPMissing(StochasticStorage):
    def __init__(self, model, batchsize=1):
        self.d = xrange(model.Y_normalized.shape[1])

class SparseGPStochastics(StochasticStorage):
    """
    For the sparse gp we need to store the dimension we are in,
    and the indices corresponding to those
    """
    def __init__(self, model, batchsize=1):
        import itertools
        self.batchsize = batchsize
        if self.batchsize == 1:
            self.dimensions = itertools.cycle(range(model.Y_normalized.shape[1]))
        else:
            import numpy as np
            self.dimensions = lambda: np.random.choice(model.Y_normalized.shape[1], size=batchsize, replace=False)
        self.d = None
        self.do_stochastics()

    def do_stochastics(self):
        if self.batchsize == 1:
            self.d = [self.dimensions.next()]
        else:
            self.d = self.dimensions()