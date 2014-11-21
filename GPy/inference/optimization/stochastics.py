# Copyright (c) 2012-2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

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

    def reset(self):
        """
        Reset the state of this stochastics generator.
        """

class SparseGPMissing(StochasticStorage):
    def __init__(self, model, batchsize=1):
        """
        Here we want to loop over all dimensions everytime.
        Thus, we can just make sure the loop goes over self.d every
        time.
        """
        self.d = xrange(model.Y_normalized.shape[1])

class SparseGPStochastics(StochasticStorage):
    """
    For the sparse gp we need to store the dimension we are in,
    and the indices corresponding to those
    """
    def __init__(self, model, batchsize=1):
        self.batchsize = batchsize
        self.output_dim = model.Y.shape[1]
        self.reset()
        self.do_stochastics()

    def do_stochastics(self):
        if self.batchsize == 1:
            self.current_dim = (self.current_dim+1)%self.output_dim
            self.d = [self.current_dim]
        else:
            import numpy as np
            self.d = np.random.choice(self.output_dim, size=self.batchsize, replace=False)

    def reset(self):
        self.current_dim = -1
        self.d = None
