# Copyright (c) 2012-2014, Max Zwiessele
# Licensed under the BSD 3-clause license (see LICENSE.txt)

class StochasticStorage(object):
    '''
    This is a container for holding the stochastic parameters,
    such as subset indices or step length and so on.
    
    self.d has to be a list of lists:
    [dimension indices, nan indices for those dimensions]
    so that the minibatches can be used as efficiently as possible.10
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
        time. We will try to get batches which look the same together
        which speeds up calculations significantly.
        """
        import numpy as np
        self.Y = model.Y_normalized
        bdict = {}
        for d in range(self.Y.shape[1]):
            inan = np.isnan(self.Y[:, d])
            arr_str = np.array2string(inan, 
                                      np.inf, 0, 
                                      True, '', 
                                      formatter={'bool':lambda x: '1' if x else '0'})
            try:
                bdict[arr_str][0].append(d)
            except:
                bdict[arr_str] = [[d], ~inan]
        self.d = bdict.values()

class SparseGPStochastics(StochasticStorage):
    """
    For the sparse gp we need to store the dimension we are in,
    and the indices corresponding to those
    """
    def __init__(self, model, batchsize=1):
        self.batchsize = batchsize
        self.output_dim = model.Y.shape[1]
        self.Y = model.Y_normalized
        self.reset()
        self.do_stochastics()

    def do_stochastics(self):
        if self.batchsize == 1:
            self.current_dim = (self.current_dim+1)%self.output_dim
            self.d = [[[self.current_dim], np.isnan(self.Y[:, self.d])]]
        else:
            import numpy as np
            self.d = np.random.choice(self.output_dim, size=self.batchsize, replace=False)
            bdict = {}
            for d in self.d:
                inan = np.isnan(self.Y[:, d])
                arr_str = int(np.array2string(inan, 
                                          np.inf, 0, 
                                          True, '', 
                                          formatter={'bool':lambda x: '1' if x else '0'}), 2)
                try:
                    bdict[arr_str][0].append(d)
                except:
                    bdict[arr_str] = [[d], ~inan]
            self.d = bdict.values()

    def reset(self):
        self.current_dim = -1
        self.d = None
