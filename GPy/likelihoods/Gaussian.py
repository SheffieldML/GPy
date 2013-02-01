import numpy as np
from likelihood import likelihood

class Gaussian(likelihood):
    def __init__(self,data,variance=1.,normalize=False):
        self.is_heteroscedastic = False
        self.data = data
        self.N,D = data.shape
        self.Z = 0. # a correction factor which accounts for the approximation made

        #normalisation
        if normalize:
            self._mean = data.mean(0)[None,:]
            self._std = data.std(0)[None,:]
            self.Y = (self.data - self._mean)/self._std
        else:
            self._mean = np.zeros((1,D))
            self._std = np.ones((1,D))
            self.Y = self.data

        self.YYT = np.dot(self.Y,self.Y.T)
        self._set_params(np.asarray(variance))


    def _get_params(self):
        return np.asarray(self._variance)

    def _get_param_names(self):
        return ["noise variance"]

    def _set_params(self,x):
        self._variance = x
        self.covariance_matrix = np.eye(self.N)*self._variance
        self.precision = 1./self._variance

    def fit(self):
        """
        No approximations needed
        """
        pass

    def _gradients(self,partial):
        return np.sum(np.diag(partial))
