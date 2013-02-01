import numpy as np

class likelihood:
    """
    The atom for a likelihood class

    This object interfaces the GP and the data. The most basic likelihood
    (Gaussian) inherits directly from this, as does the EP algorithm

    Some things must be defined for this to work properly:
    self.Y : the effective Gaussian target of the GP
    self.N, self.D : Y.shape
    self.covariance_matrix : the effective (noise) covariance of the GP targets
    self.Z : a factor which gets added to the likelihood (0 for a Gaussian, Z_EP for EP)
    self.is_heteroscedastic : enables significant computational savings in GP
    self.precision : a scalar or vector representation of the effective target precision
    self.YYT : (optional) = np.dot(self.Y, self.Y.T) enables computational savings for D>N
    """
    def __init__(self,data):
        raise ValueError, "this class is not to be instantiated"

    def _get_params(self):
        raise NotImplementedError

    def _get_param_names(self):
        raise NotImplementedError

    def _set_params(self,x):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def _gradients(self,partial):
        raise NotImplementedError
