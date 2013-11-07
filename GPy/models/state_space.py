import numpy as np
from ../core import Model

class StateSpace(Model):
    def __init__(self, X, Y, kernel=None):
        self.num_data, input_dim = X.shape
        assert input_dim==1, "State space methods for time only"
        num_data_Y, self.output_dim = Y.shape
        assert num_data_Y == self.num_data, "X and Y data don't match"
        assert self.output_dim == 1, "State space methods for single outputs only"

        self.X = X
        self.Y = Y
        
        self.sigma2 = 1.

        if kernel is None:
            self.kern = kern.Matern32(1)
        else:
            self.kern = kernel

    def set_params(self, x):
        self.kern.set_params(x[:self.kern.num_params_transformed()])
        self.sigma2 = x[-1]

    def get_params(self):
        return np.append(self.kern.get_params_transformed(), self.sigma2)

    def get_param_names(self):
        return self.kern._get_param_names_transformed() + ['noise_variance']

    def log_likelihood(self):
        #TODO

    def log_likelihood_gradients(self):
        #TODO

    def predict_raw(self):
        #TODO

    def predict(self):
        #TODO

    def plot(self):
        #TODO
