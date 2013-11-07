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

        #TODO:assert something about the kernel being an AR kernel?


    def set_params(self, x):
        self.kern.set_params(x[:self.kern.num_params_transformed()])
        self.sigma2 = x[-1]

        #get the new model matrices from the kernel

        #run the kalman filter

        #run the rts smoother

    def get_params(self):
        return np.append(self.kern.get_params_transformed(), self.sigma2)

    def get_param_names(self):
        return self.kern._get_param_names_transformed() + ['noise_variance']

    def log_likelihood(self):
        #TODO

    def _log_likelihood_gradients(self):
        #TODO
        dL_dsigma2 = ???
        dL_dtheta = self.kern.dL_dtheta_via_FL(self.dL_dF, self.dL_dL)
        return np.hstack((dL_dtheta, dL_dsigma2))

    def predict_raw(self, Xnew):
        #TODO
        #make a single matrix containing traingin and testing points

        #sort the matrix (save the order

        #run the kalman filter again

        #run the smoother

        #put the data back in the original order, return the posterior of the state

    def predict(self):
        #TODO

        #run the kalman filter to get the state, add the noise variance to the state variance

    def plot(self):
        #TODO

    def posterior_samples_f(self,X,size=10):
        #TODO

    def posterior_samples(self, X, size=10):
        #TODO
