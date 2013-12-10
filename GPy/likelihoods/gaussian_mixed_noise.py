# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from scipy import stats
from ..util.linalg import pdinv,mdot,jitchol,chol_inv,DSYR,tdot,dtrtrs
from likelihood import likelihood
from . import Gaussian


class Gaussian_Mixed_Noise(likelihood):
    """
    Gaussian Likelihood for multiple outputs

    This is a wrapper around likelihood.Gaussian class

    :param data_list: data observations
    :type data_list: list of numpy arrays (num_data_output_i x 1), one array per output
    :param noise_params: noise parameters of each output
    :type noise_params: list of floats, one per output
    :param normalize:  whether to normalize the data before computing (predictions will be in original scales)
    :type normalize: False|True
    """
    def __init__(self, data_list, noise_params=None, normalize=True):
        self.num_params = len(data_list)
        self.n_list = [data.size for data in data_list]
        self.index = np.vstack([np.repeat(i,n)[:,None] for i,n in zip(range(self.num_params),self.n_list)])

        if noise_params is None:
            noise_params = [1.] * self.num_params
        else:
            assert self.num_params == len(noise_params), 'Number of noise parameters does not match the number of noise models.'

        self.noise_model_list = [Gaussian(Y,variance=v,normalize = normalize) for Y,v in zip(data_list,noise_params)]
        self.n_params = [noise_model._get_params().size for noise_model in self.noise_model_list]
        self.data = np.vstack(data_list)
        self.N, self.output_dim = self.data.shape
        self._offset = np.zeros((1, self.output_dim))
        self._scale = np.ones((1, self.output_dim))

        self.is_heteroscedastic = True
        self.Z = 0. # a correction factor which accounts for the approximation made

        self.set_data(data_list)
        self._set_params(np.asarray(noise_params))

        super(Gaussian_Mixed_Noise, self).__init__()

    def set_data(self, data_list):
        self.data = np.vstack(data_list)
        self.N, D = self.data.shape
        assert D == self.output_dim
        self.Y = (self.data - self._offset) / self._scale
        if D > self.N:
            raise NotImplementedError
            #self.YYT = np.dot(self.Y, self.Y.T)
            #self.trYYT = np.trace(self.YYT)
            #self.YYT_factor = jitchol(self.YYT)
        else:
            self.YYT = None
            self.trYYT = np.sum(np.square(self.Y))
            self.YYT_factor = self.Y

    def predictive_values(self,mu,var,full_cov,noise_model):
        """
        Predicts the output given the GP

        :param mu: GP's mean
        :param var: GP's variance
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: False|True
        :param noise_model: noise model to use
        :type noise_model: integer
        """
        if full_cov:
            raise NotImplementedError, "Cannot make correlated predictions with an EP likelihood"
        return self.noise_model_list[noise_model].predictive_values(mu,var,full_cov)

    def _get_params(self):
        return np.hstack([noise_model._get_params().flatten() for noise_model in self.noise_model_list])

    def _get_param_names(self):
        if len(self.noise_model_list) == 1:
            names = self.noise_model_list[0]._get_param_names()
        else:
            names = []
            for noise_model,i in zip(self.noise_model_list,range(len(self.n_list))):
                names.append(''.join(noise_model._get_param_names() + ['_%s' %i]))
        return names

    def _set_params(self,p):
        cs_params = np.cumsum([0]+self.n_params)

        for i in range(len(self.n_params)):
            self.noise_model_list[i]._set_params(p[cs_params[i]:cs_params[i+1]])
        self.precision = np.hstack([np.repeat(noise_model.precision,n) for noise_model,n in zip(self.noise_model_list,self.n_list)])[:,None]

        self.V = self.precision * self.Y
        self.VVT_factor = self.precision * self.YYT_factor
        self.covariance_matrix = np.eye(self.N) * 1./self.precision

    def _gradients(self,partial):
        gradients = []
        aux = np.cumsum([0]+self.n_list)
        for ai,af,noise_model in zip(aux[:-1],aux[1:],self.noise_model_list):
            gradients += [noise_model._gradients(partial[ai:af])]
        return np.hstack(gradients)
