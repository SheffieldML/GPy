# Copyright (c) 2013, Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core import GP
from .. import likelihoods
from .. import kern

class GPMultioutputRegression(GP):
    """
    Multiple output Gaussian process with Gaussian noise

    This is a wrapper around the models.GP class, with a set of sensible defaults

    :param X_list: input observations
    :type X_list: list of numpy arrays (num_data_output_i x input_dim), one array per output
    :param Y_list: observed values
    :type Y_list: list of numpy arrays (num_data_output_i x 1), one array per output
    :param kernel_list: GPy kernels, defaults to rbf
    :type kernel_list: list of GPy kernels
    :param noise_variance_list: noise parameters per output, defaults to 1.0 for every output
    :type noise_variance_list: list of floats
    :param normalize_X:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_X: False|True
    :param normalize_Y:  whether to normalize the input data before computing (predictions will be in original scales)
    :type normalize_Y: False|True
    :param rank: number tuples of the corregionalization parameters 'coregion_W' (see coregionalize kernel documentation)
    :type rank: integer
    """

    def __init__(self,X_list,Y_list,kernel_list=None,noise_variance_list=None,normalize_X=False,normalize_Y=False,rank=1):

        self.output_dim = len(Y_list)
        assert len(X_list) == self.output_dim, 'Number of outputs do not match length of inputs list.'

        #Inputs indexing
        i = 0
        index = []
        for x,y in zip(X_list,Y_list):
            assert x.shape[0] == y.shape[0]
            index.append(np.repeat(i,x.size)[:,None])
            i += 1
        index = np.vstack(index)
        X = np.hstack([np.vstack(X_list),index])
        original_dim = X.shape[1] - 1

        #Mixed noise likelihood definition
        likelihood = likelihoods.Gaussian_Mixed_Noise(Y_list,noise_params=noise_variance_list,normalize=normalize_Y)

        #Coregionalization kernel definition
        if kernel_list is None:
            kernel_list = [kern.rbf(original_dim)]
        mkernel = kern.build_lcm(input_dim=original_dim, output_dim=self.output_dim, kernel_list = kernel_list, rank=rank)

        self.multioutput = True
        GP.__init__(self, X, likelihood, mkernel, normalize_X=normalize_X)
        self.ensure_default_constraints()

    def _add_output_index(self,X,output):
        """
        In a multioutput model, appends an index column to X to specify the output it is related to.

        :param X: Input data
        :type X: np.ndarray, N x self.input_dim
        :param output: output X is related to
        :type output: integer in {0,..., output_dim-1}

        .. Note:: For multiple non-independent outputs models only.
        """

        assert hasattr(self,'multioutput'), 'This function is for multiple output models only.'

        index = np.ones((X.shape[0],1))*output
        return np.hstack((X,index))

    def plot_single_output(self, X, output):
        """
        A simple wrapper around self.plot, with appropriate setting of the fixed_inputs argument
        """
        raise NotImplementedError

    def _raw_predict_single_output(self, _Xnew, output, which_parts='all', full_cov=False,stop=False):
        """
        For a specific output, calls _raw_predict() at the new point(s) _Xnew.
        This functions calls _add_output_index(), so _Xnew should not have an index column specifying the output.
        ---------

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param output: output to predict
        :type output: integer in {0,..., output_dim-1}
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal

        .. Note:: For multiple non-independent outputs models only.
        """
        _Xnew = self._add_output_index(_Xnew, output)
        return self._raw_predict(_Xnew, which_parts=which_parts,full_cov=full_cov, stop=stop)

    def predict_single_output(self, Xnew,output=0, which_parts='all', full_cov=False, likelihood_args=dict()):
        """
        For a specific output, calls predict() at the new point(s) Xnew.
        This functions calls _add_output_index(), so Xnew should not have an index column specifying the output.

        :param Xnew: The points at which to make a prediction
        :type Xnew: np.ndarray, Nnew x self.input_dim
        :param which_parts:  specifies which outputs kernel(s) to use in prediction
        :type which_parts: ('all', list of bools)
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: bool
        :returns: mean: posterior mean,  a Numpy array, Nnew x self.input_dim
        :returns: var: posterior variance, a Numpy array, Nnew x 1 if full_cov=False, Nnew x Nnew otherwise
        :returns: lower and upper boundaries of the 95% confidence intervals, Numpy arrays,  Nnew x self.input_dim

        .. Note:: For multiple non-independent outputs models only.
        """
        Xnew = self._add_output_index(Xnew, output)
        return self.predict(Xnew, which_parts=which_parts, full_cov=full_cov, likelihood_args=likelihood_args)

    def plot_single_output_f(self, output=None, samples=0, plot_limits=None, which_data='all', which_parts='all', resolution=None, full_cov=False, fignum=None, ax=None):
        """
        For a specific output, in a multioutput model, this function works just as plot_f on single output models.

        :param output: which output to plot (for multiple output models only)
        :type output: integer (first output is 0)
        :param samples: the number of a posteriori samples to plot
        :param plot_limits: The limits of the plot. If 1D [xmin,xmax], if 2D [[xmin,ymin],[xmax,ymax]]. Defaluts to data limits
        :param which_data: which if the training data to plot (default all)
        :type which_data: 'all' or a slice object to slice self.X, self.Y
        :param which_parts: which of the kernel functions to plot (additively)
        :type which_parts: 'all', or list of bools
        :param resolution: the number of intervals to sample the GP on. Defaults to 200 in 1D and 50 (a 50x50 grid) in 2D
        :type resolution: int
        :param full_cov:
        :type full_cov: bool
                :param fignum: figure to plot on.
        :type fignum: figure number
        :param ax: axes to plot on.
        :type ax: axes handle
        """
        assert output is not None, "An output must be specified."
        assert len(self.likelihood.noise_model_list) > output, "The model has only %s outputs." %(self.output_dim + 1)

        if which_data == 'all':
            which_data = slice(None)

        if ax is None:
            fig = pb.figure(num=fignum)
            ax = fig.add_subplot(111)

        if self.X.shape[1] == 2:
            Xu = self.X[self.X[:,-1]==output ,0:1]
            Xnew, xmin, xmax = x_frame1D(Xu, plot_limits=plot_limits)
            Xnew_indexed = self._add_output_index(Xnew,output)

            m, v = self._raw_predict(Xnew_indexed, which_parts=which_parts)

            if samples:
                Ysim = self.posterior_samples_f(Xnew_indexed, samples, which_parts=which_parts, full_cov=True)
                for yi in Ysim.T:
                    ax.plot(Xnew, yi[:,None], Tango.colorsHex['darkBlue'], linewidth=0.25)

            gpplot(Xnew, m, m - 2 * np.sqrt(v), m + 2 * np.sqrt(v), axes=ax)
            ax.plot(Xu[which_data], self.likelihood.Y[self.likelihood.index==output][:,None], 'kx', mew=1.5)
            ax.set_xlim(xmin, xmax)
            ymin, ymax = min(np.append(self.likelihood.Y, m - 2 * np.sqrt(np.diag(v)[:, None]))), max(np.append(self.likelihood.Y, m + 2 * np.sqrt(np.diag(v)[:, None])))
            ymin, ymax = ymin - 0.1 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)
            ax.set_ylim(ymin, ymax)


