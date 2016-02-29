# ## Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy
np = numpy

from ..core.parameterization import Param
from GPy.core.model import Model
from ..util.block_matrices import get_blocks, get_block_shapes, unblock, get_blocks_3d, get_block_shapes_3d

def get_shape(x):
    if isinstance(x, numpy.ndarray):
        return x.shape
    return ()

def at_least_one_element(x):
    if isinstance(x, (list, tuple)):
        return x
    return [x]

def flatten_if_needed(x):
    return numpy.atleast_1d(x).flatten()

class GradientChecker(Model):

    def __init__(self, f, df, x0, names=None, *args, **kwargs):
        """
        :param f: Function to check gradient for
        :param df: Gradient of function to check
        :param x0:
            Initial guess for inputs x (if it has a shape (a,b) this will be reflected in the parameter names).
            Can be a list of arrays, if takes a list of arrays. This list will be passed
            to f and df in the same order as given here.
            If only one argument, make sure not to pass a list!!!

        :type x0: [array-like] | array-like | float | int
        :param names:
            Names to print, when performing gradcheck. If a list was passed to x0
            a list of names with the same length is expected.
        :param args: Arguments passed as f(x, *args, **kwargs) and df(x, *args, **kwargs)

        Examples:
        ---------
            from GPy.models import GradientChecker
            N, M, Q = 10, 5, 3

            Sinusoid:

                X = numpy.random.rand(N, Q)
                grad = GradientChecker(numpy.sin,numpy.cos,X,'x')
                grad.checkgrad(verbose=1)

            Using GPy:

                X, Z = numpy.random.randn(N,Q), numpy.random.randn(M,Q)
                kern = GPy.kern.linear(Q, ARD=True) + GPy.kern.rbf(Q, ARD=True)
                grad = GradientChecker(kern.K,
                                       lambda x: 2*kern.dK_dX(numpy.ones((1,1)), x),
                                       x0 = X.copy(),
                                       names='X')
                grad.checkgrad(verbose=1)
                grad.randomize()
                grad.checkgrad(verbose=1)
        """
        super(GradientChecker, self).__init__(name='GradientChecker')
        if isinstance(x0, (list, tuple)) and names is None:
            self.shapes = [get_shape(xi) for xi in x0]
            self.names = ['X{i}'.format(i=i) for i in range(len(x0))]
        elif isinstance(x0, (list, tuple)) and names is not None:
            self.shapes = [get_shape(xi) for xi in x0]
            self.names = names
        elif names is None:
            self.names = ['X']
            self.shapes = [get_shape(x0)]
        else:
            self.names = names
            self.shapes = [get_shape(x0)]

        for name, xi in zip(self.names, at_least_one_element(x0)):
            self.__setattr__(name, Param(name, xi))
            self.link_parameter(self.__getattribute__(name))
#         self._param_names = []
#         for name, shape in zip(self.names, self.shapes):
#             self._param_names.extend(map(lambda nameshape: ('_'.join(nameshape)).strip('_'), itertools.izip(itertools.repeat(name), itertools.imap(lambda t: '_'.join(map(str, t)), itertools.product(*map(lambda xi: range(xi), shape))))))
        self.args = args
        self.kwargs = kwargs
        self.f = f
        self.df = df

    def _get_x(self):
        if len(self.names) > 1:
            return [self.__getattribute__(name) for name in self.names] + list(self.args)
        return [self.__getattribute__(self.names[0])] + list(self.args)

    def log_likelihood(self):
        return float(numpy.sum(self.f(*self._get_x(), **self.kwargs)))

    def _log_likelihood_gradients(self):
        return numpy.atleast_1d(self.df(*self._get_x(), **self.kwargs)).flatten()

    #def _get_params(self):
        #return numpy.atleast_1d(numpy.hstack(map(lambda name: flatten_if_needed(self.__getattribute__(name)), self.names)))

    #def _set_params(self, x):
        #current_index = 0
        #for name, shape in zip(self.names, self.shapes):
            #current_size = numpy.prod(shape)
            #self.__setattr__(name, x[current_index:current_index + current_size].reshape(shape))
            #current_index += current_size

    #def _get_param_names(self):
        #_param_names = []
        #for name, shape in zip(self.names, self.shapes):
            #_param_names.extend(map(lambda nameshape: ('_'.join(nameshape)).strip('_'), itertools.izip(itertools.repeat(name), itertools.imap(lambda t: '_'.join(map(str, t)), itertools.product(*map(lambda xi: range(xi), shape))))))
        #return _param_names


class HessianChecker(GradientChecker):

    def __init__(self, f, df, ddf, x0, names=None, *args, **kwargs):
        """
        :param f: Function (only used for numerical hessian gradient)
        :param df: Gradient of function to check
        :param ddf: Analytical gradient function
        :param x0:
            Initial guess for inputs x (if it has a shape (a,b) this will be reflected in the parameter names).
            Can be a list of arrays, if takes a list of arrays. This list will be passed
            to f and df in the same order as given here.
            If only one argument, make sure not to pass a list!!!

        :type x0: [array-like] | array-like | float | int
        :param names:
            Names to print, when performing gradcheck. If a list was passed to x0
            a list of names with the same length is expected.
        :param args: Arguments passed as f(x, *args, **kwargs) and df(x, *args, **kwargs)

        """
        super(HessianChecker, self).__init__(df, ddf, x0, names=names, *args, **kwargs)
        self._f = f
        self._df = df
        self._ddf = ddf

    def checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3, block_indices=None, plot=False):
        """
        Overwrite checkgrad method to check whole block instead of looping through

        Shows diagnostics using matshow instead

        :param verbose: If True, print a "full" checking of each parameter
        :type verbose: bool
        :param step: The size of the step around which to linearise the objective
        :type step: float (default 1e-6)
        :param tolerance: the tolerance allowed (see note)
        :type tolerance: float (default 1e-3)

        Note:-
           The gradient is considered correct if the ratio of the analytical
           and numerical gradients is within <tolerance> of unity.
        """
        try:
            import numdifftools as nd
        except:
            raise ImportError("Don't have numdifftools package installed, it is not a GPy dependency as of yet, it is only used for hessian tests")

        if target_param:
            raise NotImplementedError('Only basic functionality is provided with this gradchecker')

        #Repeat for each parameter, not the nicest but shouldn't be many cases where there are many
        #variables
        current_index = 0
        for name, shape in zip(self.names, self.shapes):
            current_size = numpy.prod(shape)
            x = self.optimizer_array.copy()
            #x = self._get_params_transformed().copy()
            x = x[current_index:current_index + current_size].reshape(shape)

            # Check gradients
            analytic_hess = self._ddf(x)
            if analytic_hess.shape[1] == 1:
                analytic_hess = numpy.diagflat(analytic_hess)

            #From the docs:
            #x0 : vector location
            #at which to differentiate fun
            #If x0 is an N x M array, then fun is assumed to be a function
            #of N*M variables., thus we must have it flat, not (N,1), but just (N,)
            #numeric_hess_partial = nd.Hessian(self._f, vectorized=False)
            numeric_hess_partial = nd.Jacobian(self._df, vectorized=False)
            #numeric_hess_partial = nd.Derivative(self._df, vectorized=True)
            numeric_hess = numeric_hess_partial(x)

            check_passed = self.checkgrad_block(analytic_hess, numeric_hess, verbose=verbose, step=step, tolerance=tolerance, block_indices=block_indices, plot=plot)
            current_index += current_size
        return check_passed

    def checkgrad_block(self, analytic_hess, numeric_hess, verbose=False, step=1e-6, tolerance=1e-3, block_indices=None, plot=False):
        """
        Checkgrad a block matrix
        """
        if analytic_hess.dtype is np.dtype('object'):
            #Make numeric hessian also into a block matrix
            real_size = get_block_shapes(analytic_hess)
            num_elements = np.sum(real_size)
            if (num_elements, num_elements) == numeric_hess.shape:
                #If the sizes are the same we assume they are the same
                #(we have not fixed any values so the numeric is the whole hessian)
                numeric_hess = get_blocks(numeric_hess, real_size)
            else:
                #Make a fake empty matrix and fill out the correct block
                tmp_numeric_hess = get_blocks(np.zeros((num_elements, num_elements)), real_size)
                tmp_numeric_hess[block_indices] = numeric_hess.copy()
                numeric_hess = tmp_numeric_hess

        if block_indices is not None:
            #Extract the right block
            analytic_hess = analytic_hess[block_indices]
            numeric_hess = numeric_hess[block_indices]
        else:
            #Unblock them if they are in blocks and you aren't checking a single block (checking whole hessian)
            if analytic_hess.dtype is np.dtype('object'):
                analytic_hess = unblock(analytic_hess)
                numeric_hess = unblock(numeric_hess)

        ratio = numeric_hess / (numpy.where(analytic_hess==0, 1e-10, analytic_hess))
        difference = numpy.abs(analytic_hess - numeric_hess)

        check_passed = numpy.all((numpy.abs(1 - ratio)) < tolerance) or numpy.allclose(numeric_hess, analytic_hess, atol = tolerance)

        if verbose:
            if block_indices:
                print("\nBlock {}".format(block_indices))
            else:
                print("\nAll blocks")

            header = ['Checked', 'Max-Ratio', 'Min-Ratio', 'Min-Difference', 'Max-Difference']
            header_string = map(lambda x: ' | '.join(header), [header])
            separator = '-' * len(header_string[0])
            print('\n'.join([header_string[0], separator]))
            min_r = '%.6f' % float(numpy.min(ratio))
            max_r = '%.6f' % float(numpy.max(ratio))
            max_d = '%.6f' % float(numpy.max(difference))
            min_d = '%.6f' % float(numpy.min(difference))
            cols = [max_r, min_r, min_d, max_d]

            if check_passed:
                checked = "\033[92m  True \033[0m"
            else:
                checked = "\033[91m  False \033[0m"

            grad_string = "{} | {}  | {} |    {}    |   {} ".format(checked, cols[0], cols[1], cols[2], cols[3])
            print(grad_string)

            if plot:
                from matplotlib import pyplot as pb
                fig, axes = pb.subplots(2, 2)
                max_lim = numpy.max(numpy.vstack((analytic_hess, numeric_hess)))
                min_lim = numpy.min(numpy.vstack((analytic_hess, numeric_hess)))
                msa = axes[0,0].matshow(analytic_hess, vmin=min_lim, vmax=max_lim)
                axes[0,0].set_title('Analytic hessian')
                axes[0,0].xaxis.set_ticklabels([None])
                axes[0,0].yaxis.set_ticklabels([None])
                axes[0,0].xaxis.set_ticks([None])
                axes[0,0].yaxis.set_ticks([None])
                msn = axes[0,1].matshow(numeric_hess, vmin=min_lim, vmax=max_lim)
                pb.colorbar(msn, ax=axes[0,1])
                axes[0,1].set_title('Numeric hessian')
                axes[0,1].xaxis.set_ticklabels([None])
                axes[0,1].yaxis.set_ticklabels([None])
                axes[0,1].xaxis.set_ticks([None])
                axes[0,1].yaxis.set_ticks([None])
                msr = axes[1,0].matshow(ratio)
                pb.colorbar(msr, ax=axes[1,0])
                axes[1,0].set_title('Ratio')
                axes[1,0].xaxis.set_ticklabels([None])
                axes[1,0].yaxis.set_ticklabels([None])
                axes[1,0].xaxis.set_ticks([None])
                axes[1,0].yaxis.set_ticks([None])
                msd = axes[1,1].matshow(difference)
                pb.colorbar(msd, ax=axes[1,1])
                axes[1,1].set_title('difference')
                axes[1,1].xaxis.set_ticklabels([None])
                axes[1,1].yaxis.set_ticklabels([None])
                axes[1,1].xaxis.set_ticks([None])
                axes[1,1].yaxis.set_ticks([None])
                if block_indices:
                    fig.suptitle("Block: {}".format(block_indices))
                pb.show()

        return check_passed

class SkewChecker(HessianChecker):

    def __init__(self, df, ddf, dddf, x0, names=None, *args, **kwargs):
        """
        :param df: gradient of function
        :param ddf: Gradient of function to check (hessian)
        :param dddf: Analytical gradient function (third derivative)
        :param x0:
            Initial guess for inputs x (if it has a shape (a,b) this will be reflected in the parameter names).
            Can be a list of arrays, if takes a list of arrays. This list will be passed
            to f and df in the same order as given here.
            If only one argument, make sure not to pass a list!!!

        :type x0: [array-like] | array-like | float | int
        :param names:
            Names to print, when performing gradcheck. If a list was passed to x0
            a list of names with the same length is expected.
        :param args: Arguments passed as f(x, *args, **kwargs) and df(x, *args, **kwargs)

        """
        super(SkewChecker, self).__init__(df, ddf, dddf, x0, names=names, *args, **kwargs)

    def checkgrad(self, target_param=None, verbose=False, step=1e-6, tolerance=1e-3, block_indices=None, plot=False, super_plot=False):
        """
        Gradient checker that just checks each hessian individually

        super_plot will plot the hessian wrt every parameter, plot will just do the first one
        """
        try:
            import numdifftools as nd
        except:
            raise ImportError("Don't have numdifftools package installed, it is not a GPy dependency as of yet, it is only used for hessian tests")

        if target_param:
            raise NotImplementedError('Only basic functionality is provided with this gradchecker')

        #Repeat for each parameter, not the nicest but shouldn't be many cases where there are many
        #variables
        current_index = 0
        for name, n_shape in zip(self.names, self.shapes):
            current_size = numpy.prod(n_shape)
            x = self.optimizer_array.copy()
            #x = self._get_params_transformed().copy()
            x = x[current_index:current_index + current_size].reshape(n_shape)

            # Check gradients
            #Actually the third derivative
            analytic_hess = self._ddf(x)

            #Can only calculate jacobian for one variable at a time
            #From the docs:
            #x0 : vector location
            #at which to differentiate fun
            #If x0 is an N x M array, then fun is assumed to be a function
            #of N*M variables., thus we must have it flat, not (N,1), but just (N,)
            #numeric_hess_partial = nd.Hessian(self._f, vectorized=False)
            #Actually _df is already the hessian
            numeric_hess_partial = nd.Jacobian(self._df, vectorized=True)
            numeric_hess = numeric_hess_partial(x)

            print("Done making numerical hessian")
            if analytic_hess.dtype is np.dtype('object'):
                #Blockify numeric_hess aswell
                blocksizes, pagesizes = get_block_shapes_3d(analytic_hess)
                #HACK
                real_block_size = np.sum(blocksizes)
                numeric_hess = numeric_hess.reshape(real_block_size, real_block_size, pagesizes)
                #numeric_hess = get_blocks_3d(numeric_hess, blocksizes)#, pagesizes)
            else:
                numeric_hess = numeric_hess.reshape(*analytic_hess.shape)

            #Check every block individually (for ease)
            check_passed = [False]*numeric_hess.shape[2]
            for block_ind in range(numeric_hess.shape[2]):
                #Unless super_plot is set, just plot the first one
                p = True if (plot and block_ind == numeric_hess.shape[2]-1) or super_plot else False
                if verbose:
                    print("Checking derivative of hessian wrt parameter number {}".format(block_ind))
                check_passed[block_ind] = self.checkgrad_block(analytic_hess[:,:,block_ind], numeric_hess[:,:,block_ind], verbose=verbose, step=step, tolerance=tolerance, block_indices=block_indices, plot=p)

            current_index += current_size
        return np.all(check_passed)

