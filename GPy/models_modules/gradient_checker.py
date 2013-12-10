'''
Created on 17 Jul 2013

@author: maxz
'''
from GPy.core.model import Model
import itertools
import numpy

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
            Can be a list of arrays, if f takes a list of arrays. This list will be passed
            to f and df in the same order as given here.
            If f takes only one argument, make sure not to pass a list for x0!!!
        :type x0: [array-like] | array-like | float | int
        :param list names:
            Names to print, when performing gradcheck. If a list was passed to x0
            a list of names with the same length is expected.
        :param args kwargs: Arguments passed as f(x, *args, **kwargs) and df(x, *args, **kwargs)

        Examples:
        ---------
        from GPy.models import GradientChecker
        N, M, Q = 10, 5, 3

        Sinusoid:

            X = numpy.random.rand(N, Q)
            grad = GradientChecker(numpy.sin,numpy.cos,X,'sin_in')
            grad.checkgrad(verbose=1)

        Using GPy:

            X, Z = numpy.random.randn(N,Q), numpy.random.randn(M,Q)
            kern = GPy.kern.linear(Q, ARD=True) + GPy.kern.rbf(Q, ARD=True)
            grad = GradientChecker(kern.K,
                                   lambda x: kern.dK_dX(numpy.ones((1,1)), x),
                                   x0 = X.copy(),
                                   names=['X_input'])
            grad.checkgrad(verbose=1)
            grad.randomize()
            grad.checkgrad(verbose=1)
        """
        Model.__init__(self)
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
            self.__setattr__(name, numpy.float_(xi))
#         self._param_names = []
#         for name, shape in zip(self.names, self.shapes):
#             self._param_names.extend(map(lambda nameshape: ('_'.join(nameshape)).strip('_'), itertools.izip(itertools.repeat(name), itertools.imap(lambda t: '_'.join(map(str, t)), itertools.product(*map(lambda xi: range(xi), shape))))))
        self.args = args
        self.kwargs = kwargs
        self._f = f
        self._df = df

    def _get_x(self):
        if len(self.names) > 1:
            return [self.__getattribute__(name) for name in self.names] + list(self.args)
        return [self.__getattribute__(self.names[0])] + list(self.args)

    def log_likelihood(self):
        return float(numpy.sum(self._f(*self._get_x(), **self.kwargs)))

    def _log_likelihood_gradients(self):
        return numpy.atleast_1d(self._df(*self._get_x(), **self.kwargs)).flatten()


    def _get_params(self):
        return numpy.atleast_1d(numpy.hstack(map(lambda name: flatten_if_needed(self.__getattribute__(name)), self.names)))


    def _set_params(self, x):
        current_index = 0
        for name, shape in zip(self.names, self.shapes):
            current_size = numpy.prod(shape)
            self.__setattr__(name, x[current_index:current_index + current_size].reshape(shape))
            current_index += current_size

    def _get_param_names(self):
        _param_names = []
        for name, shape in zip(self.names, self.shapes):
            _param_names.extend(map(lambda nameshape: ('_'.join(nameshape)).strip('_'), itertools.izip(itertools.repeat(name), itertools.imap(lambda t: '_'.join(map(str, t)), itertools.product(*map(lambda xi: range(xi), shape))))))
        return _param_names
