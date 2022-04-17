from GPy.core.mapping import Mapping
from GPy.core import Param
import numpy as np

class PiecewiseLinear(Mapping):
    """
    A piecewise-linear mapping.

    The parameters of this mapping are the positions and values of the function where it is broken (self.breaks, self.values).

    Outside the range of the breaks, the function is assumed to have gradient 1
    """
    def __init__(self, input_dim, output_dim, values, breaks, name='piecewise_linear'):

        assert input_dim==1
        assert output_dim==1

        super(PiecewiseLinear, self).__init__(input_dim, output_dim, name)

        values, breaks = np.array(values).flatten(), np.array(breaks).flatten()
        assert values.size == breaks.size
        self.values = Param('values', values)
        self.breaks = Param('breaks', breaks)
        self.link_parameter(self.values)
        self.link_parameter(self.breaks)

    def parameters_changed(self):
        self.order = np.argsort(self.breaks)*1
        self.reverse_order = np.zeros_like(self.order)
        self.reverse_order[self.order] = np.arange(self.order.size)

        self.sorted_breaks = self.breaks[self.order]
        self.sorted_values = self.values[self.order]

        self.grads = np.diff(self.sorted_values)/np.diff(self.sorted_breaks)

    def f(self, X):
        x = X.flatten()
        y = x.copy()

        #first adjus the points below the first value
        y[x<self.sorted_breaks[0]]  = x[x<self.sorted_breaks[0]] + self.sorted_values[0] - self.sorted_breaks[0]

        #now all the points pas the last break
        y[x>self.sorted_breaks[-1]]  = x[x>self.sorted_breaks[-1]] + self.sorted_values[-1] - self.sorted_breaks[-1]

        #loop throught the pairs of points
        for low, up, g, v in zip(self. sorted_breaks[:-1], self.sorted_breaks[1:], self.grads, self.sorted_values[:-1]):
            i = np.logical_and(x>low, x<up)
            y[i] = v + (x[i]-low)*g

        return y.reshape(-1,1)

    def update_gradients(self, dL_dF, X):
        x = X.flatten()
        dL_dF = dL_dF.flatten()

        dL_db = np.zeros(self.sorted_breaks.size)
        dL_dv = np.zeros(self.sorted_values.size)

        #loop across each interval, computing the gradient for each of the 4 parameters that define it
        for i, (low, up, g, v) in enumerate(zip(self. sorted_breaks[:-1], self.sorted_breaks[1:], self.grads, self.sorted_values[:-1])):
            index = np.logical_and(x>low, x<up)
            xx = x[index]
            grad = dL_dF[index]
            span = up-low
            dL_dv[i] += np.sum(grad*( (low - xx)/span + 1))
            dL_dv[i+1] += np.sum(grad*(xx-low)/span)
            dL_db[i] += np.sum(grad*g*(xx-up)/span)
            dL_db[i+1] += np.sum(grad*g*(low-xx)/span)

        #now the end parts
        dL_db[0] -= np.sum(dL_dF[x<self.sorted_breaks[0]])
        dL_db[-1] -= np.sum(dL_dF[x>self.sorted_breaks[-1]])
        dL_dv[0] += np.sum(dL_dF[x<self.sorted_breaks[0]])
        dL_dv[-1] += np.sum(dL_dF[x>self.sorted_breaks[-1]])

        #now put the gradients back in the correct order!
        self.breaks.gradient = dL_db[self.reverse_order]
        self.values.gradient = dL_dv[self.reverse_order]

    def gradients_X(self, dL_dF, X):
        x = X.flatten()

        #outside the range of the breakpoints, the function is just offset by a contant, so the partial derivative is 1.
        dL_dX = dL_dF.copy().flatten()

        #insude the breakpoints, the partial derivative is self.grads
        for low, up, g, v in zip(self. sorted_breaks[:-1], self.sorted_breaks[1:], self.grads, self.sorted_values[:-1]):
            i = np.logical_and(x>low, x<up)
            dL_dX[i] = dL_dF[i]*g

        return dL_dX.reshape(-1,1)

