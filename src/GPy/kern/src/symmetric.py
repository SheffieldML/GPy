import numpy as np

from .kern import Kern


class Symmetric(Kern):
    """
    Symmetric kernel that models a function with even or odd symmetry:

    For even symmetry we have:

    .. math::

        f(x) = f(Ax)

    we then model the function as:

    .. math::

        f(x) = g(x) + g(Ax)

    the corresponding kernel is:

    .. math::

        k(x, x') + k(Ax, x') + k(x, Ax') + k(Ax, Ax')

    For odd symmetry we have:

    .. math::

        f(x) = -f(Ax)

    it does this by modelling:

    .. math::

        f(x) = g(x) - g(Ax)

    with kernel

    .. math::

        k(x, x') - k(Ax, x') - k(x, Ax') + k(Ax, Ax')

    where k(x, x') is the kernel of g(x)

    :param base_kernel: kernel to make symmetric
    :param transform: transformation matrix describing symmetry plane, A in equations above
    :param symmetry_type: 'odd' or 'even' depending on the symmetry needed
    """

    def __init__(self, base_kernel, transform, symmetry_type='even'):
        n_dims = max(base_kernel.active_dims) + 1
        super(Symmetric, self).__init__(n_dims, list(range(n_dims)), name='symmetric_kernel')
        if symmetry_type is 'odd':
            self.symmetry_sign = -1.
        elif symmetry_type is 'even':
            self.symmetry_sign = 1.
        else:
            raise ValueError('symmetry_type input must be ''odd'' or ''even''')
        self.transform = transform
        self.base_kernel = base_kernel
        self.param_names = base_kernel.parameter_names()
        self.link_parameters(self.base_kernel)

    def K(self, X, X2):
        X_sym = X.dot(self.transform)

        if X2 is None:
            X2 = X
            X2_sym = X_sym
        else:
            X2_sym = X2.dot(self.transform)

        cross_term_x_ax = self.symmetry_sign * self.base_kernel.K(X, X2_sym)

        if X2 is None:
            cross_term_ax_x = cross_term_x_ax.T
        else:
            cross_term_ax_x = self.symmetry_sign * \
                self.base_kernel.K(X_sym, X2)

        return (self.base_kernel.K(X, X2) + cross_term_x_ax + cross_term_ax_x
                + self.base_kernel.K(X_sym, X2_sym))

    def Kdiag(self, X):
        n_points = X.shape[0]
        X_sym = X.dot(self.transform)

        # Evaluate cross terms in batches, taking the diag of a larger matrix
        # is wasteful, but is more efficient than calling kernel.K for each data point
        batch_size = 100
        n_batches = int(np.ceil(n_points / float(batch_size)))
        cross_term = np.zeros(X.shape[0])
        for i in range(n_batches):
            i_start = i * batch_size
            i_end = np.min([(i + 1) * batch_size, n_points])
            cross_term[i_start:i_end] = np.diag(self.base_kernel.K(
                X_sym[i_start:i_end, :], X[i_start:i_end, :]))

        return self.base_kernel.Kdiag(X) + 2 * self.symmetry_sign * cross_term + self.base_kernel.Kdiag(X_sym)

    def update_gradients_full(self, dL_dK, X, X2):
        X_sym = X.dot(self.transform)
        if X2 is None:
            X2 = X
        X2_sym = X2.dot(self.transform)

        # Get gradients from base kernel one term at a time
        self.base_kernel.update_gradients_full(dL_dK, X_sym, X2)
        gradient = self.symmetry_sign * self.base_kernel.gradient.copy()

        self.base_kernel.update_gradients_full(dL_dK, X, X2_sym)
        gradient += self.symmetry_sign * self.base_kernel.gradient.copy()

        self.base_kernel.update_gradients_full(dL_dK, X_sym, X2_sym)
        gradient += self.base_kernel.gradient.copy()

        self.base_kernel.update_gradients_full(dL_dK, X, X2)
        gradient += self.base_kernel.gradient.copy()

        # Set gradients
        self.base_kernel.gradient = gradient

    def update_gradients_diag(self, dL_dK, X):

        dL_dK_full = np.diag(dL_dK)
        X_sym = X.dot(self.transform)

        # Calculate gradient for k(Ax, Ax')
        self.base_kernel.update_gradients_diag(dL_dK, X_sym)
        gradient = self.base_kernel.gradient.copy()

        # Calculate gradient for k(x, x')
        self.base_kernel.update_gradients_diag(dL_dK, X)
        gradient += self.base_kernel.gradient.copy()

        # Batch process cross term for speed
        batch_size = 100
        n_points = dL_dK.shape[0]
        n_batches = int(np.ceil(n_points / float(batch_size)))
        gradient_part = np.zeros(gradient.shape)
        for i in range(n_batches):
            i_start = i * batch_size
            i_end = np.min([(i + 1) * batch_size, n_points])
            dL_dK_part = dL_dK_full[i_start:i_end, i_start:i_end]
            X_part = X[i_start:i_end, :]
            X_sym_part = X_sym[i_start:i_end, :]
            self.base_kernel.update_gradients_full(
                dL_dK_part, X_part, X_sym_part)
            gradient_part += self.base_kernel.gradient.copy()

        gradient += 2 * self.symmetry_sign * gradient_part

        self.base_kernel.gradient = gradient

    def gradients_X(self, dL_dK, X, X2):
        X_sym = X.dot(self.transform)
        if X2 is None:
            X2 = X
            X2_sym = X.dot(self.transform)
            dL_dK = dL_dK + dL_dK.T
        else:
            X2_sym = X2.dot(self.transform)

        return (self.base_kernel.gradients_X(dL_dK, X, X2)
                + self.base_kernel.gradients_X(dL_dK, X_sym, X2_sym).dot(self.transform.T)
                + self.symmetry_sign * self.base_kernel.gradients_X(dL_dK, X, X2_sym)
                + self.symmetry_sign * self.base_kernel.gradients_X(dL_dK, X_sym, X2).dot(self.transform.T))
