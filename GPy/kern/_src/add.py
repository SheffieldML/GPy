# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from ...core.parameterization import Parameterized
from ...util.caching import Cache_this
from kern import CombinationKernel

class Add(CombinationKernel):
    """
    Add given list of kernels together.
    propagates gradients thorugh.
    """
    def __init__(self, subkerns, name='add'):
        super(Add, self).__init__(subkerns, name)

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        """
        Add all kernels together.
        If a list of parts (of this kernel!) `which_parts` is given, only
        the parts of the list are taken to compute the covariance.
        """
        assert X.shape[1] == self.input_dim
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.add, (p.K(X, X2) for p in which_parts))

    @Cache_this(limit=2, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        assert X.shape[1] == self.input_dim
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.add, (p.Kdiag(X) for p in which_parts))

    def update_gradients_full(self, dL_dK, X, X2=None):
        [p.update_gradients_full(dL_dK, X, X2) for p in self.parts]

    def update_gradients_diag(self, dL_dK, X):
        [p.update_gradients_diag(dL_dK, X) for p in self.parts]

    def update_gradients_diag(self, dL_dKdiag, X):
        [p.update_gradients_diag(dL_dKdiag, X[:,i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

    def gradients_X(self, dL_dK, X, X2=None):
        """Compute the gradient of the objective function with respect to X.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)"""

        target = np.zeros(X.shape)
        [target.__setitem__([Ellipsis, p.active_dims], target[:, p.active_dims]+p.gradients_X(dL_dK, X, X2)) for p in self.parts]
        return target

    def psi0(self, Z, variational_posterior):
        return reduce(np.add, (p.psi0(Z, variational_posterior) for p in self.parts))

    def psi1(self, Z, variational_posterior):
        return reduce(np.add, (p.psi1(Z, variational_posterior) for p in self.parts))

    def psi2(self, Z, variational_posterior):
        psi2 = reduce(np.add, (p.psi2(Z, variational_posterior) for p in self.parts))
        #return psi2
        # compute the "cross" terms
        from static import White, Bias
        from rbf import RBF
        #from rbf_inv import RBFInv
        from linear import Linear
        #ffrom fixed import Fixed

        for p1, p2 in itertools.combinations(self.parts, 2):
            # i1, i2 = p1.active_dims, p2.active_dims
            # white doesn;t combine with anything
            if isinstance(p1, White) or isinstance(p2, White):
                pass
            # rbf X bias
            #elif isinstance(p1, (Bias, Fixed)) and isinstance(p2, (RBF, RBFInv)):
            elif isinstance(p1,  Bias) and isinstance(p2, (RBF, Linear)):
                tmp = p2.psi1(Z, variational_posterior)
                psi2 += p1.variance * (tmp[:, :, None] + tmp[:, None, :])
            #elif isinstance(p2, (Bias, Fixed)) and isinstance(p1, (RBF, RBFInv)):
            elif isinstance(p2, Bias) and isinstance(p1, (RBF, Linear)):
                tmp = p1.psi1(Z, variational_posterior)
                psi2 += p2.variance * (tmp[:, :, None] + tmp[:, None, :])
            elif isinstance(p2, (RBF, Linear)) and isinstance(p1, (RBF, Linear)):
                assert np.intersect1d(p1.active_dims, p2.active_dims).size == 0, "only non overlapping kernel dimensions allowed so far"
                tmp1 = p1.psi1(Z, variational_posterior)
                tmp2 = p2.psi1(Z, variational_posterior)
                psi2 += (tmp1[:, :, None] * tmp2[:, None, :]) + (tmp2[:, :, None] * tmp1[:, None, :])
            else:
                raise NotImplementedError, "psi2 cannot be computed for this kernel"
        return psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        for p1 in self.parts:
            #compute the effective dL_dpsi1. Extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2 in self.parts:
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:# np.setdiff1d(p1.active_dims, ar2, assume_unique): # TODO: Careful, not correct for overlapping active_dims
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
            p1.update_gradients_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)

    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        target = np.zeros(Z.shape)
        for p1 in self.parts:
            #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2 in self.parts:
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
            target[:, p1.active_dims] += p1.gradients_Z_expectations(eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        return target

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        target_mu = np.zeros(variational_posterior.shape)
        target_S = np.zeros(variational_posterior.shape)
        for p1 in self._parameters_:
            #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2 in self._parameters_:
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z, variational_posterior) * 2.
            a, b = p1.gradients_qX_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
            target_mu[:, p1.active_dims] += a
            target_S[:, p1.active_dims] += b
        return target_mu, target_S

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return super(Add, self)._getstate()

    def _setstate(self, state):
        super(Add, self)._setstate(state)


