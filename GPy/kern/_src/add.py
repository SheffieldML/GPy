# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import numpy as np
import itertools
from linear import Linear
from ...core.parameterization import Parameterized
from ...core.parameterization.param import Param
from kern import Kern

class Add(Kern):
    def __init__(self, subkerns, tensor):
        assert all([isinstance(k, Kern) for k in subkerns])
        if tensor:
            input_dim  = sum([k.input_dim for k in subkerns])
            self.input_slices = []
            n = 0
            for k in subkerns:
                self.input_slices.append(slice(n, n+k.input_dim))
                n += k.input_dim
        else:
            assert all([k.input_dim == subkerns[0].input_dim for k in subkerns])
            input_dim = subkerns[0].input_dim
            self.input_slices = [slice(None) for k in subkerns]
        super(Add, self).__init__(input_dim, 'add')
        self.add_parameters(*subkerns)


    def K(self, X, X2=None):
        """
        Compute the kernel function.

        :param X: the first set of inputs to the kernel
        :param X2: (optional) the second set of arguments to the kernel. If X2
                   is None, this is passed throgh to the 'part' object, which
                   handles this as X2 == X.
        """
        assert X.shape[1] == self.input_dim
        if X2 is None:
            return sum([p.K(X[:, i_s], None) for p, i_s in zip(self._parameters_, self.input_slices)])
        else:
            return sum([p.K(X[:, i_s], X2[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)])

    def update_gradients_full(self, dL_dK, X):
        [p.update_gradients_full(dL_dK, X[:,i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

    def update_gradients_sparse(self, dL_dKmm, dL_dKnm, dL_dKdiag, X, Z):
        [p.update_gradients_sparse(dL_dKmm, dL_dKnm, dL_dKdiag, X[:,i_s], Z[:,i_s]) for p, i_s in zip(self._parameters_, i_s)]

    def update_gradients_variational(self, dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z):
        [p.update_gradients_variational(dL_dKmm, dL_dpsi0, dL_dpsi1, dL_dpsi2, mu, S, Z) for p in self._parameters_]

    def gradients_X(self, dL_dK, X, X2=None):
        """Compute the gradient of the objective function with respect to X.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)"""

        target = np.zeros_like(X)
        if X2 is None:
            [np.add(target[:,i_s], p.gradients_X(dL_dK, X[:, i_s], None), target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        else:
            [np.add(target[:,i_s], p.gradients_X(dL_dK, X[:, i_s], X2[:,i_s]), target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def Kdiag(self, X):
        """Compute the diagonal of the covariance function for inputs X."""
        assert X.shape[1] == self.input_dim
        return sum([p.Kdiag(X[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)])

    def psi0(self, Z, mu, S):
        target = np.zeros(mu.shape[0])
        [p.psi0(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S):
        target = np.zeros(self.size)
        [p.dpsi0_dtheta(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self._parameters_, self._param_slices_, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S):
        target_mu, target_S = np.zeros_like(mu), np.zeros_like(S)
        [p.dpsi0_dmuS(dL_dpsi0, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target_mu, target_S

    def psi1(self, Z, mu, S):
        target = np.zeros((mu.shape[0], Z.shape[0]))
        [p.psi1(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S):
        target = np.zeros((self.size))
        [p.dpsi1_dtheta(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, ps, i_s in zip(self._parameters_, self._param_slices_, self.input_slices)]
        return self._transform_gradients(target)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi1_dZ(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S):
        """return shapes are num_samples,num_inducing,input_dim"""
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi1_dmuS(dL_dpsi1, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        return target_mu, target_S

    def psi2(self, Z, mu, S):
        """
        Computer the psi2 statistics for the covariance function.

        :param Z: np.ndarray of inducing inputs (num_inducing x input_dim)
        :param mu, S: np.ndarrays of means and variances (each num_samples x input_dim)
        :returns psi2: np.ndarray (num_samples,num_inducing,num_inducing)

        """
        target = np.zeros((mu.shape[0], Z.shape[0], Z.shape[0]))
        [p.psi2(Z[:, i_s], mu[:, i_s], S[:, i_s], target) for p, i_s in zip(self._parameters_, self.input_slices)]

        # compute the "cross" terms
        # TODO: input_slices needed
        crossterms = 0

        for [p1, i_s1], [p2, i_s2] in itertools.combinations(zip(self._parameters_, self.input_slices), 2):
            if i_s1 == i_s2:
                # TODO psi1 this must be faster/better/precached/more nice
                tmp1 = np.zeros((mu.shape[0], Z.shape[0]))
                p1.psi1(Z[:, i_s1], mu[:, i_s1], S[:, i_s1], tmp1)
                tmp2 = np.zeros((mu.shape[0], Z.shape[0]))
                p2.psi1(Z[:, i_s2], mu[:, i_s2], S[:, i_s2], tmp2)

                prod = np.multiply(tmp1, tmp2)
                crossterms += prod[:, :, None] + prod[:, None, :]

        target += crossterms
        return target

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S):
        """Gradient of the psi2 statistics with respect to the parameters."""
        target = np.zeros(self.size)
        [p.dpsi2_dtheta(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[ps]) for p, i_s, ps in zip(self._parameters_, self.input_slices, self._param_slices_)]

        # compute the "cross" terms
        # TODO: better looping, input_slices
        for i1, i2 in itertools.permutations(range(len(self._parameters_)), 2):
            p1, p2 = self._parameters_[i1], self._parameters_[i2]
#             ipsl1, ipsl2 = self.input_slices[i1], self.input_slices[i2]
            ps1, ps2 = self._param_slices_[i1], self._param_slices_[i2]

            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dtheta((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target[ps2])

        return self._transform_gradients(target)

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S):
        target = np.zeros_like(Z)
        [p.dpsi2_dZ(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]
        # target *= 2

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self._parameters_, 2):
#             if p1.name == 'linear' and p2.name == 'linear':
#                 raise NotImplementedError("We don't handle linear/linear cross-terms")
            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dZ((tmp[:, None, :] * dL_dpsi2).sum(1), Z, mu, S, target)

        return target * 2

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S):
        target_mu, target_S = np.zeros((2, mu.shape[0], mu.shape[1]))
        [p.dpsi2_dmuS(dL_dpsi2, Z[:, i_s], mu[:, i_s], S[:, i_s], target_mu[:, i_s], target_S[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

        # compute the "cross" terms
        # TODO: we need input_slices here.
        for p1, p2 in itertools.permutations(self._parameters_, 2):
#             if p1.name == 'linear' and p2.name == 'linear':
#                 raise NotImplementedError("We don't handle linear/linear cross-terms")
            tmp = np.zeros((mu.shape[0], Z.shape[0]))
            p1.psi1(Z, mu, S, tmp)
            p2.dpsi1_dmuS((tmp[:, None, :] * dL_dpsi2).sum(1) * 2., Z, mu, S, target_mu, target_S)

        return target_mu, target_S

    def plot(self, *args, **kwargs):
        """
        See GPy.plotting.matplot_dep.plot
        """
        assert "matplotlib" in sys.modules, "matplotlib package has not been imported."
        from ..plotting.matplot_dep import kernel_plots
        kernel_plots.plot(self,*args)

    def _getstate(self):
        """
        Get the current state of the class,
        here just all the indices, rest can get recomputed
        """
        return Parameterized._getstate(self) + [#self._parameters_,
                self.input_dim,
                self.input_slices,
                self._param_slices_
                ]

    def _setstate(self, state):
        self._param_slices_ = state.pop()
        self.input_slices = state.pop()
        self.input_dim = state.pop()
        Parameterized._setstate(self, state)


