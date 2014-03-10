# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from ...core.parameterization import Parameterized
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
                   handLes this as X2 == X.
        """
        assert X.shape[1] == self.input_dim
        if X2 is None:
            return sum([p.K(X[:, i_s], None) for p, i_s in zip(self._parameters_, self.input_slices)])
        else:
            return sum([p.K(X[:, i_s], X2[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)])

    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:
            [p.update_gradients_full(dL_dK, X[:,i_s], X2) for p, i_s in zip(self._parameters_, self.input_slices)]
        else:
            [p.update_gradients_full(dL_dK, X[:,i_s], X2[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

    def update_gradients_diag(self, dL_dKdiag, X):
        [p.update_gradients_diag(dL_dK, X[:,i_s]) for p, i_s in zip(self._parameters_, self.input_slices)]

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
        assert X.shape[1] == self.input_dim
        return sum([p.Kdiag(X[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)])


    def psi0(self, Z, variational_posterior):
        return np.sum([p.psi0(Z[:, i_s], variational_posterior[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)],0)

    def psi1(self, Z, variational_posterior):
        return np.sum([p.psi1(Z[:, i_s], variational_posterior[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)], 0)

    def psi2(self, Z, variational_posterior):
        psi2 = np.sum([p.psi2(Z[:, i_s], variational_posterior[:, i_s]) for p, i_s in zip(self._parameters_, self.input_slices)], 0)

        # compute the "cross" terms
        from static import White, Bias
        from rbf import RBF
        #from rbf_inv import RBFInv
        from linear import Linear
        #ffrom fixed import Fixed

        for (p1, i1), (p2, i2) in itertools.combinations(itertools.izip(self._parameters_, self.input_slices), 2):
            # white doesn;t combine with anything
            if isinstance(p1, White) or isinstance(p2, White):
                pass
            # rbf X bias
            #elif isinstance(p1, (Bias, Fixed)) and isinstance(p2, (RBF, RBFInv)):
            elif isinstance(p1,  Bias) and isinstance(p2, (RBF, Linear)):
                tmp = p2.psi1(Z[:,i2], variational_posterior[:, i_s])
                psi2 += p1.variance * (tmp[:, :, None] + tmp[:, None, :])
            #elif isinstance(p2, (Bias, Fixed)) and isinstance(p1, (RBF, RBFInv)):
            elif isinstance(p2, Bias) and isinstance(p1, (RBF, Linear)):
                tmp = p1.psi1(Z[:,i1], variational_posterior[:, i_s])
                psi2 += p2.variance * (tmp[:, :, None] + tmp[:, None, :])
            else:
                raise NotImplementedError, "psi2 cannot be computed for this kernel"
        return psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        mu, S = variational_posterior.mean, variational_posterior.variance
        
        for p1, is1 in zip(self._parameters_, self.input_slices):

            #compute the effective dL_dpsi1. Extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2, is2 in zip(self._parameters_, self.input_slices):
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z[:,is2], variational_posterior[:, is1]) * 2.


            p1.update_gradients_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z[:,is1], variational_posterior[:, is1])


    def gradients_Z_expectations(self, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        
        target = np.zeros(Z.shape)
        for p1, is1 in zip(self._parameters_, self.input_slices):

            #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2, is2 in zip(self._parameters_, self.input_slices):
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z[:,is2], variational_posterior[:, is2]) * 2.


            target += p1.gradients_Z_expectations(eff_dL_dpsi1, dL_dpsi2, Z[:,is1], variational_posterior[:, is1])
        return target

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        from static import White, Bias
        
        target_mu = np.zeros(variational_posterior.shape)
        target_S = np.zeros(variational_posterior.shape)
        for p1, is1 in zip(self._parameters_, self.input_slices):

            #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2, is2 in zip(self._parameters_, self.input_slices):
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.variance * 2.
                else:
                    eff_dL_dpsi1 += dL_dpsi2.sum(1) * p2.psi1(Z[:,is2], variational_posterior[:, is2]) * 2.


            a, b = p1.gradients_qX_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z[:,is1], variational_posterior[:, is1])
            target_mu += a
            target_S += b
        return target_mu, target_S

    def input_sensitivity(self):
        in_sen = np.zeros((self.num_params, self.input_dim))
        for i, [p, i_s] in enumerate(zip(self._parameters_, self.input_slices)):
            in_sen[i, i_s] = p.input_sensitivity()
        return in_sen

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


