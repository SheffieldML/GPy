# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from paramz.caching import Cache_this
from .kern import CombinationKernel, Kern
from functools import reduce

class Add(CombinationKernel):
    """
    Add given list of kernels together.
    propagates gradients through.

    This kernel will take over the active dims of it's subkernels passed in.

    NOTE: The subkernels will be copies of the original kernels, to prevent
    unexpected behavior.
    """
    def __init__(self, subkerns, name='sum'):
        _newkerns = []
        for kern in subkerns:
            if isinstance(kern, Add):
                for part in kern.parts:
                    #kern.unlink_parameter(part)
                    _newkerns.append(part.copy())
            else:
                _newkerns.append(kern.copy())

        super(Add, self).__init__(_newkerns, name)
        self._exact_psicomp = self._check_exact_psicomp()

    def _check_exact_psicomp(self):
        from .. import RBF,Linear,Bias,White
        n_kerns = len(self.parts)
        n_rbf = len([k  for k in self.parts if isinstance(k,RBF)])
        n_linear = len([k  for k in self.parts if isinstance(k,Linear)])
        n_bias = len([k  for k in self.parts if isinstance(k,Bias)])
        n_white = len([k  for k in self.parts if isinstance(k,White)])
        n_others = n_kerns - n_rbf - n_linear - n_bias - n_white
        if n_rbf+n_linear<=1 and n_bias<=1 and n_white<=1 and n_others==0:
            return True
        else:
            return False

    def to_dict(self):
        input_dict = super(Add, self)._to_dict()
        input_dict["class"] = str("GPy.kern.Add")
        return input_dict

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def K(self, X, X2=None, which_parts=None):
        """
        Add all kernels together.
        If a list of parts (of this kernel!) `which_parts` is given, only
        the parts of the list are taken to compute the covariance.
        """
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.add, (p.K(X, X2) for p in which_parts))

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def Kdiag(self, X, which_parts=None):
        if which_parts is None:
            which_parts = self.parts
        elif not isinstance(which_parts, (list, tuple)):
            # if only one part is given
            which_parts = [which_parts]
        return reduce(np.add, (p.Kdiag(X) for p in which_parts))

    def update_gradients_full(self, dL_dK, X, X2=None):
        [p.update_gradients_full(dL_dK, X, X2) for p in self.parts if not p.is_fixed]

    def update_gradients_diag(self, dL_dK, X):
        [p.update_gradients_diag(dL_dK, X) for p in self.parts]

    def gradients_X(self, dL_dK, X, X2=None):
        """Compute the gradient of the objective function with respect to X.

        :param dL_dK: An array of gradients of the objective function with respect to the covariance function.
        :type dL_dK: np.ndarray (num_samples x num_inducing)
        :param X: Observed data inputs
        :type X: np.ndarray (num_samples x input_dim)
        :param X2: Observed data inputs (optional, defaults to X)
        :type X2: np.ndarray (num_inducing x input_dim)"""

        target = np.zeros(X.shape)
        [target.__iadd__(p.gradients_X(dL_dK, X, X2)) for p in self.parts]
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape)
        [target.__iadd__(p.gradients_X_diag(dL_dKdiag, X)) for p in self.parts]
        return target

    def gradients_XX(self, dL_dK, X, X2):
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0], X.shape[1], X.shape[1]))
        else:
            target = np.zeros((X.shape[0], X2.shape[0], X.shape[1], X.shape[1]))
        #else: # diagonal covariance
        #    if X2 is None:
        #        target = np.zeros((X.shape[0], X.shape[0], X.shape[1]))
        #    else:
        #        target = np.zeros((X.shape[0], X2.shape[0], X.shape[1]))
        [target.__iadd__(p.gradients_XX(dL_dK, X, X2)) for p in self.parts]
        return target

    def gradients_XX_diag(self, dL_dKdiag, X):
        target = np.zeros(X.shape+(X.shape[1],))
        [target.__iadd__(p.gradients_XX_diag(dL_dKdiag, X)) for p in self.parts]
        return target

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def psi0(self, Z, variational_posterior):
        if not self._exact_psicomp: return Kern.psi0(self,Z,variational_posterior)
        return reduce(np.add, (p.psi0(Z, variational_posterior) for p in self.parts))

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def psi1(self, Z, variational_posterior):
        if not self._exact_psicomp: return Kern.psi1(self,Z,variational_posterior)
        return reduce(np.add, (p.psi1(Z, variational_posterior) for p in self.parts))

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def psi2(self, Z, variational_posterior):
        if not self._exact_psicomp: return Kern.psi2(self,Z,variational_posterior)
        psi2 = reduce(np.add, (p.psi2(Z, variational_posterior) for p in self.parts))
        #return psi2
        # compute the "cross" terms
        from .static import White, Bias
        from .rbf import RBF
        #from rbf_inv import RBFInv
        from .linear import Linear
        #ffrom fixed import Fixed

        for p1, p2 in itertools.combinations(self.parts, 2):
            # i1, i2 = p1._all_dims_active, p2._all_dims_active
            # white doesn;t combine with anything
            if isinstance(p1, White) or isinstance(p2, White):
                pass
            # rbf X bias
            #elif isinstance(p1, (Bias, Fixed)) and isinstance(p2, (RBF, RBFInv)):
            elif isinstance(p1,  Bias) and isinstance(p2, (RBF, Linear)):
                tmp = p2.psi1(Z, variational_posterior).sum(axis=0)
                psi2 += p1.variance * (tmp[:,None]+tmp[None,:]) #(tmp[:, :, None] + tmp[:, None, :])
            #elif isinstance(p2, (Bias, Fixed)) and isinstance(p1, (RBF, RBFInv)):
            elif isinstance(p2, Bias) and isinstance(p1, (RBF, Linear)):
                tmp = p1.psi1(Z, variational_posterior).sum(axis=0)
                psi2 += p2.variance * (tmp[:,None]+tmp[None,:]) #(tmp[:, :, None] + tmp[:, None, :])
            elif isinstance(p2, (RBF, Linear)) and isinstance(p1, (RBF, Linear)):
                assert np.intersect1d(p1._all_dims_active, p2._all_dims_active).size == 0, "only non overlapping kernel dimensions allowed so far"
                tmp1 = p1.psi1(Z, variational_posterior)
                tmp2 = p2.psi1(Z, variational_posterior)
                psi2 += np.einsum('nm,no->mo',tmp1,tmp2)+np.einsum('nm,no->mo',tmp2,tmp1)
                #(tmp1[:, :, None] * tmp2[:, None, :]) + (tmp2[:, :, None] * tmp1[:, None, :])
            else:
                raise NotImplementedError("psi2 cannot be computed for this kernel")
        return psi2

    @Cache_this(limit=3, force_kwargs=['which_parts'])
    def psi2n(self, Z, variational_posterior):
        if not self._exact_psicomp: return Kern.psi2n(self, Z, variational_posterior)
        psi2 = reduce(np.add, (p.psi2n(Z, variational_posterior) for p in self.parts))
        #return psi2
        # compute the "cross" terms
        from .static import White, Bias
        from .rbf import RBF
        #from rbf_inv import RBFInv
        from .linear import Linear
        #ffrom fixed import Fixed

        for p1, p2 in itertools.combinations(self.parts, 2):
            # i1, i2 = p1._all_dims_active, p2._all_dims_active
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
                assert np.intersect1d(p1._all_dims_active, p2._all_dims_active).size == 0, "only non overlapping kernel dimensions allowed so far"
                tmp1 = p1.psi1(Z, variational_posterior)
                tmp2 = p2.psi1(Z, variational_posterior)
                psi2 += np.einsum('nm,no->nmo',tmp1,tmp2)+np.einsum('nm,no->nmo',tmp2,tmp1)
                #(tmp1[:, :, None] * tmp2[:, None, :]) + (tmp2[:, :, None] * tmp1[:, None, :])
            else:
                raise NotImplementedError("psi2 cannot be computed for this kernel")
        return psi2

    def update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        tmp = dL_dpsi2.sum(0)+ dL_dpsi2.sum(1) if len(dL_dpsi2.shape)==2 else dL_dpsi2.sum(2)+ dL_dpsi2.sum(1)

        if not self._exact_psicomp: return Kern.update_gradients_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        from .static import White, Bias
        for p1 in self.parts:
            #compute the effective dL_dpsi1. Extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2 in self.parts:
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += tmp * p2.variance
                else:# np.setdiff1d(p1._all_dims_active, ar2, assume_unique): # TODO: Careful, not correct for overlapping _all_dims_active
                    eff_dL_dpsi1 += tmp * p2.psi1(Z, variational_posterior)
            p1.update_gradients_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)

    def gradients_Z_expectations(self, dL_psi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        tmp = dL_dpsi2.sum(0)+ dL_dpsi2.sum(1) if len(dL_dpsi2.shape)==2 else dL_dpsi2.sum(2)+ dL_dpsi2.sum(1)
        if not self._exact_psicomp: return Kern.gradients_Z_expectations(self, dL_psi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        from .static import White, Bias
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
                    eff_dL_dpsi1 += tmp * p2.variance
                else:
                    eff_dL_dpsi1 += tmp * p2.psi1(Z, variational_posterior)
            target += p1.gradients_Z_expectations(dL_psi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        return target

    def gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior):
        tmp = dL_dpsi2.sum(0)+ dL_dpsi2.sum(1) if len(dL_dpsi2.shape)==2 else dL_dpsi2.sum(2)+ dL_dpsi2.sum(1)

        if not self._exact_psicomp: return Kern.gradients_qX_expectations(self, dL_dpsi0, dL_dpsi1, dL_dpsi2, Z, variational_posterior)
        from .static import White, Bias
        target_grads = [np.zeros(v.shape) for v in variational_posterior.parameters]
        for p1 in self.parameters:
            #compute the effective dL_dpsi1. extra terms appear becaue of the cross terms in psi2!
            eff_dL_dpsi1 = dL_dpsi1.copy()
            for p2 in self.parameters:
                if p2 is p1:
                    continue
                if isinstance(p2, White):
                    continue
                elif isinstance(p2, Bias):
                    eff_dL_dpsi1 += tmp * p2.variance
                else:
                    eff_dL_dpsi1 += tmp * p2.psi1(Z, variational_posterior)
            grads = p1.gradients_qX_expectations(dL_dpsi0, eff_dL_dpsi1, dL_dpsi2, Z, variational_posterior)
            [np.add(target_grads[i],grads[i],target_grads[i]) for i in range(len(grads))]
        return target_grads

    #def add(self, other):
    #    parts = self.parts
    #    if 0:#isinstance(other, Add):
    #        #other_params = other.parameters[:]
    #        for p in other.parts[:]:
    #            other.unlink_parameter(p)
    #        parts.extend(other.parts)
    #        #self.link_parameters(*other_params)
    #
    #    else:
    #        #self.link_parameter(other)
    #        parts.append(other)
    #    #self.input_dim, self._all_dims_active = self.get_input_dim_active_dims(parts)
    #    return Add([p for p in parts], self.name)

    def input_sensitivity(self, summarize=True):
        if summarize:
            i_s = np.zeros((self.input_dim))
            for k in self.parts:
                i_s[k._all_dims_active] += k.input_sensitivity(summarize)
            return i_s
        else:

            return super(Add, self).input_sensitivity(summarize)

    def sde_update_gradient_full(self, gradients):
        """
        Update gradient in the order in which parameters are represented in the
        kernel
        """
        part_start_param_index = 0
        for p in self.parts:
            if not p.is_fixed:
                part_param_num = len(p.param_array) # number of parameters in the part
                p.sde_update_gradient_full(gradients[part_start_param_index:(part_start_param_index+part_param_num)])
                part_start_param_index += part_param_num

    def sde(self):
        """
        Support adding kernels for sde representation
        """

        import scipy.linalg as la

        F     = None
        L     = None
        Qc    = None
        H     = None
        Pinf  = None
        P0    = None
        dF    = None
        dQc   = None
        dPinf = None
        dP0   = None
        n = 0
        nq = 0
        nd = 0

         # Assign models
        for p in self.parts:
            (Ft,Lt,Qct,Ht,Pinft,P0t,dFt,dQct,dPinft,dP0t) = p.sde()
            F = la.block_diag(F,Ft) if (F is not None) else Ft
            L = la.block_diag(L,Lt) if (L is not None) else Lt
            Qc = la.block_diag(Qc,Qct) if (Qc is not None) else Qct
            H = np.hstack((H,Ht)) if (H is not None) else Ht

            Pinf = la.block_diag(Pinf,Pinft) if (Pinf is not None) else Pinft
            P0 = la.block_diag(P0,P0t) if (P0 is not None) else P0t

            if dF is not None:
                dF = np.pad(dF,((0,dFt.shape[0]),(0,dFt.shape[1]),(0,dFt.shape[2])),
                        'constant', constant_values=0)
                dF[-dFt.shape[0]:,-dFt.shape[1]:,-dFt.shape[2]:] = dFt
            else:
                dF = dFt

            if dQc is not None:
                dQc = np.pad(dQc,((0,dQct.shape[0]),(0,dQct.shape[1]),(0,dQct.shape[2])),
                        'constant', constant_values=0)
                dQc[-dQct.shape[0]:,-dQct.shape[1]:,-dQct.shape[2]:] = dQct
            else:
                dQc = dQct

            if dPinf is not None:
                dPinf = np.pad(dPinf,((0,dPinft.shape[0]),(0,dPinft.shape[1]),(0,dPinft.shape[2])),
                        'constant', constant_values=0)
                dPinf[-dPinft.shape[0]:,-dPinft.shape[1]:,-dPinft.shape[2]:] = dPinft
            else:
                dPinf = dPinft

            if dP0 is not None:
                dP0 = np.pad(dP0,((0,dP0t.shape[0]),(0,dP0t.shape[1]),(0,dP0t.shape[2])),
                        'constant', constant_values=0)
                dP0[-dP0t.shape[0]:,-dP0t.shape[1]:,-dP0t.shape[2]:] = dP0t
            else:
                dP0 = dP0t

            n += Ft.shape[0]
            nq += Qct.shape[0]
            nd += dFt.shape[2]

        assert (F.shape[0] == n and F.shape[1]==n), "SDE add: Check of F Dimensions failed"
        assert (L.shape[0] == n and L.shape[1]==nq), "SDE add: Check of L Dimensions failed"
        assert (Qc.shape[0] == nq and Qc.shape[1]==nq), "SDE add: Check of Qc Dimensions failed"
        assert (H.shape[0] == 1 and H.shape[1]==n), "SDE add: Check of H Dimensions failed"
        assert (Pinf.shape[0] == n and Pinf.shape[1]==n), "SDE add: Check of Pinf Dimensions failed"
        assert (P0.shape[0] == n and P0.shape[1]==n), "SDE add: Check of P0 Dimensions failed"
        assert (dF.shape[0] == n and dF.shape[1]==n and dF.shape[2]==nd), "SDE add: Check of dF Dimensions failed"
        assert (dQc.shape[0] == nq and dQc.shape[1]==nq and dQc.shape[2]==nd), "SDE add: Check of dQc Dimensions failed"
        assert (dPinf.shape[0] == n and dPinf.shape[1]==n and dPinf.shape[2]==nd), "SDE add: Check of dPinf Dimensions failed"
        assert (dP0.shape[0] == n and dP0.shape[1]==n and dP0.shape[2]==nd), "SDE add: Check of dP0 Dimensions failed"

        return (F,L,Qc,H,Pinf,P0,dF,dQc,dPinf,dP0)
