# Copyright (c) 2012, James Hesnsman
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .kern import CombinationKernel
import numpy as np
import itertools
from ...util.multioutput import index_to_slices


class IndependentOutputs(CombinationKernel):
    """
    A kernel which can represent several independent functions.  this kernel
    'switches off' parts of the matrix where the output indexes are different.

    The index of the functions is given by the last column in the input X the
    rest of the columns of X are passed to the underlying kernel for
    computation (in blocks).

    :param kernels: either a kernel, or list of kernels to work with. If it is
    a list of kernels the indices in the index_dim, index the kernels you gave!
    """
    def __init__(self, kernels, index_dim=-1, name='independ'):
        assert isinstance(index_dim, int), "IndependentOutputs kernel is only defined with one input dimension being the index"
        if not isinstance(kernels, list):
            self.single_kern = True
            self.kern = kernels
            kernels = [kernels]
        else:
            self.single_kern = False
            self.kern = kernels
        super(IndependentOutputs, self).__init__(kernels=kernels, extra_dims=[index_dim], name=name)
        # The combination kernel ALLWAYS puts the extra dimension last.
        # Thus, the index dimension of this kernel is always the last dimension
        # after slicing. This is why the index_dim is just the last column:
        self.index_dim = -1

    def K(self,X ,X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        if X2 is None:
            target = np.zeros((X.shape[0], X.shape[0]))
            #[[target.__setitem__((s,ss), kern.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices_i, slices_i)] for kern, slices_i in zip(kerns, slices)]
            [[target.__setitem__((s,ss), kern.K(X[s,:]) if s==ss else kern.K(X[s,:], X[ss,:])) for s,ss in itertools.product(slices_i, slices_i)] for kern, slices_i in zip(kerns, slices)]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            target = np.zeros((X.shape[0], X2.shape[0]))
            [[target.__setitem__((s,s2), kern.K(X[s,:],X2[s2,:])) for s,s2 in itertools.product(slices_i, slices_j)] for kern, slices_i,slices_j in zip(kerns, slices,slices2)]
        return target

    def Kdiag(self,X):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        target = np.zeros(X.shape[0])
        [[np.copyto(target[s], kern.Kdiag(X[s])) for s in slices_i] for kern, slices_i in zip(kerns, slices)]
        return target

    def update_gradients_full(self,dL_dK,X,X2=None):
        slices = index_to_slices(X[:,self.index_dim])
        if self.single_kern:
            target = np.zeros(self.kern.size)
            kerns = itertools.repeat(self.kern)
        else:
            kerns = self.kern
            target = [np.zeros(kern.size) for kern, _ in zip(kerns, slices)]
        def collate_grads(kern, i, dL, X, X2):
            kern.update_gradients_full(dL,X,X2)
            if self.single_kern: target[:] += kern.gradient
            else: target[i][:] += kern.gradient
        if X2 is None:
            [[collate_grads(kern, i, dL_dK[s,ss], X[s], X[ss]) for s,ss in itertools.product(slices_i, slices_i)] for i,(kern,slices_i) in enumerate(zip(kerns,slices))]
        else:
            slices2 = index_to_slices(X2[:,self.index_dim])
            [[[collate_grads(kern, i, dL_dK[s,s2],X[s],X2[s2]) for s in slices_i] for s2 in slices_j] for i,(kern,slices_i,slices_j) in enumerate(zip(kerns,slices,slices2))]
        if self.single_kern:
            self.kern.gradient = target
        else:
            [kern.gradient.__setitem__(Ellipsis, target[i]) for i, [kern, _] in enumerate(zip(kerns, slices))]

    def gradients_X(self,dL_dK, X, X2=None):
        target = np.zeros(X.shape)
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        if X2 is None:
            values = np.unique(X[:,self.index_dim])
            slices = [X[:,self.index_dim]==i for i in values]
            for kern, s in zip(kerns, slices):
                target[s] += kern.gradients_X(dL_dK[s, :][:, s],X[s], None)
            #slices = index_to_slices(X[:,self.index_dim])
            #[[np.add(target[s], kern.gradients_X(dL_dK[s,s], X[s]), out=target[s])
            #  for s in slices_i] for kern, slices_i in zip(kerns, slices)]
            #import ipdb;ipdb.set_trace()
            #[[(np.add(target[s ], kern.gradients_X(dL_dK[s ,ss],X[s ], X[ss]), out=target[s ]),
            #   np.add(target[ss], kern.gradients_X(dL_dK[ss,s ],X[ss], X[s ]), out=target[ss]))
            #  for s, ss in itertools.combinations(slices_i, 2)] for kern, slices_i in zip(kerns, slices)]
        else:
            values = np.unique(X[:,self.index_dim])
            slices = [X[:,self.index_dim]==i for i in values]
            slices2 = [X2[:,self.index_dim]==i for i in values]
            for kern, s, s2 in zip(kerns, slices, slices2):
                target[s] += kern.gradients_X(dL_dK[s, :][:, s2],X[s],X2[s2])
            # TODO: make work with index_to_slices
            #slices = index_to_slices(X[:,self.index_dim])
            #slices2 = index_to_slices(X2[:,self.index_dim])
            #[[target.__setitem__(s, target[s] + kern.gradients_X(dL_dK[s,s2], X[s], X2[s2])) for s, s2 in itertools.product(slices_i, slices_j)] for kern, slices_i,slices_j in zip(kerns, slices,slices2)]
        return target

    def gradients_X_diag(self, dL_dKdiag, X):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        target = np.zeros(X.shape)
        for kern, slices_i in zip(kerns, slices):
            for s in slices_i:
                target[s] += kern.gradients_X_diag(dL_dKdiag[s],X[s])
        return target

    def update_gradients_diag(self, dL_dKdiag, X):
        slices = index_to_slices(X[:,self.index_dim])
        kerns = itertools.repeat(self.kern) if self.single_kern else self.kern
        if self.single_kern: target = np.zeros(self.kern.size)
        else: target = [np.zeros(kern.size) for kern, _ in zip(kerns, slices)]
        def collate_grads(kern, i, dL, X):
            kern.update_gradients_diag(dL,X)
            if self.single_kern: target[:] += kern.gradient
            else: target[i][:] += kern.gradient
        [[collate_grads(kern, i, dL_dKdiag[s], X[s,:]) for s in slices_i] for i, (kern, slices_i) in enumerate(zip(kerns, slices))]
        if self.single_kern: self.kern.gradient = target
        else:[kern.gradient.__setitem__(Ellipsis, target[i]) for i, [kern, _] in enumerate(zip(kerns, slices))]

class Hierarchical(CombinationKernel):
    """
    A kernel which can represent a simple hierarchical model.

    See Hensman et al 2013, "Hierarchical Bayesian modelling of gene expression time
    series across irregularly sampled replicates and clusters"
    http://www.biomedcentral.com/1471-2105/14/252

    To construct this kernel, you must pass a list of kernels. the first kernel
    will be assumed to be the 'base' kernel, and will be computed everywhere.
    For every additional kernel, we assume another layer in the hierachy, with
    a corresponding column of the input matrix which indexes which function the
    data are in at that level.

    For more, see the ipython notebook documentation on Hierarchical
    covariances.
    """
    def __init__(self, kernels, name='hierarchy'):
        assert all([k.input_dim==kernels[0].input_dim for k in kernels])
        assert len(kernels) > 1
        self.levels = len(kernels) -1
        input_max = max([k.input_dim for k in kernels])
        super(Hierarchical, self).__init__(kernels=kernels, extra_dims = range(input_max, input_max + len(kernels)-1), name=name)

    def K(self,X ,X2=None):
        K = self.parts[0].K(X, X2) # compute 'base' kern everywhere
        slices = [index_to_slices(X[:,i]) for i in self.extra_dims]
        if X2 is None:
            [[[np.add(K[s,s], k.K(X[s], None), K[s, s]) for s in slices_i] for slices_i in slices_k] for k, slices_k in zip(self.parts[1:], slices)]
        else:
            slices2 = [index_to_slices(X2[:,i]) for i in self.extra_dims]
            [[[np.add(K[s,ss], k.K(X[s], X2[ss]), K[s, ss]) for s,ss in zip(slices_i, slices_j)] for slices_i, slices_j in zip(slices_k1, slices_k2)] for k, slices_k1, slices_k2 in zip(self.parts[1:], slices, slices2)]
        return K

    def Kdiag(self,X):
        return np.diag(self.K(X))

    def gradients_X(self, dL_dK, X, X2=None):
        raise NotImplementedError

    def update_gradients_full(self,dL_dK,X,X2=None):
        slices = [index_to_slices(X[:,i]) for i in self.extra_dims]
        if X2 is None:
            self.parts[0].update_gradients_full(dL_dK, X, None)
            for k, slices_k in zip(self.parts[1:], slices):
                target = np.zeros(k.size)
                def collate_grads(dL, X, X2, target):
                    k.update_gradients_full(dL,X,X2)
                    target += k.gradient
                [[collate_grads(dL_dK[s,s], X[s], None, target) for s in slices_i] for slices_i in slices_k]
                k.gradient[:] = target
        else:
            raise NotImplementedError


