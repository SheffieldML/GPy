# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
import pylab as pb
from ..core.parameterised import parameterised
from kernpart import kernpart
import itertools
from product_orthogonal import product_orthogonal
from product import product

class kern(parameterised):
    def __init__(self,D,parts=[], input_slices=None):
        """
        This kernel does 'compound' structures.

        The compund structure enables many features of GPy, including
         - Hierarchical models
         - Correleated output models
         - multi-view learning

        Hadamard product and outer-product kernels will require a new class.
        This feature is currently WONTFIX. for small number sof inputs, you can use the sympy kernel for this.

        :param D: The dimensioality of the kernel's input space
        :type D: int
        :param parts: the 'parts' (PD functions) of the kernel
        :type parts: list of kernpart objects
        :param input_slices: the slices on the inputs which apply to each kernel
        :type input_slices: list of slice objects, or list of bools

        """
        self.parts = parts
        self.Nparts = len(parts)
        self.Nparam = sum([p.Nparam for p in self.parts])

        self.D = D

        #deal with input_slices
        if input_slices is None:
            self.input_slices = [slice(None) for p in self.parts]
        else:
            assert len(input_slices)==len(self.parts)
            self.input_slices = [sl if type(sl) is slice else slice(None) for sl in input_slices]

        for p in self.parts:
            assert isinstance(p,kernpart), "bad kernel part"

        self.compute_param_slices()

        parameterised.__init__(self)

    def _transform_gradients(self,g):
        x = self._get_params()
        g[self.constrained_positive_indices] = g[self.constrained_positive_indices]*x[self.constrained_positive_indices]
        g[self.constrained_negative_indices] = g[self.constrained_negative_indices]*x[self.constrained_negative_indices]
        [np.put(g,i,g[i]*(x[i]-l)*(h-x[i])/(h-l)) for i,l,h in zip(self.constrained_bounded_indices, self.constrained_bounded_lowers, self.constrained_bounded_uppers)]
        [np.put(g,i,v) for i,v in [(t[0],np.sum(g[t])) for t in self.tied_indices]]
        if len(self.tied_indices) or len(self.constrained_fixed_indices):
            to_remove = np.hstack((self.constrained_fixed_indices+[t[1:] for t in self.tied_indices]))
            return np.delete(g,to_remove)
        else:
            return g

    def compute_param_slices(self):
        """create a set of slices that can index the parameters of each part"""
        self.param_slices = []
        count = 0
        for p in self.parts:
            self.param_slices.append(slice(count,count+p.Nparam))
            count += p.Nparam

    def _process_slices(self,slices1=None,slices2=None):
        """
        Format the slices so that they can easily be used.
        Both slices can be any of three things:
         - If None, the new points covary through every kernel part (default)
         - If a list of slices, the i^th slice specifies which data are affected by the i^th kernel part
         - If a list of booleans, specifying which kernel parts are active

        if the second arg is False, return only slices1

        returns actual lists of slice objects
        """
        if slices1 is None:
            slices1 = [slice(None)]*self.Nparts
        elif all([type(s_i) is bool for s_i in slices1]):
            slices1 = [slice(None) if s_i else slice(0) for s_i in slices1]
        else:
            assert all([type(s_i) is slice for s_i in slices1]), "invalid slice objects"
        if slices2 is None:
            slices2 = [slice(None)]*self.Nparts
        elif slices2 is False:
            return slices1
        elif all([type(s_i) is bool for s_i in slices2]):
            slices2 = [slice(None) if s_i else slice(0) for s_i in slices2]
        else:
            assert all([type(s_i) is slice for s_i in slices2]), "invalid slice objects"
        return slices1, slices2

    def __add__(self,other):
        assert self.D == other.D
        newkern =  kern(self.D,self.parts+other.parts, self.input_slices + other.input_slices)
        #transfer constraints:
        newkern.constrained_positive_indices = np.hstack((self.constrained_positive_indices, self.Nparam + other.constrained_positive_indices))
        newkern.constrained_negative_indices = np.hstack((self.constrained_negative_indices, self.Nparam + other.constrained_negative_indices))
        newkern.constrained_bounded_indices = self.constrained_bounded_indices + [self.Nparam + x for x in other.constrained_bounded_indices]
        newkern.constrained_bounded_lowers = self.constrained_bounded_lowers + other.constrained_bounded_lowers
        newkern.constrained_bounded_uppers = self.constrained_bounded_uppers + other.constrained_bounded_uppers
        newkern.constrained_fixed_indices = self.constrained_fixed_indices + [self.Nparam + x for x in other.constrained_fixed_indices]
        newkern.constrained_fixed_values = self.constrained_fixed_values + other.constrained_fixed_values
        newkern.tied_indices = self.tied_indices + [self.Nparam + x for x in other.tied_indices]
        return newkern

    def add(self,other):
        """
        Add another kernel to this one. Both kernels are defined on the same _space_
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        return self + other

    def add_orthogonal(self,other):
        """
        Add another kernel to this one. Both kernels are defined on separate spaces
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        #deal with input slices
        D = self.D + other.D
        self_input_slices = [slice(*sl.indices(self.D)) for sl in self.input_slices]
        other_input_indices = [sl.indices(other.D) for sl in other.input_slices]
        other_input_slices = [slice(i[0]+self.D,i[1]+self.D,i[2]) for i in other_input_indices]

        newkern = kern(D, self.parts + other.parts, self_input_slices + other_input_slices)

        #transfer constraints:
        newkern.constrained_positive_indices = np.hstack((self.constrained_positive_indices, self.Nparam + other.constrained_positive_indices))
        newkern.constrained_negative_indices = np.hstack((self.constrained_negative_indices, self.Nparam + other.constrained_negative_indices))
        newkern.constrained_bounded_indices = self.constrained_bounded_indices + [self.Nparam + x for x in other.constrained_bounded_indices]
        newkern.constrained_bounded_lowers = self.constrained_bounded_lowers + other.constrained_bounded_lowers
        newkern.constrained_bounded_uppers = self.constrained_bounded_uppers + other.constrained_bounded_uppers
        newkern.constrained_fixed_indices = self.constrained_fixed_indices + [self.Nparam + x for x in other.constrained_fixed_indices]
        newkern.constrained_fixed_values = self.constrained_fixed_values + other.constrained_fixed_values
        newkern.tied_indices = self.tied_indices + [self.Nparam + x for x in other.tied_indices]
        return newkern

    def __mul__(self,other):
        """
        Shortcut for `prod_orthogonal`. Note that `+` assumes that we sum 2 kernels defines on the same space whereas `*` assumes that the kernels are defined on different subspaces.
        """
        return self.prod(other)

    def prod(self,other):
        """
        multiply two kernels defined on the same spaces.
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        K1 = self.copy()
        K2 = other.copy()

        newkernparts = [product(k1,k2) for k1, k2 in itertools.product(K1.parts,K2.parts)]

        slices = []
        for sl1, sl2 in itertools.product(K1.input_slices,K2.input_slices):
            s1, s2 = [False]*K1.D, [False]*K2.D
            s1[sl1], s2[sl2] = [True], [True]
            slices += [s1+s2]

        newkern = kern(K1.D, newkernparts, slices)
        newkern._follow_constrains(K1,K2)

        return newkern

    def prod_orthogonal(self,other):
        """
        multiply two kernels. Both kernels are defined on separate spaces.
        :param other: the other kernel to be added
        :type other: GPy.kern
        """
        K1 = self.copy()
        K2 = other.copy()

        newkernparts = [product_orthogonal(k1,k2) for k1, k2 in itertools.product(K1.parts,K2.parts)]

        slices = []
        for sl1, sl2 in itertools.product(K1.input_slices,K2.input_slices):
            s1, s2 = [False]*K1.D, [False]*K2.D
            s1[sl1], s2[sl2] = [True], [True]
            slices += [s1+s2]

        newkern = kern(K1.D + K2.D, newkernparts, slices)
        newkern._follow_constrains(K1,K2)

        return newkern

    def _follow_constrains(self,K1,K2):

        # Build the array that allows to go from the initial indices of the param to the new ones
        K1_param = []
        n = 0
        for k1 in K1.parts:
            K1_param += [range(n,n+k1.Nparam)]
            n += k1.Nparam
        n = 0
        K2_param = []
        for k2 in K2.parts:
            K2_param += [range(K1.Nparam+n,K1.Nparam+n+k2.Nparam)]
            n += k2.Nparam
        index_param = []
        for p1 in K1_param:
            for p2 in K2_param:
                index_param += p1 + p2
        index_param = np.array(index_param)

        # Get the ties and constrains of the kernels before the multiplication
        prev_ties = K1.tied_indices + [arr + K1.Nparam for arr in K2.tied_indices]

        prev_constr_pos = np.append(K1.constrained_positive_indices, K1.Nparam + K2.constrained_positive_indices)
        prev_constr_neg = np.append(K1.constrained_negative_indices, K1.Nparam + K2.constrained_negative_indices)

        prev_constr_fix = K1.constrained_fixed_indices + [arr + K1.Nparam for arr in K2.constrained_fixed_indices]
        prev_constr_fix_values = K1.constrained_fixed_values + K2.constrained_fixed_values

        prev_constr_bou = K1.constrained_bounded_indices + [arr + K1.Nparam for arr in K2.constrained_bounded_indices]
        prev_constr_bou_low = K1.constrained_bounded_lowers + K2.constrained_bounded_lowers
        prev_constr_bou_upp = K1.constrained_bounded_uppers + K2.constrained_bounded_uppers

        # follow the previous ties
        for arr in prev_ties:
            for j in arr:
                index_param[np.where(index_param==j)[0]] = arr[0]

        # ties and constrains
        for i in range(K1.Nparam + K2.Nparam):
            index = np.where(index_param==i)[0]
            if index.size > 1:
                self.tie_param(index)
        for i in prev_constr_pos:
            self.constrain_positive(np.where(index_param==i)[0])
        for i in prev_constr_neg:
            self.constrain_neg(np.where(index_param==i)[0])
        for j, i in enumerate(prev_constr_fix):
            self.constrain_fixed(np.where(index_param==i)[0],prev_constr_fix_values[j])
        for j, i in enumerate(prev_constr_bou):
            self.constrain_bounded(np.where(index_param==i)[0],prev_constr_bou_low[j],prev_constr_bou_upp[j])

    def _get_params(self):
        return np.hstack([p._get_params() for p in self.parts])

    def _set_params(self,x):
        [p._set_params(x[s]) for p, s in zip(self.parts, self.param_slices)]

    def _get_param_names(self):
        #this is a bit nasty: we wat to distinguish between parts with the same name by appending a count
        part_names = np.array([k.name for k in self.parts],dtype=np.str)
        counts = [np.sum(part_names==ni) for i, ni in enumerate(part_names)]
        cum_counts = [np.sum(part_names[i:]==ni) for i, ni in enumerate(part_names)]
        names = [name+'_'+str(cum_count) if count>1 else name for name,count,cum_count in zip(part_names,counts,cum_counts)]

        return sum([[name+'_'+n for n in k._get_param_names()] for name,k in zip(names,self.parts)],[])

    def K(self,X,X2=None,slices1=None,slices2=None):
        assert X.shape[1]==self.D
        slices1, slices2 = self._process_slices(slices1,slices2)
        if X2 is None:
            X2 = X
        target = np.zeros((X.shape[0],X2.shape[0]))
        [p.K(X[s1,i_s],X2[s2,i_s],target=target[s1,s2]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]
        return target

    def dK_dtheta(self,partial,X,X2=None,slices1=None,slices2=None):
        """
        :param partial: An array of partial derivaties, dL_dK
        :type partial: Np.ndarray (N x M)
        :param X: Observed data inputs
        :type X: np.ndarray (N x D)
        :param X2: Observed dara inputs (optional, defaults to X)
        :type X2: np.ndarray (M x D)
        :param slices1: a slice object for each kernel part, describing which data are affected by each kernel part
        :type slices1: list of slice objects, or list of booleans
        :param slices2: slices for X2
        """
        assert X.shape[1]==self.D
        slices1, slices2 = self._process_slices(slices1,slices2)
        if X2 is None:
            X2 = X
        target = np.zeros(self.Nparam)
        [p.dK_dtheta(partial[s1,s2],X[s1,i_s],X2[s2,i_s],target[ps]) for p,i_s,ps,s1,s2 in zip(self.parts, self.input_slices, self.param_slices, slices1, slices2)]

        return self._transform_gradients(target)

    def dK_dX(self,partial,X,X2=None,slices1=None,slices2=None):
        if X2 is None:
            X2 = X
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros_like(X)
        [p.dK_dX(partial[s1,s2],X[s1,i_s],X2[s2,i_s],target[s1,i_s]) for p, i_s, s1, s2 in zip(self.parts, self.input_slices, slices1, slices2)]
        return target

    def Kdiag(self,X,slices=None):
        assert X.shape[1]==self.D
        slices = self._process_slices(slices,False)
        target = np.zeros(X.shape[0])
        [p.Kdiag(X[s,i_s],target=target[s]) for p,i_s,s in zip(self.parts,self.input_slices,slices)]
        return target

    def dKdiag_dtheta(self,partial,X,slices=None):
        assert X.shape[1]==self.D
        assert len(partial.shape)==1
        assert partial.size==X.shape[0]
        slices = self._process_slices(slices,False)
        target = np.zeros(self.Nparam)
        [p.dKdiag_dtheta(partial[s],X[s,i_s],target[ps]) for p,i_s,s,ps in zip(self.parts,self.input_slices,slices,self.param_slices)]
        return self._transform_gradients(target)

    def dKdiag_dX(self, partial, X, slices=None):
        assert X.shape[1]==self.D
        slices = self._process_slices(slices,False)
        target = np.zeros_like(X)
        [p.dKdiag_dX(partial[s],X[s,i_s],target[s,i_s]) for p,i_s,s in zip(self.parts,self.input_slices,slices)]
        return target

    def psi0(self,Z,mu,S,slices=None):
        slices = self._process_slices(slices,False)
        target = np.zeros(mu.shape[0])
        [p.psi0(Z,mu[s],S[s],target[s]) for p,s in zip(self.parts,slices)]
        return target

    def dpsi0_dtheta(self,partial,Z,mu,S,slices=None):
        slices = self._process_slices(slices,False)
        target = np.zeros(self.Nparam)
        [p.dpsi0_dtheta(partial[s],Z,mu[s],S[s],target[ps]) for p,ps,s in zip(self.parts, self.param_slices,slices)]
        return self._transform_gradients(target)

    def dpsi0_dmuS(self,partial,Z,mu,S,slices=None):
        slices = self._process_slices(slices,False)
        target_mu,target_S = np.zeros_like(mu),np.zeros_like(S)
        [p.dpsi0_dmuS(partial,Z,mu[s],S[s],target_mu[s],target_S[s]) for p,s in zip(self.parts,slices)]
        return target_mu,target_S

    def psi1(self,Z,mu,S,slices1=None,slices2=None):
        """Think N,M,Q """
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros((mu.shape[0],Z.shape[0]))
        [p.psi1(Z[s2],mu[s1],S[s1],target[s1,s2]) for p,s1,s2 in zip(self.parts,slices1,slices2)]
        return target

    def dpsi1_dtheta(self,partial,Z,mu,S,slices1=None,slices2=None):
        """N,M,(Ntheta)"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros((self.Nparam))
        [p.dpsi1_dtheta(partial[s2,s1],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[ps]) for p,ps,s1,s2,i_s in zip(self.parts, self.param_slices,slices1,slices2,self.input_slices)]
        return self._transform_gradients(target)

    def dpsi1_dZ(self,partial,Z,mu,S,slices1=None,slices2=None):
        """N,M,Q"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros_like(Z)
        [p.dpsi1_dZ(partial[s2,s1],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[s2,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]
        return target

    def dpsi1_dmuS(self,partial,Z,mu,S,slices1=None,slices2=None):
        """return shapes are N,M,Q"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target_mu, target_S = np.zeros((2,mu.shape[0],mu.shape[1]))
        [p.dpsi1_dmuS(partial[s2,s1],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target_mu[s1,i_s],target_S[s1,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]
        return target_mu, target_S

    def psi2(self,Z,mu,S,slices1=None,slices2=None):
        """
        :Z: np.ndarray of inducing inputs (M x Q)
        : mu, S: np.ndarrays of means and variacnes (each N x Q)
        :returns psi2: np.ndarray (N,M,M,Q) """
        target = np.zeros((mu.shape[0],Z.shape[0],Z.shape[0]))
        slices1, slices2 = self._process_slices(slices1,slices2)
        [p.psi2(Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[s1,s2,s2]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]

        #compute the "cross" terms
        for p1, p2 in itertools.combinations(self.parts,2):
            #white doesn;t compine with anything
            if p1.name=='white' or p2.name=='white':
                pass
            #rbf X bias
            elif p1.name=='bias' and p2.name=='rbf':
                target += p1.variance*(p2._psi1[:,:,None]+p2._psi1[:,None,:])
            elif p2.name=='bias' and p1.name=='rbf':
                target += p2.variance*(p1._psi1[:,:,None]+p1._psi1[:,None,:])
            #rbf X linear
            elif p1.name=='linear' and p2.name=='rbf':
                raise NotImplementedError #TODO
            elif p2.name=='linear' and p1.name=='rbf':
                raise NotImplementedError #TODO
            else:
                raise NotImplementedError, "psi2 cannot be computed for this kernel"





        # "crossterms". Here we are recomputing psi1 for white (we don't need to), but it's
        # not really expensive, since it's just a matrix of zeroes.
        # psi1_matrices = [np.zeros((mu.shape[0], Z.shape[0])) for p in self.parts]
        # [p.psi1(Z[s2],mu[s1],S[s1],psi1_target[s1,s2]) for p,s1,s2,psi1_target in zip(self.parts,slices1,slices2, psi1_matrices)]

        crossterms = 0.0
        # for 3 kernels this returns something like
        # [(0,1), (0,2), (1,2)]
        # in theory, we should also account for (1,0), (2,0) and so on, but
        # the transpose deals exactly with that
        # for a,b in itertools.combinations(psi1_matrices, 2):
        #     tmp = np.multiply(a,b)
        #     crossterms += tmp[:,None,:] + tmp[:, :,None]

        return target + crossterms

    def dpsi2_dtheta(self,partial,partial1,Z,mu,S,slices1=None,slices2=None):
        """Returns shape (N,M,M,Ntheta)"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros(self.Nparam)
        [p.dpsi2_dtheta(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[ps]) for p,i_s,s1,s2,ps in zip(self.parts,self.input_slices,slices1,slices2,self.param_slices)]

        #compute the "cross" terms
        #TODO: better looping
        for i1, i2 in itertools.combinations(range(len(self.parts)),2):
            p1,p2 = self.parts[i1], self.parts[i2]
            ipsl1, ipsl2 = self.input_slices[i1], self.input_slices[i2]
            ps1, ps2 = self.param_slices[i1], self.param_slices[i2]

            #white doesn;t compine with anything
            if p1.name=='white' or p2.name=='white':
                pass
            #rbf X bias
            elif p1.name=='bias' and p2.name=='rbf':
                p2.dpsi1_dtheta(partial.sum(1)*p1.variance,Z,mu,S,target[ps2])
                p1.dpsi1_dtheta(partial.sum(1)*p2._psi1,Z,mu,S,target[ps1])
            elif p2.name=='bias' and p1.name=='rbf':
                p1.dpsi1_dtheta(partial.sum(1)*p2.variance,Z,mu,S,target[ps1])
                p2.dpsi1_dtheta(partial.sum(1)*p1._psi1,Z,mu,S,target[ps2])
            #rbf X linear
            elif p1.name=='linear' and p2.name=='rbf':
                raise NotImplementedError #TODO
            elif p2.name=='linear' and p1.name=='rbf':
                raise NotImplementedError #TODO
            else:
                raise NotImplementedError, "psi2 cannot be computed for this kernel"

        # # "crossterms"
        # # 1. get all the psi1 statistics
        # psi1_matrices = [np.zeros((mu.shape[0], Z.shape[0])) for p in self.parts]
        # [p.psi1(Z[s2],mu[s1],S[s1],psi1_target[s1,s2]) for p,s1,s2,psi1_target in zip(self.parts,slices1,slices2, psi1_matrices)]

        # partial1 = np.ones_like(partial1)
        # # 2. get all the dpsi1/dtheta gradients
        # psi1_gradients = [np.zeros(self.Nparam) for p in self.parts]
        # [p.dpsi1_dtheta(partial1[s2,s1],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],psi1g_target[ps]) for p,ps,s1,s2,i_s,psi1g_target in zip(self.parts, self.param_slices,slices1,slices2,self.input_slices,psi1_gradients)]


        # # 3. multiply them somehow
        # for a,b in itertools.combinations(range(len(psi1_matrices)), 2):

        #     tmp = (psi1_gradients[a][None, None] * psi1_matrices[b][:,:, None])
        #     # target += (tmp[None] + tmp[:,None]).sum(0).sum(0).sum(0)
        #     # gne = (psi1_gradients[a].sum()*psi1_matrices[b].sum())
        #     # target += gne
        #     #target += (gne[None] + gne[:, None]).sum(0)
        #     target += (partial.sum(0)[:,:,None] * (tmp[:, None] + tmp[:,:,None]).sum(0)).sum(0).sum(0)
        return self._transform_gradients(target)

    def dpsi2_dZ(self,partial,Z,mu,S,slices1=None,slices2=None):
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros_like(Z)
        [p.dpsi2_dZ(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[s2,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]

        #compute the "cross" terms
        #TODO: slices (need to iterate around the input slices also...)
        for p1, p2 in itertools.combinations(self.parts,2):
            #white doesn;t compine with anything
            if p1.name=='white' or p2.name=='white':
                pass
            #rbf X bias
            elif p1.name=='bias' and p2.name=='rbf':
                target += p2.dpsi1_dX(partial.sum(1)*p1.variance,Z,mu,S)
            elif p2.name=='bias' and p1.name=='rbf':
                target += p1.dpsi1_dZ(partial.sum(2)*p2.variance,Z,mu,S)
            #rbf X linear
            elif p1.name=='linear' and p2.name=='rbf':
                raise NotImplementedError #TODO
            elif p2.name=='linear' and p1.name=='rbf':
                raise NotImplementedError #TODO
            else:
                raise NotImplementedError, "psi2 cannot be computed for this kernel"


        return target

    def dpsi2_dmuS(self,partial,Z,mu,S,slices1=None,slices2=None):
        """return shapes are N,M,M,Q"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target_mu, target_S = np.zeros((2,mu.shape[0],mu.shape[1]))
        [p.dpsi2_dmuS(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target_mu[s1,i_s],target_S[s1,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]

        #TODO: there are some extra terms to compute here!
        return target_mu, target_S

    def plot(self, x = None, plot_limits=None,which_functions='all',resolution=None,*args,**kwargs):
        if which_functions=='all':
            which_functions = [True]*self.Nparts
        if self.D == 1:
            if x is None:
                x = np.zeros((1,1))
            else:
                x = np.asarray(x)
                assert x.size == 1, "The size of the fixed variable x is not 1"
                x = x.reshape((1,1))

            if plot_limits == None:
                xmin, xmax = (x-5).flatten(), (x+5).flatten()
            elif len(plot_limits) == 2:
                xmin, xmax = plot_limits
            else:
                raise ValueError, "Bad limits for plotting"

            Xnew = np.linspace(xmin,xmax,resolution or 201)[:,None]
            Kx = self.K(Xnew,x,slices2=which_functions)
            pb.plot(Xnew,Kx,*args,**kwargs)
            pb.xlim(xmin,xmax)
            pb.xlabel("x")
            pb.ylabel("k(x,%0.1f)" %x)

        elif self.D == 2:
            if x is None:
                x = np.zeros((1,2))
            else:
                x = np.asarray(x)
                assert x.size == 2, "The size of the fixed variable x is not 2"
                x = x.reshape((1,2))

            if plot_limits == None:
                xmin, xmax = (x-5).flatten(), (x+5).flatten()
            elif len(plot_limits) == 2:
                xmin, xmax = plot_limits
            else:
                raise ValueError, "Bad limits for plotting"

            resolution = resolution or 51
            xx,yy = np.mgrid[xmin[0]:xmax[0]:1j*resolution,xmin[1]:xmax[1]:1j*resolution]
            xg = np.linspace(xmin[0],xmax[0],resolution)
            yg = np.linspace(xmin[1],xmax[1],resolution)
            Xnew = np.vstack((xx.flatten(),yy.flatten())).T
            Kx = self.K(Xnew,x,slices2=which_functions)
            Kx = Kx.reshape(resolution,resolution).T
            pb.contour(xg,yg,Kx,vmin=Kx.min(),vmax=Kx.max(),cmap=pb.cm.jet,*args,**kwargs)
            pb.xlim(xmin[0],xmax[0])
            pb.ylim(xmin[1],xmax[1])
            pb.xlabel("x1")
            pb.ylabel("x2")
            pb.title("k(x1,x2 ; %0.1f,%0.1f)" %(x[0,0],x[0,1]) )
        else:
            raise NotImplementedError, "Cannot plot a kernel with more than two input dimensions"
