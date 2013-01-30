# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


import numpy as np
from ..core.parameterised import parameterised
from functools import partial
from kernpart import kernpart
import itertools

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

    def get_param(self):
        return np.hstack([p.get_param() for p in self.parts])

    def set_param(self,x):
        [p.set_param(x[s]) for p, s in zip(self.parts, self.param_slices)]

    def get_param_names(self):
        #this is a bit nasty: we wat to distinguish between parts with the same name by appending a count
        part_names = np.array([k.name for k in self.parts],dtype=np.str)
        counts = [np.sum(part_names==ni) for i, ni in enumerate(part_names)]
        cum_counts = [np.sum(part_names[i:]==ni) for i, ni in enumerate(part_names)]
        names = [name+'_'+str(cum_count) if count>1 else name for name,count,cum_count in zip(part_names,counts,cum_counts)]

        return sum([[name+'_'+n for n in k.get_param_names()] for name,k in zip(names,self.parts)],[])

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
        return target

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
        return target

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
        return target

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
        return target

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

        # MASSIVE TODO: do something smart for white
        # "crossterms"
        psi1_matrices = [np.zeros((mu.shape[0], Z.shape[0])) for p in self.parts]
        [p.psi1(Z[s2],mu[s1],S[s1],psi1_target[s1,s2]) for p,s1,s2,psi1_target in zip(self.parts,slices1,slices2, psi1_matrices)]
        for a,b in itertools.combinations(psi1_matrices, 2):
            tmp = np.multiply(a,b)
            target += tmp[:,None,:] + tmp[:, :,None]

        return target

    def dpsi2_dtheta(self,partial,partial1,Z,mu,S,slices1=None,slices2=None):
        """Returns shape (N,M,M,Ntheta)"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros(self.Nparam)
        [p.dpsi2_dtheta(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[ps]) for p,i_s,s1,s2,ps in zip(self.parts,self.input_slices,slices1,slices2,self.param_slices)]


        # "crossterms"
        # 1. get all the psi1 statistics
        psi1_matrices = [np.zeros((mu.shape[0], Z.shape[0])) for p in self.parts]
        [p.psi1(Z[s2],mu[s1],S[s1],psi1_target[s1,s2]) for p,s1,s2,psi1_target in zip(self.parts,slices1,slices2, psi1_matrices)]
        # 2. get all the dpsi1/dtheta gradients
        psi1_gradients = [np.zeros(self.Nparam) for p in self.parts]
        [p.dpsi1_dtheta(partial1[s2,s1],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],psi1g_target[ps]) for p,ps,s1,s2,i_s,psi1g_target in zip(self.parts, self.param_slices,slices1,slices2,self.input_slices,psi1_gradients)]

        # 3. multiply them somehow
        for a,b in itertools.combinations(range(len(psi1_matrices)), 2):
            gne = (psi1_gradients[a][None]*psi1_matrices[b].sum(0)[:,None]).sum(0)

            target += 0#(gne[None] + gne[:, None]).sum(0)
        return target

    def dpsi2_dZ(self,partial,Z,mu,S,slices1=None,slices2=None):
        slices1, slices2 = self._process_slices(slices1,slices2)
        target = np.zeros_like(Z)
        [p.dpsi2_dZ(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target[s2,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]

        return target

    def dpsi2_dmuS(self,partial,Z,mu,S,slices1=None,slices2=None):
        """return shapes are N,M,M,Q"""
        slices1, slices2 = self._process_slices(slices1,slices2)
        target_mu, target_S = np.zeros((2,mu.shape[0],mu.shape[1]))
        [p.dpsi2_dmuS(partial[s1,s2,s2],Z[s2,i_s],mu[s1,i_s],S[s1,i_s],target_mu[s1,i_s],target_S[s1,i_s]) for p,i_s,s1,s2 in zip(self.parts,self.input_slices,slices1,slices2)]

        #TODO: there are some extra terms to compute here!
        return target_mu, target_S
