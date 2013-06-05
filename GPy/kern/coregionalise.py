# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
from GPy.util.linalg import mdot, pdinv
import pdb
from scipy import weave

class Coregionalise(Kernpart):
    """
    Kernel for Intrinsic Corregionalization Models
    """
    def __init__(self,Nout,R=1, W=None, kappa=None):
        self.input_dim = 1
        self.name = 'coregion'
        self.Nout = Nout
        self.R = R
        if W is None:
            self.W = np.ones((self.Nout,self.R))
        else:
            assert W.shape==(self.Nout,self.R)
            self.W = W
        if kappa is None:
            kappa = np.ones(self.Nout)
        else:
            assert kappa.shape==(self.Nout,)
        self.kappa = kappa
        self.num_params = self.Nout*(self.R + 1)
        self._set_params(np.hstack([self.W.flatten(),self.kappa]))

    def _get_params(self):
        return np.hstack([self.W.flatten(),self.kappa])

    def _set_params(self,x):
        assert x.size == self.num_params
        self.kappa = x[-self.Nout:]
        self.W = x[:-self.Nout].reshape(self.Nout,self.R)
        self.B = np.dot(self.W,self.W.T) + np.diag(self.kappa)

    def _get_param_names(self):
        return sum([['W%i_%i'%(i,j) for j in range(self.R)] for i in range(self.Nout)],[]) + ['kappa_%i'%i for i in range(self.Nout)]

    def K(self,index,index2,target):
        index = np.asarray(index,dtype=np.int)

        #here's the old code (numpy)
        #if index2 is None:
            #index2 = index
        #else:
            #index2 = np.asarray(index2,dtype=np.int)
        #false_target = target.copy()
        #ii,jj = np.meshgrid(index,index2)
        #ii,jj = ii.T, jj.T
        #false_target += self.B[ii,jj]

        if index2 is None:
            code="""
            for(int i=0;i<N; i++){
              target[i+i*N] += B[index[i]+Nout*index[i]];
              for(int j=0; j<i; j++){
                  target[j+i*N] += B[index[i]+Nout*index[j]];
                  target[i+j*N] += target[j+i*N];
                }
              }
            """
            N,B,Nout = index.size, self.B, self.Nout
            weave.inline(code,['target','index','N','B','Nout'])
        else:
            index2 = np.asarray(index2,dtype=np.int)
            code="""
            for(int i=0;i<num_inducing; i++){
              for(int j=0; j<N; j++){
                  target[i+j*num_inducing] += B[Nout*index[j]+index2[i]];
                }
              }
            """
            N,num_inducing,B,Nout = index.size,index2.size, self.B, self.Nout
            weave.inline(code,['target','index','index2','N','num_inducing','B','Nout'])


    def Kdiag(self,index,target):
        target += np.diag(self.B)[np.asarray(index,dtype=np.int).flatten()]

    def dK_dtheta(self,dL_dK,index,index2,target):
        index = np.asarray(index,dtype=np.int)
        dL_dK_small = np.zeros_like(self.B)
        if index2 is None:
            index2 = index
        else:
            index2 = np.asarray(index2,dtype=np.int)

        code="""
        for(int i=0; i<num_inducing; i++){
          for(int j=0; j<N; j++){
            dL_dK_small[index[j] + Nout*index2[i]] += dL_dK[i+j*num_inducing];
          }
        }
        """
        N, num_inducing, Nout = index.size, index2.size, self.Nout
        weave.inline(code, ['N','num_inducing','Nout','dL_dK','dL_dK_small','index','index2'])

        dkappa = np.diag(dL_dK_small)
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:,None,:]*dL_dK_small[:,:,None]).sum(0)

        target += np.hstack([dW.flatten(),dkappa])

    def dK_dtheta_old(self,dL_dK,index,index2,target):
        if index2 is None:
            index2 = index
        else:
            index2 = np.asarray(index2,dtype=np.int)
        ii,jj = np.meshgrid(index,index2)
        ii,jj = ii.T, jj.T

        dL_dK_small = np.zeros_like(self.B)
        for i in range(self.Nout):
            for j in range(self.Nout):
                tmp = np.sum(dL_dK[(ii==i)*(jj==j)])
                dL_dK_small[i,j] = tmp

        dkappa = np.diag(dL_dK_small)
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:,None,:]*dL_dK_small[:,:,None]).sum(0)

        target += np.hstack([dW.flatten(),dkappa])

    def dKdiag_dtheta(self,dL_dKdiag,index,target):
        index = np.asarray(index,dtype=np.int).flatten()
        dL_dKdiag_small = np.zeros(self.Nout)
        for i in range(self.Nout):
            dL_dKdiag_small[i] += np.sum(dL_dKdiag[index==i])
        dW = 2.*self.W*dL_dKdiag_small[:,None]
        dkappa = dL_dKdiag_small
        target += np.hstack([dW.flatten(),dkappa])

    def dK_dX(self,dL_dK,X,X2,target):
        pass



