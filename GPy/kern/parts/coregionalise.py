# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
from GPy.util.linalg import mdot, pdinv
import pdb
from scipy import weave

class Coregionalise(Kernpart):
    """
    Coregionalisation kernel. 

    Used for computing covariance functions of the form
    .. math::
       k_2(x, y)=B k(x, y)
    where
    .. math::
       B = WW^\top + diag(kappa)

    :param output_dim: the number of output dimensions
    :type output_dim: int
    :param rank: the rank of the coregionalisation matrix.
    :type rank: int
    :param W: a low rank matrix that determines the correlations between the different outputs, together with kappa it forms the coregionalisation matrix B.
    :type W: ndarray
    :param kappa: a diagonal term which allows the outputs to behave independently.
    :rtype: kernel object

    .. Note: see coregionalisation examples in GPy.examples.regression for some usage.
    """
    def __init__(self,output_dim,rank=1, W=None, kappa=None):
        self.input_dim = 1
        self.name = 'coregion'
        self.output_dim = output_dim
        self.rank = rank
        if W is None:
            self.W = 0.5*np.random.randn(self.output_dim,self.rank)/np.sqrt(self.rank)
        else:
            assert W.shape==(self.output_dim,self.rank)
            self.W = W
        if kappa is None:
            kappa = 0.5*np.ones(self.output_dim)
        else:
            assert kappa.shape==(self.output_dim,)
        self.kappa = kappa
        self.num_params = self.output_dim*(self.rank + 1)
        self._set_params(np.hstack([self.W.flatten(),self.kappa]))

    def _get_params(self):
        return np.hstack([self.W.flatten(),self.kappa])

    def _set_params(self,x):
        assert x.size == self.num_params
        self.kappa = x[-self.output_dim:]
        self.W = x[:-self.output_dim].reshape(self.output_dim,self.rank)
        self.B = np.dot(self.W,self.W.T) + np.diag(self.kappa)

    def _get_param_names(self):
        return sum([['W%i_%i'%(i,j) for j in range(self.rank)] for i in range(self.output_dim)],[]) + ['kappa_%i'%i for i in range(self.output_dim)]

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
              target[i+i*N] += B[index[i]+output_dim*index[i]];
              for(int j=0; j<i; j++){
                  target[j+i*N] += B[index[i]+output_dim*index[j]];
                  target[i+j*N] += target[j+i*N];
                }
              }
            """
            N,B,output_dim = index.size, self.B, self.output_dim
            weave.inline(code,['target','index','N','B','output_dim'])
        else:
            index2 = np.asarray(index2,dtype=np.int)
            code="""
            for(int i=0;i<num_inducing; i++){
              for(int j=0; j<N; j++){
                  target[i+j*num_inducing] += B[output_dim*index[j]+index2[i]];
                }
              }
            """
            N,num_inducing,B,output_dim = index.size,index2.size, self.B, self.output_dim
            weave.inline(code,['target','index','index2','N','num_inducing','B','output_dim'])


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
            dL_dK_small[index[j] + output_dim*index2[i]] += dL_dK[i+j*num_inducing];
          }
        }
        """
        N, num_inducing, output_dim = index.size, index2.size, self.output_dim
        weave.inline(code, ['N','num_inducing','output_dim','dL_dK','dL_dK_small','index','index2'])

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
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                tmp = np.sum(dL_dK[(ii==i)*(jj==j)])
                dL_dK_small[i,j] = tmp

        dkappa = np.diag(dL_dK_small)
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:,None,:]*dL_dK_small[:,:,None]).sum(0)

        target += np.hstack([dW.flatten(),dkappa])

    def dKdiag_dtheta(self,dL_dKdiag,index,target):
        index = np.asarray(index,dtype=np.int).flatten()
        dL_dKdiag_small = np.zeros(self.output_dim)
        for i in range(self.output_dim):
            dL_dKdiag_small[i] += np.sum(dL_dKdiag[index==i])
        dW = 2.*self.W*dL_dKdiag_small[:,None]
        dkappa = dL_dKdiag_small
        target += np.hstack([dW.flatten(),dkappa])

    def dK_dX(self,dL_dK,X,X2,target):
        pass



