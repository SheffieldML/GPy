# Copyright (c) 2012, James Hensman and Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
from GPy.util.linalg import mdot, pdinv
import pdb
from scipy import weave

class Coregionalize(Kernpart):
    """
    Covariance function for intrinsic/linear coregionalization models

    This covariance has the form
    .. math::
       \mathbf{B} = \mathbf{W}\mathbf{W}^\top + kappa \mathbf{I}

    An intrinsic/linear coregionalization covariance function of the form
    .. math::
       k_2(x, y)=\mathbf{B} k(x, y)

    it is obtained as the tensor product between a covariance function
    k(x,y) and B.

    :param num_outputs: number of outputs to coregionalize
    :type num_outputs: int
    :param W_columns: number of columns of the W matrix (this parameter is ignored if parameter W is not None)
    :type W_colunns: int
    :param W: a low rank matrix that determines the correlations between the different outputs, together with kappa it forms the coregionalization matrix B
    :type W: numpy array of dimensionality (num_outpus, W_columns)
    :param kappa: a vector which allows the outputs to behave independently
    :type kappa: numpy array of dimensionality  (num_outputs,)

    .. Note: see coregionalization examples in GPy.examples.regression for some usage.
    """
    def __init__(self,num_outputs,W_columns=1, W=None, kappa=None):
        self.input_dim = 1
        self.name = 'coregion'
        self.num_outputs = num_outputs
        self.W_columns = W_columns
        if W is None:
            self.W = 0.5*np.random.randn(self.num_outputs,self.W_columns)/np.sqrt(self.W_columns)
        else:
            assert W.shape==(self.num_outputs,self.W_columns)
            self.W = W
        if kappa is None:
            kappa = 0.5*np.ones(self.num_outputs)
        else:
            assert kappa.shape==(self.num_outputs,)
        self.kappa = kappa
        self.num_params = self.num_outputs*(self.W_columns + 1)
        self._set_params(np.hstack([self.W.flatten(),self.kappa]))

    def _get_params(self):
        return np.hstack([self.W.flatten(),self.kappa])

    def _set_params(self,x):
        assert x.size == self.num_params
        self.kappa = x[-self.num_outputs:]
        self.W = x[:-self.num_outputs].reshape(self.num_outputs,self.W_columns)
        self.B = np.dot(self.W,self.W.T) + np.diag(self.kappa)

    def _get_param_names(self):
        return sum([['W%i_%i'%(i,j) for j in range(self.W_columns)] for i in range(self.num_outputs)],[]) + ['kappa_%i'%i for i in range(self.num_outputs)]

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
              target[i+i*N] += B[index[i]+num_outputs*index[i]];
              for(int j=0; j<i; j++){
                  target[j+i*N] += B[index[i]+num_outputs*index[j]];
                  target[i+j*N] += target[j+i*N];
                }
              }
            """
            N,B,num_outputs = index.size, self.B, self.num_outputs
            weave.inline(code,['target','index','N','B','num_outputs'])
        else:
            index2 = np.asarray(index2,dtype=np.int)
            code="""
            for(int i=0;i<num_inducing; i++){
              for(int j=0; j<N; j++){
                  target[i+j*num_inducing] += B[num_outputs*index[j]+index2[i]];
                }
              }
            """
            N,num_inducing,B,num_outputs = index.size,index2.size, self.B, self.num_outputs
            weave.inline(code,['target','index','index2','N','num_inducing','B','num_outputs'])


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
            dL_dK_small[index[j] + num_outputs*index2[i]] += dL_dK[i+j*num_inducing];
          }
        }
        """
        N, num_inducing, num_outputs = index.size, index2.size, self.num_outputs
        weave.inline(code, ['N','num_inducing','num_outputs','dL_dK','dL_dK_small','index','index2'])

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
        for i in range(self.num_outputs):
            for j in range(self.num_outputs):
                tmp = np.sum(dL_dK[(ii==i)*(jj==j)])
                dL_dK_small[i,j] = tmp

        dkappa = np.diag(dL_dK_small)
        dL_dK_small += dL_dK_small.T
        dW = (self.W[:,None,:]*dL_dK_small[:,:,None]).sum(0)

        target += np.hstack([dW.flatten(),dkappa])

    def dKdiag_dtheta(self,dL_dKdiag,index,target):
        index = np.asarray(index,dtype=np.int).flatten()
        dL_dKdiag_small = np.zeros(self.num_outputs)
        for i in range(self.num_outputs):
            dL_dKdiag_small[i] += np.sum(dL_dKdiag[index==i])
        dW = 2.*self.W*dL_dKdiag_small[:,None]
        dkappa = dL_dKdiag_small
        target += np.hstack([dW.flatten(),dkappa])

    def dK_dX(self,dL_dK,X,X2,target):
        pass
