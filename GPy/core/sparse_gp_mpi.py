# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from sparse_gp import SparseGP
from parameterization.param import Param
from ..inference.latent_function_inference import var_dtc
from .. import likelihoods
from parameterization.variational import VariationalPosterior
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch
from ..core.parameterization.parameter_core import OptimizationHandlable

import logging
logger = logging.getLogger("sparse gp mpi")

class SparseGP_MPI(SparseGP):
    """
    A general purpose Sparse GP model with MPI parallelization support

    This model allows (approximate) inference using variational DTC or FITC
    (Gaussian likelihoods) as well as non-conjugate sparse methods based on
    these.

    :param X: inputs
    :type X: np.ndarray (num_data x input_dim)
    :param likelihood: a likelihood instance, containing the observed data
    :type likelihood: GPy.likelihood.(Gaussian | EP | Laplace)
    :param kernel: the kernel (covariance function). See link kernels
    :type kernel: a GPy.kern.kern instance
    :param X_variance: The uncertainty in the measurements of X (Gaussian variance)
    :type X_variance: np.ndarray (num_data x input_dim) | None
    :param Z: inducing inputs
    :type Z: np.ndarray (num_inducing x input_dim)
    :param num_inducing: Number of inducing points (optional, default 10. Ignored if Z is not None)
    :type num_inducing: int
    :param mpi_comm: The communication group of MPI, e.g. mpi4py.MPI.COMM_WORLD
    :type mpi_comm: mpi4py.MPI.Intracomm

    """

    def __init__(self, X, Y, Z, kernel, likelihood, variational_prior=None, inference_method=None, name='sparse gp mpi', Y_metadata=None, mpi_comm=None):
        self._IN_OPTIMIZATION_ = False
        if mpi_comm != None:
            if inference_method is None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                assert isinstance(inference_method, VarDTC_minibatch), 'inference_method has to support MPI!'
                        
        super(SparseGP_MPI, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, name=name, Y_metadata=Y_metadata)
        self.updates = False
        self.add_parameter(self.X, index=0)
        if variational_prior is not None:
            self.add_parameter(variational_prior)
        self.X.fix()

        self.mpi_comm = mpi_comm
        # Manage the data (Y) division
        if mpi_comm != None:
            from ..util.mpi import divide_data
            N_start, N_end, N_list = divide_data(Y.shape[0], mpi_comm)
            self.N_range = (N_start, N_end)
            self.N_list = np.array(N_list)
            self.Y_local = self.Y[N_start:N_end]
            print 'MPI RANK '+str(self.mpi_comm.rank)+' with the data range '+str(self.N_range)
            mpi_comm.Bcast(self.param_array, root=0)
        self.updates = True

    def __getstate__(self):
        dc = super(SparseGP_MPI, self).__getstate__()
        dc['mpi_comm'] = None
        if self.mpi_comm != None:
            del dc['N_range']
            del dc['N_list']
            del dc['Y_local']
        return dc

    #=====================================================
    # The MPI parallelization 
    #     - can move to model at some point
    #=====================================================
    
    @SparseGP.optimizer_array.setter
    def optimizer_array(self, p):        
        if self.mpi_comm != None:
            if self._IN_OPTIMIZATION_ and self.mpi_comm.rank==0:
                self.mpi_comm.Bcast(np.int32(1),root=0)
            self.mpi_comm.Bcast(p, root=0)
        SparseGP.optimizer_array.fset(self,p)
        
    def optimize(self, optimizer=None, start=None, **kwargs):
        self._IN_OPTIMIZATION_ = True
        if self.mpi_comm==None:
            super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=0)
        elif self.mpi_comm.rank>0:
            x = self._get_params_transformed().copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    self._set_params_transformed(x)
                elif flag==-1:
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False

    def parameters_changed(self):
        update_gradients(self, mpi_comm=self.mpi_comm)

