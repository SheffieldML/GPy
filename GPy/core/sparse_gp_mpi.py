# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .sparse_gp import SparseGP
from numpy.linalg.linalg import LinAlgError
from ..inference.latent_function_inference.var_dtc_parallel import update_gradients, VarDTC_minibatch

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

    def __init__(self, X, Y, Z, kernel, likelihood, variational_prior=None,
                 mean_function=None, inference_method=None, name='sparse gp',
                 Y_metadata=None, mpi_comm=None, normalizer=False):
        self._IN_OPTIMIZATION_ = False
        if mpi_comm != None:
            if inference_method is None:
                inference_method = VarDTC_minibatch(mpi_comm=mpi_comm)
            else:
                assert isinstance(inference_method, VarDTC_minibatch), 'inference_method has to support MPI!'

        super(SparseGP_MPI, self).__init__(X, Y, Z, kernel, likelihood, inference_method=inference_method, mean_function=mean_function, name=name, Y_metadata=Y_metadata, normalizer=normalizer)
        self.update_model(False)

        if variational_prior is not None:
            self.link_parameter(variational_prior)

        self.mpi_comm = mpi_comm
        # Manage the data (Y) division
        if mpi_comm != None:
            from ..util.parallel import divide_data
            N_start, N_end, N_list = divide_data(Y.shape[0], mpi_comm.rank, mpi_comm.size)
            self.N_range = (N_start, N_end)
            self.N_list = np.array(N_list)
            self.Y_local = self.Y[N_start:N_end]
            print('MPI RANK '+str(self.mpi_comm.rank)+' with the data range '+str(self.N_range))
            mpi_comm.Bcast(self.param_array, root=0)
        self.update_model(True)

    def __getstate__(self):
        dc = super(SparseGP_MPI, self).__getstate__()
        dc['mpi_comm'] = None
        if self.mpi_comm != None:
            del dc['N_range']
            del dc['N_list']
            del dc['Y_local']
        if 'normalizer' not in dc:
            dc['normalizer'] = None
            dc['Y_normalized'] = dc['Y']
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
            ret = super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
        elif self.mpi_comm.rank==0:
            ret = super(SparseGP_MPI, self).optimize(optimizer,start,**kwargs)
            self.mpi_comm.Bcast(np.int32(-1),root=0)
        elif self.mpi_comm.rank>0:
            x = self.optimizer_array.copy()
            flag = np.empty(1,dtype=np.int32)
            while True:
                self.mpi_comm.Bcast(flag,root=0)
                if flag==1:
                    try:
                        self.optimizer_array = x
                        self._fail_count = 0
                    except (LinAlgError, ZeroDivisionError, ValueError):
                        if self._fail_count >= self._allowed_failures:
                            raise
                        self._fail_count += 1
                elif flag==-1:
                    ret = None
                    break
                else:
                    self._IN_OPTIMIZATION_ = False
                    raise Exception("Unrecognizable flag for synchronization!")
        self._IN_OPTIMIZATION_ = False
        return ret

    def parameters_changed(self):
        if isinstance(self.inference_method,VarDTC_minibatch):
            update_gradients(self, mpi_comm=self.mpi_comm)
        else:
            super(SparseGP_MPI,self).parameters_changed()
