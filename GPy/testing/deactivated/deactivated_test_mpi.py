# Copyright (c) 2013-2014, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

try:
    import subprocess

    class TestMPI:
        def test_BayesianGPLVM_MPI(self):
            code = """
import numpy as np
import GPy
from mpi4py import MPI
np.random.seed(123456)
comm = MPI.COMM_WORLD
N = 100
x = np.linspace(-6., 6., N)
y = np.sin(x) + np.random.randn(N) * 0.05
comm.Bcast(y)
data = np.vstack([x,y])
infr = GPy.inference.latent_function_inference.VarDTC_minibatch(mpi_comm=comm)
m = GPy.models.BayesianGPLVM(data.T,1,mpi_comm=comm)
m.optimize(max_iters=10)
if comm.rank==0:
    print float(m.objective_function())
    m.inference_method.mpi_comm=None
    m.mpi_comm=None
    m._trigger_params_changed()
    print float(m.objective_function())
            """
            with open("mpi_test__.py", "w") as f:
                f.write(code)
                f.close()
            p = subprocess.Popen(
                "mpirun -n 4 python mpi_test__.py", stdout=subprocess.PIPE, shell=True
            )
            (stdout, _stderr) = p.communicate()
            L1 = float(stdout.splitlines()[-2])
            L2 = float(stdout.splitlines()[-1])
            self.assertTrue(np.allclose(L1, L2))
            import os

            os.remove("mpi_test__.py")

        def test_SparseGPRegression_MPI(self):
            code = """
import numpy as np
import GPy
from mpi4py import MPI
np.random.seed(123456)
comm = MPI.COMM_WORLD
N = 100
x = np.linspace(-6., 6., N)
y = np.sin(x) + np.random.randn(N) * 0.05
comm.Bcast(y)
data = np.vstack([x,y])
#infr = GPy.inference.latent_function_inference.VarDTC_minibatch(mpi_comm=comm)
m = GPy.models.SparseGPRegression(data[:1].T,data[1:2].T,mpi_comm=comm)
m.optimize(max_iters=10)
if comm.rank==0:
    print float(m.objective_function())
    m.inference_method.mpi_comm=None
    m.mpi_comm=None
    m._trigger_params_changed()
    print float(m.objective_function())
            """
            with open("mpi_test__.py", "w") as f:
                f.write(code)
                f.close()
            p = subprocess.Popen(
                "mpirun -n 4 python mpi_test__.py", stdout=subprocess.PIPE, shell=True
            )
            (stdout, stderr) = p.communicate()
            L1 = float(stdout.splitlines()[-2])
            L2 = float(stdout.splitlines()[-1])
            assert np.allclose(L1, L2)
            import os

            os.remove("mpi_test__.py")

except:
    pass
