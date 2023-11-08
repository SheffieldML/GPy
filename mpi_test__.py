
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
            