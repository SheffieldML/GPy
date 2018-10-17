"""
The package for scikits.cuda initialization

Global variables: initSuccess
providing CUBLAS handle: cublas_handle
"""

gpu_initialized = False
gpu_device = None
gpu_context = None
MPI_enabled = False

try:
    import pycuda.autoinit
    gpu_initialized = True
except:
    pass

# def initGPU():
#     try:
#         from mpi4py import MPI
#         MPI_enabled = True
#     except:
#         pass
#     try:
#         if MPI_enabled and MPI.COMM_WORLD.size>1:
#             from .parallel import get_id_within_node
#             gpuid = get_id_within_node()
#             import pycuda.driver
#             pycuda.driver.init()
#             if gpuid>=pycuda.driver.Device.count():
#                 print('['+MPI.Get_processor_name()+'] more processes than the GPU numbers!')
#                 raise
#             gpu_device = pycuda.driver.Device(gpuid)
#             gpu_context = gpu_device.make_context()
#             gpu_initialized = True
#         else:
#             import pycuda.autoinit
#             gpu_initialized = True
#     except:
#         pass
    
def closeGPU():
    if gpu_context is not None:
        gpu_context.detach()
