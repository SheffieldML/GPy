"""
The package for scikits.cuda initialization

Global variables: initSuccess
providing CUBLAS handle: cublas_handle
"""

try:
    from scikits.cuda import cublas
    import scikits.cuda.linalg as culinalg
    culinalg.init()
    cublas_handle = cublas.cublasCreate()
except:

gpu_initialized = False
gpu_device = None
gpu_context = None
    
def initGPU(gpuid=None):
    if gpu_initialized:
        return
    if gpuid==None:
        try:
            import pycuda.autoinit
            gpu_initialized = True
        except:
            pass
    else:
        try:
            import pycuda.driver
            pycuda.driver.init()
            if gpuid>=pycuda.driver.Device.count():
                return
            gpu_device = pycuda.driver.Device(gpuid)
            gpu_context = gpu_device.make_context()
            gpu_initialized = True
        except:
            pass

def closeGPU():
    if gpu_context is not None:
        gpu_context.detach()
