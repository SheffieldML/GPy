"""
The package for scikits.cuda initialization

Global variables: initSuccess
providing CUBLAS handle: cublas_handle
"""

try:
    import pycuda.autoinit
    from scikits.cuda import cublas
    import scikits.cuda.linalg as culinalg
    culinalg.init()
    cublas_handle = cublas.cublasCreate()
    initSuccess = True
except:
    initSuccess = False
    
def initGPU(gpuid=None):
    if gpuid==None:
        return pycuda.tools.make_default_context()
    else:
        return pycuda.driver.Device(gpuid).make_context()