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