import numpy as np
import warnings
from .. import kern

def build_cor_kernel(input_dim, Nout, CK = [], NC = [], W=1):
    """
    Builds an appropiate coregionalized kernel

    :input_dim: Input dimensionality
    :Nout: Number of outputs
    :param CK: List of coregionalized kernels (i.e., this will be multiplied by a coregionalise kernel).
    :param K: List of kernels that won't be multiplied by a coregionalise kernel
    :W:
    """

    for k in CK:
        if k.input_dim <> input_dim:
            k.input_dim = input_dim
            #raise Warning("kernel's input dimension overwritten to fit input_dim parameter.")
            warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    for k in NC:
        if k.input_dim <> input_dim + 1:
            k.input_dim = input_dim + 1
            #raise Warning("kernel's input dimension overwritten to fit input_dim parameter.")
            warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    kernel = CK[0].prod(kern.coregionalise(Nout,W),tensor=True)
    for k in CK[1:]:
        kernel += k.prod(kern.coregionalise(Nout,W),tensor=True)

    for k in NC:
        kernel += k

    return kernel
