import numpy as np
import warnings
from .. import kern

def build_lcm(input_dim, num_outputs, CK = [], NC = [], W_columns=1,W=None,kappa=None):
    #TODO build_icm or build_lcm
    """
    Builds a kernel for a linear coregionalization model

    :input_dim: Input dimensionality
    :num_outputs: Number of outputs
    :param CK: List of coregionalized kernels (i.e., this will be multiplied by a coregionalize kernel).
    :param K: List of kernels that will be added up together with CK, but won't be multiplied by a coregionalize kernel
    :param W_columns: number tuples of the corregionalization parameters 'coregion_W'
    :type W_columns: integer
    """

    for k in CK:
        if k.input_dim <> input_dim:
            k.input_dim = input_dim
            warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    for k in NC:
        if k.input_dim <> input_dim + 1:
            k.input_dim = input_dim + 1
            warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    kernel = CK[0].prod(kern.coregionalize(num_outputs,W_columns,W,kappa),tensor=True)
    for k in CK[1:]:
        k_coreg = kern.coregionalize(num_outputs,W_columns,W,kappa)
        kernel += k.prod(k_coreg,tensor=True)
    for k in NC:
        kernel += k

    return kernel
