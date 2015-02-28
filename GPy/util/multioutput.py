import numpy as np
import warnings
import GPy


def get_slices(input_list):
    num_outputs = len(input_list)
    _s = [0] + [ _x.shape[0] for _x in input_list ]
    _s = np.cumsum(_s)
    slices = [slice(a,b) for a,b in zip(_s[:-1],_s[1:])]
    return slices

def build_XY(input_list,output_list=None,index=None):
    num_outputs = len(input_list)
    if output_list is not None:
        assert num_outputs == len(output_list)
        Y = np.vstack(output_list)
    else:
        Y = None

    if index is not None:
        assert len(index) == num_outputs
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,index)] )
    else:
        I = np.hstack( [np.repeat(j,_x.shape[0]) for _x,j in zip(input_list,range(num_outputs))] )

    X = np.vstack(input_list)
    X = np.hstack([X,I[:,None]])

    return X,Y,I[:,None]#slices

def build_likelihood(Y_list,noise_index,likelihoods_list=None):
    Ny = len(Y_list)
    if likelihoods_list is None:
       likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j) for y,j in zip(Y_list,range(Ny))]
    else:
        assert len(likelihoods_list) == Ny
    #likelihood = GPy.likelihoods.mixed_noise.MixedNoise(likelihoods_list=likelihoods_list, noise_index=noise_index)
    likelihood = GPy.likelihoods.mixed_noise.MixedNoise(likelihoods_list=likelihoods_list)
    return likelihood


def ICM(input_dim, num_outputs, kernel, W_rank=1,W=None,kappa=None,name='ICM'):
    """
    Builds a kernel for an Intrinsic Coregionalization Model

    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    if kernel.input_dim != input_dim:
        kernel.input_dim = input_dim
        warnings.warn("kernel's input dimension overwritten to fit input_dim parameter.")

    K = kernel.prod(GPy.kern.Coregionalize(1, num_outputs, active_dims=[input_dim], rank=W_rank,W=W,kappa=kappa,name='B'),name=name)
    return K


def LCM(input_dim, num_outputs, kernels_list, W_rank=1,name='ICM'):
    """
    Builds a kernel for an Linear Coregionalization Model

    :input_dim: Input dimensionality (does not include dimension of indices)
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    Nk = len(kernels_list)
    K = ICM(input_dim,num_outputs,kernels_list[0],W_rank,name='%s%s' %(name,0))
    j = 1
    for kernel in kernels_list[1:]:
        K += ICM(input_dim,num_outputs,kernel,W_rank,name='%s%s' %(name,j))
        j += 1
    return K


def Private(input_dim, num_outputs, kernel, output, kappa=None,name='X'):
    """
    Builds a kernel for an Intrinsic Coregionalization Model

    :input_dim: Input dimensionality
    :num_outputs: Number of outputs
    :param kernel: kernel that will be multiplied by the coregionalize kernel (matrix B).
    :type kernel: a GPy kernel
    :param W_rank: number tuples of the corregionalization parameters 'W'
    :type W_rank: integer
    """
    K = ICM(input_dim,num_outputs,kernel,W_rank=1,kappa=kappa,name=name)
    K.B.W.fix(0)
    _range = range(num_outputs)
    _range.pop(output)
    for j in _range:
        K.B.kappa[j] = 0
        K.B.kappa[j].fix()
    return K
