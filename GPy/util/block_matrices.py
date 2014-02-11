# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np

def get_blocks(A, blocksizes):
    assert (A.shape[0]==A.shape[1]) and len(A.shape)==2, "can;t blockify this non-square matrix"
    N = np.sum(blocksizes)
    assert A.shape[0] == N, "bad blocksizes"
    num_blocks = len(blocksizes)
    B = np.empty(shape=(num_blocks, num_blocks), dtype=np.object)
    count_i = 0
    for Bi, i in enumerate(blocksizes):
        count_j = 0
        for Bj, j in enumerate(blocksizes):
            B[Bi, Bj] = A[count_i:count_i + i, count_j : count_j + j]
            count_j += j
        count_i += i
    return B

def get_block_shapes(B):
    assert B.dtype is np.dtype('object'), "Must be a block matrix"
    return [B[b,b].shape[0] for b in range(0, B.shape[0])]

def unblock(B):
    assert B.dtype is np.dtype('object'), "Must be a block matrix"
    block_shapes = get_block_shapes(B)
    num_elements = np.sum(block_shapes)
    A = np.empty(shape=(num_elements, num_elements))
    count_i = 0
    for Bi, i in enumerate(block_shapes):
        count_j = 0
        for Bj, j in enumerate(block_shapes):
            A[count_i:count_i + i, count_j:count_j + j] = B[Bi, Bj]
            count_j += j
        count_i += i
    return A


if __name__=='__main__':
    A = np.zeros((5,5))
    B = get_blocks(A,[2,3])
    B[0,0] += 7
    print B

    assert np.all(unblock(B) == A)

    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

