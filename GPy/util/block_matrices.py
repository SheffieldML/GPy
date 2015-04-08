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

def block_dot(A, B):
    """
    Element wise dot product on block matricies

    +------+------+   +------+------+    +-------+-------+
    |      |      |   |      |      |    |A11.B11|B12.B12|
    | A11  | A12  |   | B11  | B12  |    |       |       |
    +------+------+ o +------+------| =  +-------+-------+
    |      |      |   |      |      |    |A21.B21|A22.B22|
    | A21  | A22  |   | B21  | B22  |    |       |       |
    +-------------+   +------+------+    +-------+-------+

    ..Note
        If either (A or B) of the diagonal matrices are stored as vectors then a more
        efficient dot product using numpy broadcasting will be used, i.e. A11*B11
    """
    #Must have same number of blocks and be a block matrix
    assert A.dtype is np.dtype('object'), "Must be a block matrix"
    assert B.dtype is np.dtype('object'), "Must be a block matrix"
    Ashape = A.shape
    Bshape = B.shape
    assert Ashape == Bshape
    def f(A,B):
        if Ashape[0] == Ashape[1] or Bshape[0] == Bshape[1]:
            #FIXME: Careful if one is transpose of other, would make a matrix
            return A*B
        else:
            return np.dot(A,B)
    dot = np.vectorize(f, otypes = [np.object])
    return dot(A,B)


if __name__=='__main__':
    A = np.zeros((5,5))
    B = get_blocks(A,[2,3])
    B[0,0] += 7
    print B

    assert np.all(unblock(B) == A)
