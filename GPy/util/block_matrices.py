# Copyright (c) 2014-2015, Alan Saul
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np


def get_blocks_3d(A, blocksizes, pagesizes=None):
    """
    Given a 3d matrix, make a block matrix, where the first and second dimensions are blocked according
    to blocksizes, and the pages are blocked using pagesizes
    """
    assert (A.shape[0] == A.shape[1]) and len(
        A.shape
    ) == 3, "can't blockify this non-square matrix, may need to use 2d version"
    N = np.sum(blocksizes)
    assert A.shape[0] == N, "bad blocksizes"
    num_blocks = len(blocksizes)
    if pagesizes is None:
        # Assume each page of A should be its own dimension
        pagesizes = range(A.shape[2])  # [0]*A.shape[2]
    num_pages = len(pagesizes)
    B = np.empty(shape=(num_blocks, num_blocks, num_pages), dtype=np.object)
    for Bk in pagesizes:
        count_i = 0
        for Bi, i in enumerate(blocksizes):
            count_j = 0
            for Bj, j in enumerate(blocksizes):
                # We want to have it count_k:count_k + k but its annoying as it makes a NxNx1 array is page sizes are set to 1
                B[Bi, Bj, Bk] = A[count_i : count_i + i, count_j : count_j + j, Bk]
                count_j += j
            count_i += i
    return B


def get_blocks(A, blocksizes):
    assert (A.shape[0] == A.shape[1]) and len(
        A.shape
    ) == 2, "can't blockify this non-square matrix"
    N = np.sum(blocksizes)
    assert A.shape[0] == N, "bad blocksizes"
    num_blocks = len(blocksizes)
    B = np.empty(shape=(num_blocks, num_blocks), dtype=np.object)
    count_i = 0
    for Bi, i in enumerate(blocksizes):
        count_j = 0
        for Bj, j in enumerate(blocksizes):
            B[Bi, Bj] = A[count_i : count_i + i, count_j : count_j + j]
            count_j += j
        count_i += i
    return B


def get_block_shapes_3d(B):
    assert B.dtype is np.dtype("object"), "Must be a block matrix"
    # FIXME: This isn't general AT ALL...
    return get_block_shapes(B[:, :, 0]), B.shape[2]


def get_block_shapes(B):
    assert B.dtype is np.dtype("object"), "Must be a block matrix"
    return [B[b, b].shape[0] for b in range(0, B.shape[0])]


def unblock(B):
    assert B.dtype is np.dtype("object"), "Must be a block matrix"
    block_shapes = get_block_shapes(B)
    num_elements = np.sum(block_shapes)
    A = np.empty(shape=(num_elements, num_elements))
    count_i = 0
    for Bi, i in enumerate(block_shapes):
        count_j = 0
        for Bj, j in enumerate(block_shapes):
            A[count_i : count_i + i, count_j : count_j + j] = B[Bi, Bj]
            count_j += j
        count_i += i
    return A


def block_dot(A, B, diagonal=False):
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
        If any block of either (A or B) are stored as 1d vectors then we assume
        that it denotes a diagonal matrix efficient dot product using numpy
        broadcasting will be used, i.e. A11*B11

        If either (A or B) of the diagonal matrices are stored as vectors then a more
        efficient dot product using numpy broadcasting will be used, i.e. A11*B11
    """
    # Must have same number of blocks and be a block matrix
    assert A.dtype is np.dtype("object"), "Must be a block matrix"
    assert B.dtype is np.dtype("object"), "Must be a block matrix"
    assert A.shape == B.shape

    def f(C, D):
        """
        C is an element of A, D is the associated element of B
        """
        Cshape = C.shape
        Dshape = D.shape
        if diagonal and (
            len(Cshape) == 1
            or len(Dshape) == 1
            or C.shape[0] != C.shape[1]
            or D.shape[0] != D.shape[1]
        ):
            print("Broadcasting, C: {} D:{}".format(C.shape, D.shape))
            return C * D
        else:
            print("Dotting, C: {} C:{}".format(C.shape, D.shape))
            return np.dot(C, D)

    dot = np.vectorize(f, otypes=[np.object])
    return dot(A, B)


if __name__ == "__main__":
    A = np.zeros((5, 5))
    B = get_blocks(A, [2, 3])
    B[0, 0] += 7
    print(B)

    assert np.all(unblock(B) == A)
