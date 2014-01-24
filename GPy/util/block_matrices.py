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



if __name__=='__main__':
    A = np.zeros((5,5))
    B = get_blocks(A,[2,3])
    B[0,0] += 7
    print B
