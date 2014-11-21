"""
The module of tools for parallelization (MPI)
"""
import numpy as np
try:
    from mpi4py import MPI
    def get_id_within_node(comm=MPI.COMM_WORLD):
        rank = comm.rank
        nodename =  MPI.Get_processor_name()
        nodelist = comm.allgather(nodename)
        return len([i for i in nodelist[:rank] if i==nodename])

    numpy_to_MPI_typemap = {
        np.dtype(np.float64) : MPI.DOUBLE,
        np.dtype(np.float32) : MPI.FLOAT,
        np.dtype(np.int)     : MPI.INT,
        np.dtype(np.int8)    : MPI.CHAR,
        np.dtype(np.uint8)   : MPI.UNSIGNED_CHAR,
        np.dtype(np.int32)   : MPI.INT,
        np.dtype(np.uint32)  : MPI.UNSIGNED_INT,
    }
except:
    pass

def divide_data(datanum, rank, size):
    assert rank<size and datanum>0
    
    residue = (datanum)%size
    datanum_list = np.empty((size),dtype=np.int32)
    for i in xrange(size):
        if i<residue:
            datanum_list[i] = int(datanum/size)+1
        else:
            datanum_list[i] = int(datanum/size)
    if rank<residue:
        size = datanum/size+1
        offset = size*rank
    else:
        size = datanum/size
        offset = size*rank+residue
    return offset, offset+size, datanum_list