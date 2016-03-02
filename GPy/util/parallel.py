"""
The module of tools for parallelization (MPI)
"""
import numpy as np

def get_id_within_node(comm=None):
    from mpi4py import MPI
    if comm is None: comm = MPI.COMM_WORLD
    rank = comm.rank
    nodename =  MPI.Get_processor_name()
    nodelist = comm.allgather(nodename)
    return len([i for i in nodelist[:rank] if i==nodename])

def divide_data(datanum, rank, size):
    assert rank<size and datanum>0
    
    residue = (datanum)%size
    datanum_list = np.empty((size),dtype=np.int32)
    for i in range(size):
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

def optimize_parallel(model, optimizer=None, messages=True, max_iters=1000, outpath='.', interval=100, name=None, **kwargs):
    from math import ceil
    from datetime import datetime
    import os
    if name is None: name = model.name
    stop = 0
    for iter in range(int(ceil(float(max_iters)/interval))):
        model.optimize(optimizer=optimizer, messages= True if messages and model.mpi_comm.rank==model.mpi_root else False, max_iters=interval, **kwargs)
        if model.mpi_comm.rank==model.mpi_root:
            timenow = datetime.now()
            timestr = timenow.strftime('%Y:%m:%d_%H:%M:%S')
            model.save(os.path.join(outpath, name+'_'+timestr+'.h5'))
            opt = model.optimization_runs[-1]
            if opt.funct_eval<opt.max_f_eval:
                stop = 1
        stop = model.mpi_comm.bcast(stop, root=model.mpi_root)
        if stop:
            break
