"""
The module of tools for parallelization (MPI)
"""

try:
    from mpi4py import MPI
    def get_id_within_node(comm=MPI.COMM_WORLD):
        rank = comm.rank
        nodename =  MPI.Get_processor_name()
        nodelist = comm.allgather(nodename)
        return len([i for i in nodelist[:rank] if i==nodename])
except:
    pass
