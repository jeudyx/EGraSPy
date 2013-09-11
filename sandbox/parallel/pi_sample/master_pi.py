from mpi4py import MPI

import numpy
import sys

print "Spawning MPI processes"
comm = MPI.COMM_SELF.Spawn(sys.executable, args=['cpi.py'], maxprocs=8)
N = numpy.array(100, 'i')
comm.Bcast([N, MPI.INT], root=MPI.ROOT)
PI = numpy.array(0.0, 'd')
comm.Reduce(None, [PI, MPI.DOUBLE],
            op=MPI.SUM, root=MPI.ROOT)

print "Calculated value of PI is: %f16" %PI
