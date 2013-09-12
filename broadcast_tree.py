import numpy as np
from mpi4py import MPI
from structures import OctreeNode, Particle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

tree = None
particles = None
numbers = None

if rank == 0:
    numbers = np.arange(21)
    star1 = Particle(-10.9, 0., 0., 0., 2.1E3, 0., 0., 1.09)
    star2 = Particle(12.8, 0., 0., 0., -2.1E3, 0., 0., 0.9)
    planet = Particle(-9.9, 0., 0., 0., -3.1E4, 0., 0., 1.09/1000.)
    particles = [star1, star2, planet]
    tree = OctreeNode(distance_to_center=(10.9+12.8))
    tree.insert_particle(star1)
    tree.insert_particle(star2)
    tree.insert_particle(planet)

tree = comm.bcast(tree, root=0)
particles = comm.bcast(particles, root=0)
#numbers = comm.scatter(numbers, root=0)

print "I am process %s. Tree -> %s" % (rank, str(tree))
print '============'
print "I am process %s. Particles -> %s" % (rank, particles)
print '============'
print "I am process %s. Numbers -> %s" % (rank, numbers)