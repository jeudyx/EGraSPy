"""Main script from where the simulation is started

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse
import numpy as np
import json
from mpi4py import MPI
from structures import OctreeNode, Particle
from barneshut import build_tree


def start_simulation(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
                     verbose=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tree = None
    particles = []

    if rank == 0:
        raw_vals = np.loadtxt(cloud_path, delimiter=',', skiprows=1)
        positions = raw_vals[:, 0:3]
        velocities = raw_vals[:, 3:6]
        masses = raw_vals[:, 6:7]
        densities = raw_vals[:, 7:8]
        tree = build_tree(positions, velocities, masses, densities, out_particles=particles)

    tree = comm.bcast(tree, root=0)
    particles = comm.bcast(particles, root=0)

    if verbose:
        print "I am process %s. Tree -> %s" % (rank, str(tree))
        print '============'
        print "I am process %s. Particles -> %s" % (rank, particles)
        print '============'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Starts the simulation')
    parser.add_argument("-cfg", "--config", help="Path to the configuration file containing all the parameters. "
                                                 "This file must be in json format. ")
    args = parser.parse_args()
    json_data=open(args.config)
    data = json.load(json_data)

    start_simulation(data['cloud_path'], data['time_step'], data['theta'], data['save_frequency'],
                     data['start_from_iteration'], data['stop_at_iteration'], data['verbose'])

if __name__ == "__main__":
    sys.exit(main())
