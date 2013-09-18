"""Main script from where the simulation is started

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse
import numpy as np
import json
from mpi4py import MPI
from barneshut import build_tree, barnes_hut_gravitational_acceleration
from integration import leapfrog_step

def start_simulation(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
                     verbose=False):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    total_nodes = comm.Get_size()

    tree = None
    particles = []
    accelerations = []

    if rank == 0:
        raw_vals = np.loadtxt(cloud_path, delimiter=',', skiprows=1)
        positions = raw_vals[:, 0:3]
        velocities = raw_vals[:, 3:6]
        masses = raw_vals[:, 6:7]
        densities = raw_vals[:, 7:8]
        tree = build_tree(positions, velocities, masses, densities, out_particles=particles)
        accelerations = [np.zeros(3.0) for p in particles]

        if verbose:
            print '%s processors available. Particles: %s' % (total_nodes, len(particles))

    tree = comm.bcast(tree, root=0)
    particles = comm.bcast(particles, root=0)
    accelerations = comm.bcast(accelerations, root=0)

    i = start_from_iteration

    n_split = 1 + len(particles) / total_nodes

    range_start = rank * n_split
    range_end = range_start + n_split

    while i < stop_at_iteration:
        particles_chunk = particles[range_start:range_end]
        accelerations_chunk = accelerations[range_start:range_end]

        accelerations_i = leapfrog_step(time_step, accelerations_chunk, barnes_hut_gravitational_acceleration,
                                        particles=particles_chunk, tree=tree, theta=theta)


        if rank == 0:
            pass
        else:
            pass

        i += 1


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
