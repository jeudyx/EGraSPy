"""Main script from where the simulation is started

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse
import numpy as np
import json
from mpi4py import MPI
from barneshut import build_tree, barnes_hut_gravitational_acceleration
from structures import OctreeNode
from integration import leapfrog_step
from physics import norm, brute_force_gravitational_acceleration
from generate_cloud import write_values

def start_simulation(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
                     verbose=False):
    _start_simulation_parallel(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
                               verbose)


def _start_simulation_serial(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
                               verbose=False):


    tree = None
    particles = []
    accelerations = []

    raw_vals = np.loadtxt(cloud_path, delimiter=',', skiprows=1)
    positions = raw_vals[:, 0:3]
    velocities = raw_vals[:, 3:6]
    masses = raw_vals[:, 6:7]
    densities = raw_vals[:, 7:8]
    tree = build_tree(positions, velocities, masses, densities, out_particles=particles)
    accelerations = [np.zeros(3.0) for p in particles]

    i = start_from_iteration

    while i < stop_at_iteration:

        accelerations = leapfrog_step(time_step, accelerations, barnes_hut_gravitational_acceleration,
                                      particles=particles, tree=tree, theta=theta)

        tree = OctreeNode(distance_to_center=max([norm(p.position) for p in particles]))
        for p in particles:
            tree.insert_particle(p)

        print '%s -- %s - %s - %s' % (i, accelerations[1], accelerations[5], accelerations[9])
        print '%s -- %s - %s - %s' % (i, particles[1].position, particles[5].position, particles[9].position)
        print '%s -- %s - %s - %s' % (i, particles[1].velocity, particles[5].velocity, particles[9].velocity)
        print '-----------------------\n'

        if i % save_frequency == 0:
            if verbose and False:
                print 'Iteration %s - Particles: %s \n Accelerations: %s\n\n' % (i, [str(p) for p in particles],
                                                                                 [str(a) for a in accelerations])

        i += 1


def _start_simulation_parallel(cloud_path, time_step, theta, save_frequency, start_from_iteration, stop_at_iteration,
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

        for node_i in range(1, total_nodes):
            comm.send((tree, particles, accelerations), dest=node_i, tag=11)
    else:
        tree, particles, accelerations = comm.recv(source=0, tag=11)

    i = start_from_iteration

    n_split = 1 + len(particles) / total_nodes

    range_start = rank * n_split
    range_end = range_start + n_split

    while i < stop_at_iteration:
        particles_chunk = particles[range_start:range_end]
        accelerations_chunk = accelerations[range_start:range_end]

        #print '%s - %s - %s -- %s -- %s' % (rank, i, accelerations_chunk[0], str(particles_chunk[0]), theta)

        accelerations_i = leapfrog_step(time_step, accelerations_chunk, barnes_hut_gravitational_acceleration,
                                        particles=particles_chunk, tree=tree, theta=theta)

        if rank == 0:
            # Need to receive the results from the child nodes
            particles = []
            accelerations = []

            particles.extend(particles_chunk)
            accelerations.extend(accelerations_i)

            for node_i in range(1, total_nodes):

                particles_node, accelerations_node = comm.recv(source=node_i, tag=11)

                particles.extend(particles_node)
                accelerations.extend(accelerations_node)

        else:
            comm.send((particles_chunk, accelerations_i), dest=0, tag=11)

        if rank == 0:
            tree = OctreeNode(distance_to_center=max([norm(p.position) for p in particles]))
            for p in particles:
                tree.insert_particle(p)

            for node_i in range(1, total_nodes):
                comm.send((tree, particles, accelerations), dest=node_i, tag=11)
        else:
            tree, particles, accelerations = comm.recv(source=0, tag=11)

        if rank == 0:
            if i % save_frequency == 0:
                if verbose:
                    print '%s - Max velocity: %s' % (i, max([np.linalg.norm(p.velocity) for p in particles]))
                    #print 'Iteration %s - Particles: %s \n Accelerations: %s\n\n' % (i, [str(p) for p in particles],
                                                                                     #[str(a) for a in accelerations])
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
