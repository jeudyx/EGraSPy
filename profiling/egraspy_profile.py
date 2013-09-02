"""This module will do some basic profiling to ty to enhance performance

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import cProfile

from generate_cloud import generate_cloud, get_max_distance
from mock import MagicMock
from structures import OctreeNode, Particle
from barneshut import barnes_hut_gravitational_acceleration, build_tree, adjust_tree
from physics import brute_force_gravitational_acceleration
import time
import numpy as np
import timeit

args = MagicMock()
args.mass = 1.
args.nparticles = 10000
args.rho = 1E20
args.temperature = 10.
args.path = ''
args.rotation = 0.
args.variation = 0.75
args.config = None


def profile_brute_force(particles):
    resp = np.array([0., 0., 0.])
    for p in particles:
        resp += brute_force_gravitational_acceleration(p, particles)

    return np.linalg.norm(resp)


def profile_barnes_hut(particles, tree, theta):
    resp = np.array([0., 0., 0.])
    for p in particles:
        resp += barnes_hut_gravitational_acceleration(p, tree, theta=theta)
    return np.linalg.norm(resp)


def profile_all():
    particles = generate_cloud(args, write_file=False)
    start = time.time()
    max_distance = get_max_distance(particles)
    tree = OctreeNode(distance_to_center=max_distance)
    for i, p in enumerate(particles):
        tree.insert_particle(p)
    end = time.time()
    print 'Creating tree %.5fs' % (end - start)
    start = time.time()
    res = profile_barnes_hut(particles, tree, 0.1)
    end = time.time()
    print 'Done BH 0.1 %.5fs -- %s' % (end - start, str(res))
    start = time.time()
    res = profile_barnes_hut(particles, tree, 0.25)
    end = time.time()
    print 'Done BH 0.25 %.5fs -- %s' % (end - start, str(res))
    start = time.time()
    res = profile_barnes_hut(particles, tree, 0.5)
    end = time.time()
    print 'Done BH 0.5 %.5fs -- %s' % (end - start, str(res))
    start = time.time()
    res = profile_barnes_hut(particles, tree, 0.75)
    end = time.time()
    print 'Done BH 0.75 %.5fs -- %s' % (end - start, str(res))
    start = time.time()
    res = profile_brute_force(particles)
    end = time.time()
    print 'Done brute force %.5fs -- %s' % (end - start, str(res))


def compare_build_adjust():
    positions = np.array([[n, n, n] for n in np.arange(15000.0)])
    velocities = np.array([[n, n, n] for n in np.zeros(15000.0)])
    masses = np.ones(15000)
    densities = np.zeros(15000)
    particles = []
    start = time.time()
    tree = build_tree(positions, velocities, masses, densities, out_particles=particles)
    end = time.time()
    print 'Creating tree %.5fs' % (end - start)
    #print str(tree)
    for i, p in enumerate(particles):
        if np.random.random() >= 0.1 and i < len(particles) - 10:
            particles[i].position[0] += 5. * np.random.random()
            particles[i].position[1] += 5. * np.random.random()
            particles[i].position[2] += 5. * np.random.random()

    start = time.time()
    adjust_tree(tree, tree)
    end = time.time()
    print '========================================================='
    print 'Adjusting tree %.5fs' % (end - start)
    #print str(tree)

compare_build_adjust()
exit()

args.nparticles = 10000

particles = generate_cloud(args, write_file=False)
max_distance = get_max_distance(particles)
tree = OctreeNode(distance_to_center=max_distance)
for i, p in enumerate(particles):
    tree.insert_particle(p)


def profile_create_tree():
    tree = OctreeNode(distance_to_center=max_distance)
    for i, p in enumerate(particles):
        tree.insert_particle(p)


def profile_one():
    [barnes_hut_gravitational_acceleration(p, tree, 0.7) for o in particles]

# profile = cProfile.Profile()
# profile.runcall(profile_one)
# profile.dump_stats('./profile_all.profile')

cProfile.run("profile_one()")
#cProfile.run("profile_create_tree()")