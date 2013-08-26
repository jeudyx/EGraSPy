"""This module will do some basic profiling to ty to enhance performance

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import cProfile

from generate_cloud import generate_cloud
from mock import MagicMock
from structures import OctreeNode
from barneshut import barnes_hut_gravitational_acceleration
from physics import brute_force_gravitational_acceleration
import time
import timeit

args = MagicMock()
args.mass = 1.
args.nparticles = 500
args.rho = 1E20
args.temperature = 10.
args.path = ''
args.rotation = 0.
args.variation = 0.75
args.config = None


def profile_brute_force(particles):
    for p in particles:
        brute = brute_force_gravitational_acceleration(p, particles)


def profile_barnes_hut(particles, tree, theta):
    for p in particles:
        barnes_hut = barnes_hut_gravitational_acceleration(p, tree, theta)


def profile_all():
    start = time.time()
    particles = generate_cloud(args, write_file=False)
    tree = OctreeNode(distance_to_center=1.)
    for i, p in enumerate(particles):
        tree.insert_particle(p)
    end = time.time()
    print 'Creating tree %.5fs' % (end - start)
    start = time.time()
    profile_barnes_hut(particles, tree, 0.1)
    end = time.time()
    print 'Done BH 0.1 %.5fs' % (end - start)
    start = time.time()
    profile_barnes_hut(particles, tree, 0.25)
    end = time.time()
    print 'Done BH 0.25 %.5fs' % (end - start)
    start = time.time()
    profile_barnes_hut(particles, tree, 0.5)
    end = time.time()
    print 'Done BH 0.5 %.5fs' % (end - start)
    start = time.time()
    profile_barnes_hut(particles, tree, 0.95)
    end = time.time()
    print 'Done BH 0.95 %.5fs' % (end - start)
    start = time.time()
    profile_brute_force(particles)
    end = time.time()
    print 'Done brute force %.5fs' % (end - start)


# profile = cProfile.Profile()
# profile.runcall(profile_all)
# profile.dump_stats('./profile_all.profile')

cProfile.run("profile_all()")