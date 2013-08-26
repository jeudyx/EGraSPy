"""This module will do some basic profiling to ty to enhance performance

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import cProfile

from generate_cloud import generate_cloud
from mock import MagicMock
from structures import OctreeNode
from barneshut import barnes_hut_gravitational_acceleration
from physics import brute_force_gravitational_acceleration

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


def profile_barnes_hut(particles, theta):
    tree = OctreeNode(distance_to_center=1.)
    for i, p in enumerate(particles):
        tree.insert_particle(p)

    for p in particles:
        barnes_hut = barnes_hut_gravitational_acceleration(p, tree, theta)


def profile_all():
    particles = generate_cloud(args, write_file=False)
    profile_barnes_hut(particles, 0.1)
    print 'Done BH 0.1'
    profile_barnes_hut(particles, 0.25)
    print 'Done BH 0.25'
    profile_barnes_hut(particles, 0.5)
    print 'Done BH 0.5'
    profile_barnes_hut(particles, 0.95)
    print 'Done BH 0.95'
    profile_brute_force(particles)
    print 'Done brute force'


# profile = cProfile.Profile()
# profile.runcall(profile_all)
# profile.dump_stats('./profile_all.profile')

cProfile.run("profile_all()")