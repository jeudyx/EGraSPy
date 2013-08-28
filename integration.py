# -*- coding: UTF-8 -*-

"""Implementation of integration method(s)

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import numpy as np
from barneshut import barnes_hut_gravitational_acceleration, build_tree
from structures import OctreeNode, Particle


def leapfrog_step(positions, velocities, masses, densities, accelerations_i, dt, tree=None, theta=0.5):
    """
    Implements Leap-Frog method, with the kick-drift-kick version.
    All arrays are assumed to be numpy arrays

    :param positions: positions in meters
    :param velocities: velocities in m/s
    :param masses: masses in kg
    :param densities: densities in kg/m³
    :param accelerations_i: aceleration from previous step (or initial estimate)
    :param dt: time step, in seconds
    :param tree:
    :param theta: for BH algorithm
    :return:

    """

    if not tree:
        tree = build_tree(positions, velocities, masses, densities)

    velocities_half = velocities + (accelerations_i * (dt / 2.))
    positions += velocities_half * dt

    particles = [Particle.from_nparray(np.array([r[0], r[1], r[2], velocities[i][0],
                                                 velocities[i][1], velocities[i][2],
                                                 masses[i], densities[i], 0.])) for i, r in enumerate(positions)]

    accelerations = np.array([barnes_hut_gravitational_acceleration(p, tree, theta) for p in particles])
    velocities += velocities_half + (accelerations * (dt / 2.))

    return accelerations
