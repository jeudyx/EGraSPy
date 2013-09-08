# -*- coding: UTF-8 -*-

"""Implementation of integration method(s)

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import numpy as np
from barneshut import barnes_hut_gravitational_acceleration, build_tree
from structures import OctreeNode, Particle
from physics import total_energy, brute_force_gravitational_acceleration

def leapfrog_step_vectorized(positions, velocities, masses, densities, accelerations_i, dt, tree=None, theta=0.5):
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
    velocities = velocities_half + (accelerations * (dt / 2.))

    return accelerations


def leapfrog_step(dt, accelerations_i, gravity_function, **kwargs):
    """
    Implements Leap-Frog method, with the kick-drift-kick version.
    All arrays are assumed to be numpy arrays of the same size
    Particle velocity and position is modified

    :param particles:
    :param dt: time step, in seconds
    :param tree:
    :param theta: for BH algorithm
    :return:

    """

    accelerations = []

    particles = kwargs['particles']

    if len(particles) != len(accelerations_i):
        raise ValueError("Particles and accelerations must be the same.")

    for i, p in enumerate(particles):
        velocity_half = p.velocity + (accelerations_i[i] * (dt / 2.))
        p.position += (velocity_half * dt)
        acceleration = gravity_function(p, **kwargs)
        accelerations.append(acceleration)
        p.velocity = velocity_half + (acceleration * (dt / 2.))

    return accelerations


def get_system_total_energy(particles):
    masses = [p.mass for p in particles]
    positions = [p.position for p in particles]
    velocities = [p.velocity for p in particles]
    return total_energy(masses, velocities, positions)