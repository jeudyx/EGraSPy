# -*- coding: UTF-8 -*-

"""Implementation of integration method(s)

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

from physics import total_energy


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