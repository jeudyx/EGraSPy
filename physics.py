__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import scipy as sp
import scipy.constants
import numpy as np
from math import sqrt


def norm(p):
    return sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


def gravitational_acceleration(ri, rj, mj):
    """Newton equation of motion for 2 particles i and j.

    :param ri: position vector of particle i
    :param rj: position vector of particle j
    :param mj: mass (in kg) of particle j
    :return: Acceleration of particle i by a particle j
    """

    # Pending: considerations on smooth length

    diff = rj - ri
    diff_scalar = norm(diff)
    return sp.constants.G * mj * (diff / (diff_scalar * diff_scalar * diff_scalar))


def kinetic_energy(m, v):
    """ Kinetic energy of a body

    :param m: mass in kg
    :param v: velocity vector (in m/s)
    :return: kinetic energy in Joules
    """
    return 0.5 * m * (np.linalg.norm(v) ** 2)


def potential_energy(mi, mj, ri, rj):
    """Gravitational potential energy between two bodies i and j

    :param mi: mass (in kg) of body i
    :param mj: mass (in kg) of body j
    :param ri: position vector (in meters) of body i
    :param rj: position vector (in meters) of body j
    :return: gravitational potential energy in Joules
    """
    diff = rj - ri
    if np.linalg.norm(diff) == 0.:
        raise ZeroDivisionError
    return (-sp.constants.G * mi * mj) / np.linalg.norm(diff)


def total_energy(masses, velocities, positions):
    """ Calculates the total energy of a list of bodies

    :param masses: numpy array of masses
    :param velocities: list of velocity vectors (numpy arrays)
    :param positions: list of position vectors (numpy arrays)
    :return: total energy :raise: Value error if lists are of different sizes
    """
    ekin = 0
    epot = 0
    if not (len(masses) == len(velocities) == len(positions)):
        raise ValueError("Error calculating total energy: masses and velocities must numpy arrays of the same size")

    for mi, vi, ri in zip(masses, velocities, positions):
        ekin += kinetic_energy(mi, vi)
        for mj, vj, rj in zip(masses, velocities, positions):
            # Only consider if it is a different body
            if np.linalg.norm(ri - rj) != 0:
                epot += potential_energy(mi, mj, ri, rj)
     # Divide potential by 2 since every body was accounted twice
    epot /= 2

    return ekin + epot


def center_of_mass(mass1, position1, mass2, position2):
    """

    :param mass1: Mass of first body (kg)
    :param position1: Position of first body (numpy array 3D)
    :param mass2: Mass of second body (kg)
    :param position2: Position of second body (numpy array 3D)
    :return:
    """
    return ((mass1 * position1) + (mass2 * position2)) / (mass1 + mass2)


def center_of_mass_minus_particle(total_mass, center_of_mass, mass2, position2):
    """
    Substract a particle from the center of mass that was previously calculated considering it
    :param center_of_mass: current center of mass
    :param total_mass: total mass of the system
    :param mass2: mass of the particle we want to substract
    :param position2: position of the particle we want to substract
    :return: center of mass without particle
    """
    return (center_of_mass - (mass2 * position2 / total_mass)) * (total_mass / (total_mass-mass2))

def brute_force_gravitational_acceleration(p, system):
    resp = np.array([0., 0., 0.])
    for q in system:
        if p != q:
            resp += gravitational_acceleration(p.position, q.position, q.mass)

    return resp


def calculate_volume(mass, density):
    return mass / density


def calculate_radius(mass, density):
    return ((3.0 * mass) / (4.0 * np.pi * density)) ** (1.0/3.0)