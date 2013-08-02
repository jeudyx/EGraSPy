# -*- coding: UTF-8 -*-

__author__ = 'jeudy'

import scipy as sp
import scipy.constants
import numpy as np

def gravitational_acceleration(ri, rj, mj):
    """Newton equation of motion for 2 particles i and j.

    :param ri: position vector of particle i
    :param rj: position vector of particle j
    :param mj: mass (in kg) of particle j
    :return: Acceleration of particle i by a particle j
    """

    # Pending: considerations on smooth length

    diff = rj - ri
    return sp.constants.G * mj * (diff / (np.linalg.norm(diff) ** 3))


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
    return (-sp.constants.G * mi * mj) / np.linalg.norm(diff)

# needs rectoring

# def total_energy(cuerpos):
#     ekin = 0
#     epot = 0
#     for cuerpo in cuerpos:
#         ekin += energia_cinetica(cuerpo.masa, cuerpo.velocidad)
#         for c in cuerpos:
#             if cuerpo != c:
#                 epot += energia_potencial(cuerpo.masa, c.masa, cuerpo.posicion, c.posicion)
#
#     #Se divide entre 2 porque el aporte de cada particula se calculó 2 veces
#     epot /= 2
#
#     return ekin + epot

def center_of_mass(mass1, position1, mass2, position2):
    """

    :param mass1: Mass of first body (kg)
    :param position1: Position of first body (numpy array 3D)
    :param mass2: Mass of second body (kg)
    :param position2: Position of second body (numpy array 3D)
    :return:
    """
    return ((mass1 * position1) + (mass2 * position2)) / (mass1 + mass2)