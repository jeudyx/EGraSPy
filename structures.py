# coding=utf-8
"""

"""

import numpy as np
from astro_constants import SUN_MASS
from scipy.constants import parsec, astronomical_unit


class Particle(object):
    """Representation of a particle"""
    def __init__(self):
        """Basic constructor for empty objects """
        self.position = np.array([0.,0.,0.])
        self.velocity = np.array([0.,0.,0.])
        self.mass = 0.
        self.density = 0.

    def __init__(self, x, y, z, vx, vy, vz, rho, m):
        """Class constructor with values
        :param x: x component of position (in m from origin)
        :param y: y component of position (in m from origin)
        :param z: z component of position (in m from origin)
        :param vx: x component of velocity (in m from origin)
        :param vy: y component of velocity (in m from origin)
        :param vz: z component of velocity (in m from origin)
        :param rho: mean density of particle (in gr/cm³)
        :param m: mass (in kg)
        """
        self.position = np.array([x, y, z])
        self.velocity = np.array([vx, vy, vz])
        self.mass = 0.
        self.density = 0.

    @property
    def normalized_mass(self):
        """Normalized mass with respect to Sun mass"""
        return self.mass/SUN_MASS

    @property
    def normalized_position_parsecs(self):
        """Normalized position in parsecs"""
        return (self.position * 1000.) / parsec

    @property
    def normalized_position_au(self):
        """Normalized position in astronomical units"""
        return (self.position * 1000.) / astronomical_unit


class OctreeNode(object):
    """Representation of an octree node"""

    def __init__(self):
        self.childnodes = []
        self._level = 0
        self.mass = 0.
        self.center_of_mass = np.array([0., 0., 0.])
        self.parent_node = None

    @property
    def normalized_mass(self):
        return self.mass / SUN_MASS

    @property
    def normalized_center_of_mass_parsec(self):
        return self.center_of_mass / parsec

    @property
    def is_leave(self):
        return len(self.childnodes) > 0

