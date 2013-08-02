# coding=utf-8
"""

"""

import numpy as np
from astro_constants import SUN_MASS
from scipy.constants import parsec, astronomical_unit
from physics import center_of_mass
import physics


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
        self.mass = m
        self.density = rho

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

    def __init__(self, distance_to_center=0, center=np.array([0., 0., 0.])):
        self.childnodes = []
        self.mass = 0.
        self.center_of_mass = np.array([0., 0., 0.])
        self.parent_node = None
        self._level = 0
        self._limiting_cube = Cube(distance_to_center, center)

    @property
    def normalized_mass(self):
        return self.mass / SUN_MASS

    @property
    def normalized_center_of_mass_parsec(self):
        return self.center_of_mass / parsec

    @property
    def is_leaf(self):
        return len(self.childnodes) == 0 # and self.particle is not None

    def contains_particle(self, particle):
        return self._limiting_cube.contains_point(particle.position)

    def insert_particle(self, particle):
        if self.mass == 0:
            # Node was empty, insert here
            self.mass = particle.mass
            self.center_of_mass = particle.position
        elif self.is_leaf:
            # External node
            # If node x is an external node, say containing a body named c,
            # then there are two bodies b and c in the same region
            # Subdivide the region further by creating 8 children.
            for i in range(0, 8):
                self.childnodes.append(OctreeNode())
        else:
            # Internal node
            # If node is an internal node, update the center-of-mass and total mass of node.
            self.center_of_mass = physics.center_of_mass(self.mass, self.center_of_mass, particle.mass, particle.position)
            self.mass += particle.mass
            # Recursively insert the body b in the appropriate quadrant.
            for child_node in self.childnodes:
                if child_node.contains_particle(particle):
                    child_node.insert_particle(particle)
                    return
            raise Exception("Particle was not contained by any childnode. Particle position: %s" % particle.position)



class Cube(object):
    """Representation of cube"""

    def __init__(self, distance_to_center, center):
        """

        :param distance_to_center: Distance from center to a vertex
        :param center:
        """
        self.vertices = []  # 8 3D vectors
        self.center = center # 3D vector (numpy array) marking the center of the Cube
        self.distance_to_center = distance_to_center
        self.__create_vertices__()

    def __create_vertices__(self):
        """Populate the vertices based on the distance_to_center"""

        direction_matrix = np.array([-1., -1., 1., -1., 1., 1., 1., 1., 1.,
                                    1., -1., 1., -1., -1., -1., -1., 1., -1.,
                                    1., 1., -1., 1., -1., -1.])

        for i in range(0, 8):
            self.vertices.append([self.distance_to_center + self.center])

        self.vertices = (np.array(self.vertices).reshape(24) * direction_matrix).reshape(8,3)

    def contains_point(self, point):
        """

        :param point: 3D coordinates of a point (numpy array)
        :return: True or False depending if the point is contained within the volumen of the cube
        """

        coord_min = self.vertices.min()
        coord_max = self.vertices.max()

        return all(point >= coord_min) and all(point <= coord_max)