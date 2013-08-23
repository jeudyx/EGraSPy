# coding=utf-8
"""

"""

import numpy as np
from astro_constants import SUN_MASS
from scipy.constants import parsec, astronomical_unit
import physics


class Particle(object):
    """Representation of a particle"""

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

    def __str__(self):
        return "position: %s, velocity: %s, mass: %s, density: %s" % \
               (self.position, self.velocity, self.normalized_mass, self.position)

    @property
    def normalized_mass(self):
        """Normalized mass with respect to Sun mass"""
        return self.mass / SUN_MASS

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

    def __init__(self, distance_to_center=0, center=np.array([0., 0., 0.]), parent=None):
        self.childnodes = []
        self.mass = 0.
        self.center_of_mass = np.array([0., 0., 0.])
        self.parent_node = parent
        self._level = 0 if not parent else parent._level + 1
        self._limiting_cube = Cube(distance_to_center, center)
        self.particle = None
        self.n_particles = 0

    @property
    def normalized_mass(self):
        return self.mass / SUN_MASS

    @property
    def normalized_center_of_mass_parsec(self):
        return self.center_of_mass / parsec

    @property
    def is_leaf(self):
        return len(self.childnodes) == 0

    @property
    def num_leaves(self):
        if self.is_leaf:
            return 1
        else:
            num = 0
            for child_node in self.childnodes:
                num += child_node.num_leaves
            return num

    @property
    def num_populated_leaves(self):
        if self.is_leaf:
            if self.particle:
                return 1
            else:
                return 0
        else:
            num = 0
            for child_node in self.childnodes:
                num += child_node.num_populated_leaves
            return num

    def __str__(self):
        return "Level: %s, n_particles: %s. Total mass: %s, COM: %s" % (self._level, self.n_particles,
                                                                        self.normalized_mass, self.center_of_mass)

    def contains_particle(self, particle):
        return self._limiting_cube.contains_point(particle.position)

    def create_empty_child_nodes(self):
        direction_matrix = np.array([[1., 1., 1.], [-1., 1., 1.], [1., -1., 1.], [1., 1., -1.],
                                     [-1., -1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., -1.]])

        for i in range(0, 8):
            child_center = (self._limiting_cube.center +
                            ((self._limiting_cube.distance_to_center / 2.) * direction_matrix[i]))
            self.childnodes.append(OctreeNode(self._limiting_cube.distance_to_center / 2., child_center, self))

    def insert_particle(self, particle):
        if self.mass == 0:
            if self._limiting_cube.contains_point(particle.position):
                # Node was empty, insert here
                self.mass = particle.mass
                self.center_of_mass = particle.position
                self.particle = particle
                self.n_particles = 1
            else:
                raise Exception("Particle was not contained by root empty node. Particle position: %s" %
                                particle.position)
        elif self.is_leaf:
            # External node
            # If node x is an external node, say containing a body named c,
            # then there are two bodies b and c in the same region
            # Subdivide the region further by creating 8 children.

            self.create_empty_child_nodes()

            # Then, recursively insert both b and c into the appropriate quadrant(s).
            # Since b and c may still end up in the same quadrant,
            # there may be several subdivisions during a single insertion.

            found_new = False
            found_existing = False

            for child_node in self.childnodes:
                # Try to insert new particle
                if not found_new and child_node.contains_particle(particle):
                    child_node.insert_particle(particle)
                    found_new = True

                if not found_existing and child_node.contains_particle(self.particle):
                    child_node.insert_particle(self.particle)
                    self.particle = None
                    found_existing = True

                if found_existing and found_new:
                    break

            if not found_existing and not found_new:
                raise Exception("Exception during external node insertion")

            # Finally, update the center-of-mass and total mass of x
            self.center_of_mass = physics.center_of_mass(self.mass, self.center_of_mass, particle.mass,
                                                         particle.position)
            self.mass += particle.mass
            self.n_particles += 1

        else:
            # Internal node
            # If node is an internal node, update the center-of-mass and total mass of node.
            self.center_of_mass = physics.center_of_mass(self.mass, self.center_of_mass, particle.mass,
                                                         particle.position)
            self.mass += particle.mass
            self.n_particles += 1
            # Recursively insert the body b in the appropriate quadrant.
            for child_node in self.childnodes:
                if child_node.contains_particle(particle):
                    child_node.insert_particle(particle)
                    return
            raise Exception("Particle was not contained by any childnode. Particle position: %s" % particle.position)


class Volume(object):
    """Abstract representation of any volume"""

    def __init__(self):
        # TODO
        return

    def contains_point(self, point):
        return


class Sphere(Volume):
    """Represents a sphere object
    """
    def __init__(self, radius, center):
        """

        :param radius:
        :param center: 3D vector (numpy array)
        """
        self.radius = radius
        self.center = center

    def contains_point(self, point):
        """

        :param point: point to test to see if it is contained within the sphere volume
        :return: true or false
        """
        # Should I substract the point from the sphere center?
        return self.radius >= (np.linalg.norm(point))

class Cube(Volume):
    """Representation of cube"""

    def __init__(self, distance_to_center, center):
        """

        :param distance_to_center: Distance from center to a vertex
        :param center:
        """
        self.vertices = []  # 8 3D vectors
        self.center = center  # 3D vector (numpy array) marking the center of the Cube
        self.distance_to_center = distance_to_center
        self.__create_vertices__()

    def __create_vertices__(self):
        """Populate the vertices based on the distance_to_center"""

        direction_matrix = np.array([[-1., -1., 1.], [-1., 1., 1.], [1., 1., 1.],
                                    [1., -1., 1.], [-1., -1., -1.], [-1., 1., -1.],
                                    [1., 1., -1.], [1., -1., -1.]])

        for i in range(0, 8):
            vertex = [(self.distance_to_center * direction_matrix[i]) + self.center]
            self.vertices.append(vertex)

        self.vertices = np.array(self.vertices).reshape(8, 3)

    def contains_point(self, point):
        """

        :param point: 3D coordinates of a point (numpy array)
        :return: True or False depending if the point is contained within the volumen of the cube
        """

        coord_min = np.array([min(self.vertices[:, 0]), min(self.vertices[:, 1]), min(self.vertices[:, 2])])
        coord_max = np.array([max(self.vertices[:, 0]), max(self.vertices[:, 1]), max(self.vertices[:, 2])])

        return all(point >= coord_min) and all(point <= coord_max)