__author__ = 'jeudy'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
import physics
from structures import OctreeNode, Cube, Particle
from astro_constants import SUN_MASS

class TestPhysics(unittest.TestCase):

    def setUp(self):
        self.ri = np.random.random(3)
        self.rj = np.random.random(3)

    def test_zero_mass_acceleration(self):
        self.assertEqual(np.linalg.norm(physics.gravitational_acceleration(self.ri, self.rj, 0)), 0.)

    def test_zero_center_of_mass(self):
        self.assertEqual(np.linalg.norm(physics.center_of_mass(10., -5, 10, 5)), 0)

    def test_known_center_of_mass(self):
        self.assertAlmostEqual(np.linalg.norm(physics.center_of_mass(1000., -5, 100, 5)), 4.09090909)

class TestGeometry(unittest.TestCase):

    def setUp(self):
        RADIUS = 10.
        self.cube = Cube(RADIUS, np.array([0.,0.,0.]))
        self.point1 = np.array([0.,0.,0.])
        self.point2 = np.array([RADIUS, 0, 0])
        self.point3 = np.array([0, 0, RADIUS])
        self.point4 = np.array([RADIUS, RADIUS, RADIUS])
        self.point5 = np.array([RADIUS, RADIUS*2, RADIUS])

    def test_point_contained(self):
        self.assertTrue(self.cube.contains_point(self.point1))
        self.assertTrue(self.cube.contains_point(self.point2))
        self.assertTrue(self.cube.contains_point(self.point3))
        self.assertTrue(self.cube.contains_point(self.point4))

    def test_point_not_contained(self):
        self.assertFalse(self.cube.contains_point(self.point5))

class TestGeometry(unittest.TestCase):

    def setUp(self):
        self.particle = Particle(0., 0., 0., 0., 0., 0., 0., SUN_MASS)

    def test_insert_single_particle(self):
        node = OctreeNode()
        node.insert_particle(self.particle)
        self.assertEqual(node.normalized_mass, self.particle.normalized_mass)
        self.assertEquals(np.linalg.norm(node.normalized_center_of_mass_parsec), np.linalg.norm(self.particle.normalized_position_parsecs))

class TestOctree(unittest.TestCase):

    def setUp(self):
        self.node = OctreeNode(distance_to_center=100)

    def test_is_leaf(self):
        node = OctreeNode()
        self.assertTrue(node.is_leaf)

    def test_create_empty_child_nodes(self):
        self.node.create_empty_child_nodes()
        self.assertFalse(self.node.is_leaf)
        self.assertTrue(len(self.node.childnodes) == 8)
        # Test that the center of all child nodes are contained in main node
        for child_node in self.node.childnodes:
            self.assertTrue(self.node._limiting_cube.contains_point(child_node._limiting_cube.center))

    def test_total_mass_match(self):
        p1 = Particle(-50., 0., 0., 0., 0., 0., 0., 100.)
        p2 = Particle(0., 0., 0., 0., 0., 0., 0., 100.)
        p3 = Particle(50., 0., 0., 0., 0., 0., 0., 100.)
        new_node = OctreeNode(distance_to_center=50)
        new_node.insert_particle(p1)
        new_node.insert_particle(p2)
        new_node.insert_particle(p3)
        self.assertEqual(new_node.mass, p1.mass + p2.mass + p3.mass)