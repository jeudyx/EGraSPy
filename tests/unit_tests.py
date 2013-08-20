__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
import physics
from structures import OctreeNode, Cube, Particle, Sphere
from astro_constants import SUN_MASS
from generate_cloud import generate_mass_distribution, adjust_mass, generate_random_positions_from_a_to_b, generate_sphere_position_distribution



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

    def test_fails_total_energy_diff_len_list(self):
        with self.assertRaises(ValueError):
            physics.total_energy(np.array([0, 0]), [np.array([0, 0, 0])], [np.array([0, 0, 0])])

    def test_kin_energy_same_total_energy_same_particle(self):
        masses = np.array([10.])
        positions = [np.array([1., 1., 1.])]
        velocities = [np.array([10., 10., 10.])]
        self.assertEqual(physics.kinetic_energy(masses[0], velocities[0]),
                         physics.total_energy(masses, velocities, positions))

    def test_zero_kin_energy(self):
        self.assertEqual(physics.kinetic_energy(10, np.array([0., 0., 0.])), 0.)


class TestGeometry(unittest.TestCase):

    def setUp(self):
        RADIUS = 10.
        self.cube = Cube(RADIUS, np.array([0., 0., 0.]))
        self.point1 = np.array([0., 0., 0.])
        self.point2 = np.array([RADIUS, 0, 0])
        self.point3 = np.array([0, 0, RADIUS])
        self.point4 = np.array([RADIUS, RADIUS, RADIUS])
        self.point5 = np.array([RADIUS, RADIUS * 2, RADIUS])

    def test_point_contained(self):
        self.assertTrue(self.cube.contains_point(self.point1))
        self.assertTrue(self.cube.contains_point(self.point2))
        self.assertTrue(self.cube.contains_point(self.point3))
        self.assertTrue(self.cube.contains_point(self.point4))

    def test_point_not_contained(self):
        self.assertFalse(self.cube.contains_point(self.point5))

    def test_knownpoint_not_contained(self):
        new_cube = Cube(12.5, np.array([-37.5, 37.5, 37.5]))
        self.assertFalse(new_cube.contains_point(self.point1))

    def test_sphere_contains_point(self):
        sphere = Sphere(10., np.array([0., 0., 0.]))
        point = np.array([1., 1., 1.])
        self.assertTrue(sphere.contains_point(point))

    def test_sphere_doesnot_contains_point(self):
        sphere = Sphere(10., np.array([0., 0., 0.]))
        point = np.array([15., 5., 5.])
        self.assertFalse(sphere.contains_point(point))


class TestOctree(unittest.TestCase):

    def setUp(self):
        self.node = OctreeNode(distance_to_center=100)
        self.particle = Particle(0., 0., 0., 0., 0., 0., 0., SUN_MASS)

    def test_insert_single_particle(self):
        node = OctreeNode()
        node.insert_particle(self.particle)
        self.assertEqual(node.normalized_mass, self.particle.normalized_mass)
        self.assertEqual(node.n_particles, 1)
        self.assertEqual(node._level, 0)
        self.assertEquals(np.linalg.norm(node.normalized_center_of_mass_parsec),
                          np.linalg.norm(self.particle.normalized_position_parsecs))

    def test_fails_particle_not_contained(self):
        p = Particle(200., 200., 200., 0., 0., 0., 0., SUN_MASS)
        with self.assertRaises(Exception):
            self.node.insert_particle(p)

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

    def test_children_contained(self):
        node = OctreeNode(distance_to_center=100)
        node.create_empty_child_nodes()
        for child_node in node.childnodes:
            for vertex in child_node._limiting_cube.vertices:
                self.assertTrue(node._limiting_cube.contains_point(vertex))
                child_node.create_empty_child_nodes()
                for grand_childnode in child_node.childnodes:
                    for vertex2 in grand_childnode._limiting_cube.vertices:
                        contains = child_node._limiting_cube.contains_point(vertex2)
                        self.assertTrue(contains)

    def test_particle_contained_in_grandchildnodes(self):
        new_node = OctreeNode(distance_to_center=100)
        new_node.insert_particle(self.particle)
        new_node.create_empty_child_nodes()
        search_result_children = False
        search_result_grandchildren = False
        for child_node in new_node.childnodes:
            child_node.create_empty_child_nodes()
            if child_node._limiting_cube.contains_point(self.particle.position):
                search_result_children = True
            for grand_childnode in child_node.childnodes:
                if grand_childnode._limiting_cube.contains_point(self.particle.position):
                    search_result_grandchildren = True
                    break

        self.assertTrue(search_result_children and search_result_grandchildren)

    def test_total_mass_match(self):
        p1 = Particle(-50., 0., 0., 0., 0., 0., 0., 100.)
        p2 = Particle(0., 0., 0., 0., 0., 0., 0., 100.)
        p3 = Particle(50., 0., 0., 0., 0., 0., 0., 100.)
        new_node = OctreeNode(distance_to_center=50)
        new_node.insert_particle(p1)
        new_node.insert_particle(p2)
        new_node.insert_particle(p3)
        self.assertEqual(new_node.n_particles, 3)
        self.assertEqual(new_node.num_populated_leaves, 3)
        self.assertEqual(new_node.mass, p1.mass + p2.mass + p3.mass)
        self.assertEqual(np.linalg.norm(new_node.center_of_mass), 0.)

    def test_deep_levels(self):
        new_node = OctreeNode(distance_to_center=50)
        new_node.create_empty_child_nodes()
        new_node.childnodes[0].create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0].create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0].create_empty_child_nodes()
        self.assertEqual(new_node._level, 0)
        self.assertEqual(new_node.childnodes[0]._level, 1)
        self.assertEqual(new_node.childnodes[0].childnodes[0]._level, 2)
        self.assertEqual(new_node.childnodes[0].childnodes[0].childnodes[0]._level, 3)
        self.assertEqual(new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0]._level, 4)
        self.assertEqual(new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0]._level, 5)


class TestParticleDistribution(unittest.TestCase):

    def setUp(self):
        self.NUM_PARTICLES = 1000
        self.TOTAL_MASS = 1.
        self.VARIATION = 0.25

    def test_zero_variation(self):
        masses = generate_mass_distribution(self.TOTAL_MASS, self.NUM_PARTICLES)
        self.assertEqual(len(masses[masses == self.TOTAL_MASS/self.NUM_PARTICLES]), self.NUM_PARTICLES)

    def test_variation(self):
        masses = generate_mass_distribution(self.TOTAL_MASS, self.NUM_PARTICLES, max_variation=self.VARIATION)
        self.assertAlmostEqual(sum(masses), self.TOTAL_MASS)

    def test_total_mass(self):
        masses = generate_mass_distribution(self.TOTAL_MASS, self.NUM_PARTICLES)
        self.assertAlmostEqual(sum(masses), self.TOTAL_MASS)

    def test_adjust_mass_possitive_difference(self):
        masses_list = np.repeat((self.TOTAL_MASS / self.NUM_PARTICLES) / 2., self.NUM_PARTICLES)
        masses_list = adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_negative_difference(self):
        masses_list = np.repeat((self.TOTAL_MASS / self.NUM_PARTICLES) * 2., self.NUM_PARTICLES)
        masses_list = adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_negative_difference_variations(self):
        masses_list = [0.75, 0.5, 0.25, 0.1, 0.1, 0.1]
        masses_list = adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_positive_difference_variations(self):
        masses_list = [0.13, 0.25, 0.25, 0.01, 0.05, 0.1]
        masses_list = adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_generate_random_positions_from_a_to_b(self):
        a = -5
        b = 10
        positions = generate_random_positions_from_a_to_b(a, b, 100)
        for p in positions:
            self.assertTrue(a <= p[0] <= b)
            self.assertTrue(a <= p[0] <= b)
            self.assertTrue(a <= p[0] <= b)

    def test_generate_sphere_position_distribution(self):
        points = generate_sphere_position_distribution(10, [0., 0., 0.], 100)
        sphere = Sphere(10, [0., 0., 0.])
        self.assertEqual(len(points), 100)
        for p in points:
            self.assertTrue(sphere.contains_point(p))

    def test_generate_sphere_position_distribution_unique_points(self):
        points = generate_sphere_position_distribution(10, [0., 0., 0.], 100)
        self.assertEqual(len(points), len(set([tuple(p) for p in points])))