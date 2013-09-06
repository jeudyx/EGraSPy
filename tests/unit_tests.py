__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
import physics
from mock import patch, MagicMock
from structures import OctreeNode, Cube, Particle, Sphere
from astro_constants import SUN_MASS
from generate_cloud import generate_mass_distribution, _adjust_mass, _generate_random_positions_from_a_to_b, \
    _generate_sphere_position_distribution, ParameterReader, generate_cloud
from barneshut import barnes_hut_gravitational_acceleration
from integration import leapfrog_step

class TestBarnesHut(unittest.TestCase):

    def test_single_particle_zero_grativity(self):
        tree = OctreeNode(distance_to_center=100)
        p = Particle(10., 10., 10., 0., 0., 0., 0., SUN_MASS)
        tree.insert_particle(p)
        resp = np.zeros(3)
        self.assertEqual(np.linalg.norm(barnes_hut_gravitational_acceleration(p, tree, resp)), 0.0)

    def test_two_equal_particles_grativity(self):
        tree = OctreeNode(distance_to_center=100)
        p1 = Particle(10., 10., 10., 0., 0., 0., 0., SUN_MASS)
        p2 = Particle(20., 20., 20., 0., 0., 0., 0., SUN_MASS)
        tree.insert_particle(p1)
        tree.insert_particle(p2)
        self.assertEqual(np.linalg.norm(barnes_hut_gravitational_acceleration(p1, tree)),
                         np.linalg.norm(barnes_hut_gravitational_acceleration(p2, tree)))

    def test_compare_brute_force(self):
        tree = OctreeNode(distance_to_center=100)
        p1 = Particle(10., 10., 10., 0., 0., 0., 0., SUN_MASS/3.)
        p2 = Particle(20., 20., 20., 0., 0., 0., 0., SUN_MASS/2.)
        tree.insert_particle(p1)
        tree.insert_particle(p2)
        brute1 = physics.gravitational_acceleration(p1.position, p2.position, p2.mass)
        barnes_1 = barnes_hut_gravitational_acceleration(p1, tree)
        brute2 = physics.gravitational_acceleration(p2.position, p1.position, p1.mass)
        barnes_2 = barnes_hut_gravitational_acceleration(p2, tree)
        self.assertTrue(all(barnes_1 == brute1))
        self.assertTrue(all(barnes_2 == brute2))

    def test_compare_brute_force_system(self):
        tree = OctreeNode(distance_to_center=100)
        p1 = Particle(10., 10., 10., 0., 0., 0., 0., SUN_MASS/4.)
        p2 = Particle(10., 11., 10., 0., 0., 0., 0., SUN_MASS/3.)
        p3 = Particle(10., 10., 11., 0., 0., 0., 0., SUN_MASS/2.)
        p4 = Particle(40., 40., 40., 0., 0., 0., 0., SUN_MASS)
        particles = [p1, p2, p3, p4]
        tree.insert_particle(p1)
        tree.insert_particle(p2)
        tree.insert_particle(p3)
        tree.insert_particle(p4)
        for p in particles:
            brute = np.array([0., 0., 0.])
            for q in particles:
                if p != q:
                    brute += physics.gravitational_acceleration(p.position, q.position, q.mass)
            # theta = 0 is equivalent to brute force
            barnes_hut_zero_theta = barnes_hut_gravitational_acceleration(p, tree, theta=0.0)
            barnes_hut_0_5_theta = barnes_hut_gravitational_acceleration(p, tree, theta=0.6)

            self.assertAlmostEqual(np.log(np.linalg.norm(barnes_hut_zero_theta)),
                                   np.log(np.linalg.norm(brute)), places=3)

            self.assertAlmostEqual(np.log(np.linalg.norm(barnes_hut_0_5_theta)),
                                   np.log(np.linalg.norm(brute)), places=3)


class TestParticles(unittest.TestCase):

    def test_particle_same(self):
        p1 = Particle(0., 0., 0., 0., 0., 0., 0., SUN_MASS)
        p2 = Particle(0., 0., 0., 0., 0., 0., 0., SUN_MASS)
        self.assertTrue(p1 == p2)

    def test_particle_different(self):
        p1 = Particle(0.1, 0., 0., 0., 0., 0., 0., SUN_MASS)
        p2 = Particle(0., 0., 0., 0., 0., 0., 0., SUN_MASS)
        self.assertTrue(p1 != p2)


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

    # This test may look complex. Try to deconstruct the center of mass, particle by particle
    def test_center_of_mass_minus_particle(self):
        # 9 random position vectors
        positions = np.array([np.random.rand() for i in range(0, 27)]).reshape(9, 3)
        masses = np.array([100. * np.random.rand() for i in range(0, 9)])
        com = np.zeros(3)
        com_list = []
        total_mass = 0.
        for i, m in enumerate(masses):
            com = physics.center_of_mass(total_mass, com, m, positions[i])
            total_mass += m
            com_list.append(com)

        i = len(com_list) - 1

        while i > 0:
            prev_com = physics.center_of_mass_minus_particle(total_mass, com, masses[i], positions[i])
            total_mass -= masses[i]
            com = prev_com
            self.assertAlmostEquals(np.linalg.norm(prev_com - com_list[i - 1]), 0.)
            i -= 1

    def test_fails_total_energy_diff_len_list(self):
        with self.assertRaises(ValueError):
            physics.total_energy(np.array([0, 0]), [np.array([0, 0, 0])], [np.array([0, 0, 0])])

    def test_potential_energy_self(self):
        with self.assertRaises(ZeroDivisionError):
            physics.potential_energy(1., 1., np.array([0., 0., 0.]), np.array([0., 0., 0.]))

    def test_kin_energy_same_total_energy_same_particle(self):
        masses = np.array([10.])
        positions = [np.array([1., 1., 1.])]
        velocities = [np.array([10., 10., 10.])]
        self.assertEqual(physics.kinetic_energy(masses[0], velocities[0]),
                         physics.total_energy(masses, velocities, positions))

    def test_zero_kin_energy(self):
        self.assertEqual(physics.kinetic_energy(10, np.array([0., 0., 0.])), 0.)

    def test_calculate_radius(self):
        radius = 3.
        mass = 10.
        volume = (4./3.) * np.pi * radius ** 3
        density = mass / volume
        self.assertEqual(physics.calculate_radius(mass, density), radius)

    def test_calculate_radius(self):
        radius = 3.
        mass = 10.
        volume = (4./3.) * np.pi * radius ** 3
        density = mass / volume
        self.assertEqual(physics.calculate_radius(mass, density), radius)

    def test_calculate_volume(self):
        radius = 3.
        mass = 10.
        volume = (4./3.) * np.pi * radius ** 3
        density = mass / volume
        self.assertEqual(physics.calculate_volume(mass, density), volume)


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
        node = OctreeNode(np.linalg.norm(self.particle.position))
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
        node = OctreeNode(1)
        self.assertTrue(node.is_leaf)
        node._create_empty_child_nodes()
        for n in node.childnodes:
            self.assertTrue(n.is_leaf)

    def test_is_internal(self):
        node = OctreeNode(1)
        node._create_empty_child_nodes()
        self.assertFalse(node.is_leaf)
        self.assertTrue(node.is_internal_node)

    def test_is_interal_with_particles(self):
        node = OctreeNode(distance_to_center=100)
        for i in range(0, 10):
            node.insert_particle(Particle(10. + i, 0., 0., 0., 0., 0., 0., 1))
        self.assertTrue(node.is_internal_node)

    def test_is_external(self):
        node = OctreeNode(distance_to_center=1000)
        p = Particle(200., 200., 200., 0., 0., 0., 0., SUN_MASS)
        node.insert_particle(p)
        self.assertTrue(node.is_external_node)

    def test_create_empty_child_nodes(self):
        self.node._create_empty_child_nodes()
        self.assertFalse(self.node.is_leaf)
        self.assertTrue(len(self.node.childnodes) == 8)
        # Test that the center of all child nodes are contained in main node
        for child_node in self.node.childnodes:
            self.assertTrue(self.node._limiting_cube.contains_point(child_node._limiting_cube.center))

    def test_children_contained(self):
        node = OctreeNode(distance_to_center=100)
        node._create_empty_child_nodes()
        for child_node in node.childnodes:
            for vertex in child_node._limiting_cube.vertices:
                self.assertTrue(node._limiting_cube.contains_point(vertex))
                child_node._create_empty_child_nodes()
                for grand_childnode in child_node.childnodes:
                    for vertex2 in grand_childnode._limiting_cube.vertices:
                        contains = child_node._limiting_cube.contains_point(vertex2)
                        self.assertTrue(contains)

    def test_particle_contained_in_grandchildnodes(self):
        new_node = OctreeNode(distance_to_center=100)
        new_node.insert_particle(self.particle)
        new_node._create_empty_child_nodes()
        search_result_children = False
        search_result_grandchildren = False
        for child_node in new_node.childnodes:
            child_node._create_empty_child_nodes()
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
        new_node._create_empty_child_nodes()
        new_node.childnodes[0]._create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0]._create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0]._create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0]._create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0]._create_empty_child_nodes()
        new_node.childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0].childnodes[0]._create_empty_child_nodes()
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
        masses_list = _adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_negative_difference(self):
        masses_list = np.repeat((self.TOTAL_MASS / self.NUM_PARTICLES) * 2., self.NUM_PARTICLES)
        masses_list = _adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_negative_difference_variations(self):
        masses_list = [0.75, 0.5, 0.25, 0.1, 0.1, 0.1]
        masses_list = _adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_adjust_mass_positive_difference_variations(self):
        masses_list = [0.13, 0.25, 0.25, 0.01, 0.05, 0.1]
        masses_list = _adjust_mass(masses_list, self.TOTAL_MASS)
        self.assertAlmostEqual(sum(masses_list), self.TOTAL_MASS)

    def test_generate_random_positions_from_a_to_b(self):
        a = -5
        b = 10
        positions = _generate_random_positions_from_a_to_b(a, b, 100)
        for p in positions:
            self.assertTrue(a <= p[0] <= b)
            self.assertTrue(a <= p[0] <= b)
            self.assertTrue(a <= p[0] <= b)

    def test_generate_sphere_position_distribution(self):
        points = _generate_sphere_position_distribution(10, [0., 0., 0.], 100)
        sphere = Sphere(10, [0., 0., 0.])
        self.assertEqual(len(points), 100)
        for p in points:
            self.assertTrue(sphere.contains_point(p))

    def test_generate_sphere_position_distribution_unique_points(self):
        points = _generate_sphere_position_distribution(10, [0., 0., 0.], 100)
        self.assertEqual(len(points), len(set([tuple(p) for p in points])))


class TestCloudGeneration(unittest.TestCase):

    def test_ParameterReader_file(self):
        mock_json = {"mass": 2.0, "n_particles": 5000, "density": 1E-18, "temperature": 20,
                     "path": "./data/test_param_cloud.csv", "rotation":  0.,
                     "variation":  0.75, "center":  [5., 5., 5.]}

        with patch("__builtin__.open", create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)
            with patch("json.load", return_value=mock_json):
                reader = ParameterReader.from_configfile("dummy_path")
                self.assertEqual(reader.mass, mock_json["mass"])
                self.assertEqual(reader.n_particles, mock_json["n_particles"])
                self.assertEqual(reader.density, mock_json["density"])
                self.assertEqual(reader.temperature, mock_json["temperature"])
                self.assertEqual(reader.cloud_path, mock_json["path"])
                self.assertEqual(reader.rotation, mock_json["rotation"])
                self.assertEqual(reader.variation, mock_json["variation"])
                self.assertEqual(reader.center, mock_json["center"])

    def test_generate_cloud_individual_params(self):
        args = MagicMock()
        args.mass = 1.
        args.nparticles = 1000
        args.rho = 1E20
        args.temperature = 10.
        args.path = ''
        args.rotation = 0.
        args.variation = 0.
        args.config = None
        particles = generate_cloud(args, write_file=False)
        self.assertEqual(len(particles), 1000)
        self.assertAlmostEqual(particles[0].mass, 1. / 1000)

    def test_generate_cloud_config_file(self):
        args = MagicMock()
        args.config = 'dummy_path'
        mock_json = {"mass": 1.0, "n_particles": 1000, "density": 1E-18, "temperature": 20,
                     "path": "./data/test_param_cloud.csv", "rotation":  0.,
                     "variation":  0., "center":  [5., 5., 5.]}

        with patch("__builtin__.open", create=True) as mock_open:
            mock_open.return_value = MagicMock(spec=file)
            with patch("json.load", return_value=mock_json):
                particles = generate_cloud(args, write_file=False)
                self.assertEqual(len(particles), mock_json["n_particles"])
                self.assertAlmostEqual(particles[0].mass, mock_json["mass"] / mock_json["n_particles"])


class TestIntegration(unittest.TestCase):

    def test_leap_frog(self):
        tree = MagicMock()
        particles = [MagicMock() for i in range(0, 10)]
        accelerations_i = [0. for i in range(0, 10)]
        with patch("barneshut.barnes_hut_gravitational_acceleration", return_value=0.):
            resp = leapfrog_step(particles, tree, 0., accelerations_i)
            self.assertEqual(len(resp), len(accelerations_i))

    def test_leap_frog_different_sizes(self):
        tree = MagicMock()
        particles = [MagicMock() for i in range(0, 10)]
        accelerations_i = [0. for i in range(0, 11)]
        with self.assertRaises(ValueError):
            resp = leapfrog_step(particles, tree, 0., accelerations_i)


unittest.main()
