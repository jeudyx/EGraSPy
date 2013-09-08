__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import numpy as np
import scipy as sp
import unittest
from structures import OctreeNode, Particle
from generate_cloud import _generate_sphere_position_distribution, generate_cloud, \
    load_particles_from_file, get_max_distance
from physics import gravitational_acceleration, brute_force_gravitational_acceleration, norm
from mock import MagicMock, patch
import matplotlib.pyplot as plt
from barneshut import barnes_hut_gravitational_acceleration, build_tree, adjust_tree
from integration import leapfrog_step, get_system_total_energy
from astro_constants import SUN_MASS, EARTH_MASS
#from mpl_toolkits.mplot3d import Axes3D


class TestParticleDistributionVisualization(unittest.TestCase):

    def setUp(self):
        self.points = _generate_sphere_position_distribution(10, [0., 0., 0.], 10000)

    def xtest_sphere(self):
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]

        fig = plt.figure("TestParticleDistributionVisualization.test_sphere")

        ax = plt.axes(projection='3d')

        ax.plot(x, y, z, '.')

        plt.show()


class TestTreeConstruction(unittest.TestCase):

    PATH = './data/testcloud.csv'

    def setUp(self):
        self.particles = []
        self.total_cloud_mass = 0.
        raw_data = np.loadtxt(self.PATH, delimiter=',', skiprows=1)
        max_distance = max(raw_data[:,6])
        self.node = OctreeNode(distance_to_center=max_distance)
        for line in raw_data:
            # noinspection PyArgumentList
            self.particles.append(Particle(line[0], line[1], line[2], line[3], line[4], line[5], line[8], line[7]))
            self.total_cloud_mass += line[7]

    def test_create(self):
        for p in self.particles:
            self.node.insert_particle(p)
        self.assertEqual(self.total_cloud_mass, self.node.mass)
        self.assertEqual(self.node.n_particles, len(self.particles))
        self.assertEqual(self.node.num_populated_leaves, len(self.particles))


class TestCalculationsIntegrationAndTree(unittest.TestCase):

    def setUp(self):
        self.args = MagicMock()
        self.args.mass = 1.
        self.args.nparticles = 100
        self.args.rho = 1E-14
        self.args.temperature = 10.
        self.args.path = './data/functional_test_cloud.csv'
        self.args.rotation = 0.
        self.args.variation = 0.75
        self.args.config = None
        self.particles = generate_cloud(self.args)

    def test_write_read_cloud(self):
        read_particles = load_particles_from_file(self.args.path)
        self.assertEqual(len(self.particles), len(read_particles))
        for i, p in enumerate(self.particles):
            q = read_particles[i]
            self.assertTrue(p == q)
            self.assertEqual(p.mass, q.mass)

    def test_max_distance(self):
        read_particles = load_particles_from_file(self.args.path)
        raw_vals = np.loadtxt(self.args.path, delimiter=',', skiprows=1)
        mx = max([np.linalg.norm(i) for i in [[e[0], e[1], e[2]] for e in raw_vals]])
        self.assertEqual(mx, get_max_distance(read_particles))

    def test_barnes_hut_accuracy(self):
        raw_vals = np.loadtxt(self.args.path, delimiter=',', skiprows=1)
        positions = raw_vals[:, 0:3]
        velocities = raw_vals[:, 3:6]
        masses = raw_vals[:, 6:7]
        densities = raw_vals[:, 7:8]

        tree = build_tree(positions, velocities, masses, densities)

        for p in self.particles:
            brute = brute_force_gravitational_acceleration(p, particles=self.particles)
            barnes_hut = barnes_hut_gravitational_acceleration(p, tree=tree, theta=0.0)
            self.assertAlmostEqual(np.log(np.linalg.norm(brute)), np.log(np.linalg.norm(barnes_hut)), places=4)

    def test_compare_adjusted_tree(self):
        positions = np.array([[n, n, n] for n in np.arange(10.0)])
        velocities = np.array([[n, n, n] for n in np.zeros(10.0)])
        masses = np.ones(10)
        densities = np.zeros(10)
        particles = []
        tree = build_tree(positions, velocities, masses, densities, out_particles=particles)
        print str(tree)
        particles[0].position = np.array([1.5, 1.5, 1.5])
        particles[5].position = np.array([9.5, 9.5, 9.5])
        #particles[9].position = np.array([21.5, 21.5, 21.5])
        adjust_tree(tree, tree)
        print '---------------------------------------------------'
        print str(tree)


class TestParticleGenerationVisualization(unittest.TestCase):

    def xtest_generate_cloud(self):

        args = MagicMock()
        args.config = '../params/test_cloud.json'
        particles = generate_cloud(args, write_file=False)
        visualize_particles(particles)


def visualize_particles(particles):
    x = [p.position[0] for p in particles]
    y = [p.position[1] for p in particles]
    z = [p.position[2] for p in particles]

    fig = plt.figure("TestParticleGenerationVisualization.test_generate_cloud")

    ax = plt.axes(projection='3d')

    ax.plot(x, y, z, '.')

    plt.show()


class TestIntegration(unittest.TestCase):

    PATH = './data/testcloud.csv'

    def setUp(self):
        path = './data/functional_test_cloud.csv'
        raw_vals = np.loadtxt(path, delimiter=',', skiprows=1)
        positions = raw_vals[:, 0:3]
        velocities = raw_vals[:, 3:6]
        masses = raw_vals[:, 6:7]
        densities = raw_vals[:, 7:8]
        self.particles = []
        self.tree = build_tree(positions, velocities, masses, densities, out_particles=self.particles)

    def xtest_energy_conserved(self):
        accelerations_i = np.zeros(len(self.particles))
        e_i = get_system_total_energy(self.particles)
        for i in range(0, 10):
            accelerations_i = leapfrog_step(self.particles, self.tree, 1000000., accelerations_i)
        e_f = get_system_total_energy(self.particles)
        self.assertAlmostEqual(e_i, e_f, places=2)

    def test_small_system_conservation(self):
        star1 = Particle(-10.9 * sp.constants.astronomical_unit, 0., 0., 0., 2.1E3, 0., 0., SUN_MASS*1.09)
        star2 = Particle(12.8 * sp.constants.astronomical_unit, 0., 0., 0., -2.1E3, 0., 0., SUN_MASS*0.9)
        planet = Particle(-9.9 * sp.constants.astronomical_unit, 0., 0., 0., -3.1E4, 0., 0., EARTH_MASS)
        particles = [star1, star2, planet]
        tree = OctreeNode(distance_to_center=(10.9+12.8)* sp.constants.astronomical_unit)
        tree.insert_particle(star1)
        tree.insert_particle(star2)
        tree.insert_particle(planet)
        steps = 10000 #366 * 100 * 24
        dt = 60. * 60.
        accelerations_i = np.array([[n, n, n] for n in np.zeros(3.0)])
        e_i = get_system_total_energy(particles)
        while steps:
            accelerations_i = leapfrog_step(dt, accelerations_i, barnes_hut_gravitational_acceleration,
                                            particles=particles, tree=tree, theta=0.5)

            accelerations_i = leapfrog_step(dt, accelerations_i, brute_force_gravitational_acceleration,
                                            particles=particles)

            #print '%s ==========================================' % steps
            #print accelerations_i
            #print [p.velocity for p in particles]
            #print '%s ==========================================' % steps
            # adjust_tree(tree, tree)
#            tree = OctreeNode(distance_to_center=max([norm(p.position) for p in particles]))
#            tree.insert_particle(star1)
#            tree.insert_particle(star2)
#            tree.insert_particle(planet)

            steps -= 1
            if steps % 1000 == 0:
                print "Steps: %s" % steps

        e_f = get_system_total_energy(particles)
        self.assertAlmostEqual(e_i/e_f, 1., places=2)

unittest.main()