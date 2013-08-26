__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
from structures import OctreeNode, Particle
from generate_cloud import _generate_sphere_position_distribution, generate_cloud, \
    load_cloud_from_file, get_max_distance
from physics import gravitational_acceleration, brute_force_gravitational_acceleration
from mock import MagicMock, patch
import matplotlib.pyplot as plt
from barneshut import barnes_hut_gravitational_acceleration
from mpl_toolkits.mplot3d import Axes3D


class TestParticleDistributionVisualization(unittest.TestCase):

    def setUp(self):
        self.points = _generate_sphere_position_distribution(10, [0., 0., 0.], 10000)

    def test_sphere(self):
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


class TestGravitationalAcelerationCalculationAndTreeStuff(unittest.TestCase):

    def setUp(self):
        self.args = MagicMock()
        self.args.mass = 1.
        self.args.nparticles = 100
        self.args.rho = 1E20
        self.args.temperature = 10.
        self.args.path = './data/functional_test_cloud.csv'
        self.args.rotation = 0.
        self.args.variation = 0.75
        self.args.config = None
        self.particles = generate_cloud(self.args)

    def test_write_read_cloud(self):
        read_particles = load_cloud_from_file(self.args.path)
        self.assertEqual(len(self.particles), len(read_particles))
        for i, p in enumerate(self.particles):
            q = read_particles[i]
            self.assertTrue(p == q)
            self.assertEqual(p.mass, q.mass)

    def test_max_distance(self):
        read_particles = load_cloud_from_file(self.args.path)
        raw_vals = np.loadtxt(self.args.path, delimiter=',', skiprows=1)
        mx = max([np.linalg.norm(i) for i in [[e[0], e[1], e[2]] for e in raw_vals]])
        self.assertEqual(mx, get_max_distance(read_particles))

    def test_barnes_hut_accuracy(self):
        tree = OctreeNode(distance_to_center=1.)
        for i, p in enumerate(self.particles):
            tree.insert_particle(p)

        for p in self.particles:
            brute = brute_force_gravitational_acceleration(p, self.particles)
            barnes_hut = barnes_hut_gravitational_acceleration(p, tree, theta=0.1)
            self.assertAlmostEqual(np.log(np.linalg.norm(brute)), np.log(np.linalg.norm(barnes_hut)), places=3)


class TestParticleGenerationVisualization(unittest.TestCase):

    def test_generate_cloud(self):

        args = MagicMock()
        args.config = '../params/test_cloud.json'
        particles = generate_cloud(args, write_file=False)


        x = [p.position[0] for p in particles]
        y = [p.position[1] for p in particles]
        z = [p.position[2] for p in particles]

        fig = plt.figure("TestParticleGenerationVisualization.test_generate_cloud")

        ax = plt.axes(projection='3d')

        ax.plot(x, y, z, '.')

        plt.show()

unittest.main()