__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
from structures import OctreeNode, Particle
from generate_cloud import generate_sphere_position_distribution

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TestParticleDistributionVisualization(unittest.TestCase):

    def setUp(self):
        self.points = generate_sphere_position_distribution(10, [0., 0., 0.], 10000)

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