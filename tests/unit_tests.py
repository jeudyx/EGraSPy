__author__ = 'jeudy'

# -*- coding: UTF-8 -*-

import numpy as np
import unittest
import physics
from structures import OctreeNode

class TestPhysics(unittest.TestCase):

    def setUp(self):
        # Metodo similar al constructor donde se inicializan datos de prueba
        self.ri = np.random.random(3)
        self.rj = np.random.random(3)

    def test_zero_mass_acceleration(self):
        self.assertEqual(np.linalg.norm(physics.gravitational_acceleration(self.ri, self.rj, 0)), 0.)

    def test_is_leaf(self):
        node = OctreeNode()
        self.assertTrue(node.is_leaf)