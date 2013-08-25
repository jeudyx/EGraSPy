"""Implementation of Barnes-Hut algorithm to calculate gravitational interaction of particles
Uses the Octree structure
"""

# -*- coding: UTF-8 -*-

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

from physics import gravitational_acceleration
import numpy as np
from structures import OctreeNode, Particle


def barnes_hut_gravitational_acceleration(body, tree, theta=0.5):

    resp = np.array([0., 0., 0.])

    if tree.is_external:
        if body != tree.particle:
            return gravitational_acceleration(body.particles.position, tree.particle.position, tree.particle.mass)
        else:
            # If same particle, no acceleration
            return resp
    else:
        s = tree.cube_side
        d = np.linalg.norm(body.position - tree.center_of_mass)

        if s/d < theta:
            # If s/d < θ,treat this internal node as a single body,
            # and calculate the force it exerts on body b, and add this amount to b’s net force
            return gravitational_acceleration(body.particles.position, tree.center_of_mass, tree.mass)
        else:
            # Otherwise, run the procedure recursively on each of the current node’s children
            for child in tree.childnodes:
                resp += barnes_hut_gravitational_acceleration(body, child, theta)
            return resp
