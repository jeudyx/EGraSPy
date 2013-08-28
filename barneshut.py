"""Implementation of Barnes-Hut algorithm to calculate gravitational interaction of particles
Uses the Octree structure
"""

# -*- coding: UTF-8 -*-

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import numpy as np

from physics import gravitational_acceleration
from structures import OctreeNode, Particle
from generate_cloud import get_max_distance_positions

def barnes_hut_gravitational_acceleration(body, tree, theta=0.5):

    """

    :param body: a Particle
    :param tree: an OctreeNode tree
    :param theta: thredshold for s/d in BH algorithm
    :return:
    """
    resp = np.array([0., 0., 0.])

    if tree.is_external_node:
        if body != tree.particle:
            return gravitational_acceleration(body.position, tree.particle.position, tree.particle.mass)
        else:
            # If same particle, no acceleration
            return resp
    else:
        s = tree.cube_side
        d = np.linalg.norm(body.position - tree.center_of_mass)

        if s/d < theta:
            # If s/d < theta,treat this internal node as a single body,
            # and calculate the force it exerts on body b, and add this amount to b's net force
            return gravitational_acceleration(body.position, tree.center_of_mass, tree.mass)
        else:
            # Otherwise, run the procedure recursively on each of the current node's children
            for child in tree.childnodes:
                resp += barnes_hut_gravitational_acceleration(body, child, theta)
            return resp


def build_tree(positions, velocities, masses, densities):
    # Get the maximum distance to build tree. Too slow? profile and check
    max_dist = get_max_distance_positions(positions)
    tree = OctreeNode(distance_to_center=max_dist)
    for i, r in enumerate(positions):
        # x,y,z,vx,vy,vz,mass,rho,temp
        p = Particle.from_nparray(np.array([r[0], r[1], r[2], velocities[i][0], velocities[i][1],
                                            velocities[i][2], masses[i], densities[i], 0.]))
        tree.insert_particle(p)
    return tree