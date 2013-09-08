"""Implementation of Barnes-Hut algorithm to calculate gravitational interaction of particles
Uses the Octree structure
"""

# -*- coding: UTF-8 -*-

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import numpy as np

from physics import gravitational_acceleration, center_of_mass_minus_particle, norm
from structures import OctreeNode, Particle
from generate_cloud import get_max_distance_positions


def barnes_hut_gravitational_acceleration(body, **kwargs):
    """

    :param body: a Particle
    :param tree: an OctreeNode tree
    :param theta: thredshold for s/d in BH algorithm
    :return:
    """

    tree = kwargs['tree']
    theta = kwargs.get('theta', 0.5)

    if tree.is_external_node:
        if body != tree.particle:
            return gravitational_acceleration(body.position, tree.particle.position, tree.particle.mass)
        else:
            # If same particle, no acceleration
            return np.zeros(3)
    else:
        s = tree.cube_side
        try:
            d = norm(body.position - tree.center_of_mass)
        except Warning:
            d = 1
            pass

        if s / d < theta:
            # If s/d < theta,treat this internal node as a single body,
            # and calculate the force it exerts on body b, and add this amount to b's net force
            return gravitational_acceleration(body.position, tree.center_of_mass, tree.mass)
        else:
            # Otherwise, run the procedure recursively on each of the current node's children
            resp = np.zeros(3)
            for child in tree.childnodes:
                if child.mass == 0. or child.n_particles == 0:
                    continue
                resp += barnes_hut_gravitational_acceleration(body, tree=child, theta=theta)
            return resp


def build_tree(positions, velocities, masses, densities, out_particles=[]):
    """
    Build an Octree from given values
    :param positions:
    :param velocities:
    :param masses:
    :param densities:
    :param out_particles:
    :return:
    """
    # Get the maximum distance to build tree. Too slow? profile and check
    max_dist = get_max_distance_positions(positions)
    tree = OctreeNode(distance_to_center=max_dist)
    for i, r in enumerate(positions):
        # x,y,z,vx,vy,vz,mass,rho,temp
        p = Particle.from_nparray(np.array([r[0], r[1], r[2], velocities[i][0], velocities[i][1],
                                            velocities[i][2], masses[i], densities[i], 0.]))
        tree.insert_particle(p)
        out_particles.append(p)
    return tree


def adjust_tree(current_node, root_node):
    if current_node.is_external_node:
        if not current_node.contains_particle(current_node.particle):
            # Particle position has changed, need to reinsert!
            # Reinsert implies: changing mass and center of mass of upper level! (recursive)

            particle = current_node.particle
            _remove_particle_from_center_of_mass(current_node, current_node.particle)
            current_node.particle = None
            root_node.insert_particle(particle)
    else:
        # Continue checking tree tree in next level
        for node in current_node.childnodes:
            adjust_tree(node, root_node)


def need_to_rebuild(current_node, root_node):
    if current_node.is_external_node:
        try:
            if not current_node.parent_node.parent_node.parent_node.contains_particle(current_node.particle):
                return True
            else:
                return False
        except AttributeError:
            return False
    else:
        # Continue checking tree tree in next level
        return any([need_to_rebuild(node, root_node) for node in current_node.childnodes])


def _remove_particle_from_center_of_mass(node, particle):
    """
    Substract particle contribution to center of mass and total mass of given node
    Recursively do the same for all upper levels
    :param node: Node from where to substract particle from center of mass
    :param particle: particle to substract
    """

    node.n_particles -= 1

    if node.n_particles > 0.:
        node.center_of_mass = center_of_mass_minus_particle(node.mass, node.center_of_mass, particle.mass,
                                                            particle.position)

    node.mass -= particle.mass

    if node.mass < 0.:
        raise ValueError('Mass can not be less than zero. Mass: %s, particle: %s' % (node.mass, str(particle)))

    if node.parent_node:
        _remove_particle_from_center_of_mass(node.parent_node, particle)
