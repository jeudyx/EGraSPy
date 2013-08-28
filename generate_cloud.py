# -*- coding: utf-8 -*-

"""Script to generate an initial distribution of particles representing an interstellar gas cloud
The cloud is stored in a csv (comma separated value) file with the following format:
x, y, z, vx, vy, vz, mass, density, temperature

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse
import numpy as np
import csv
import json
from scipy.constants import parsec
from structures import Sphere, Particle
from physics import calculate_radius
from astro_constants import SUN_MASS

DEFAULT_CENTER = np.array([0., 0., 0.])


class ParameterReader(object):

    def __init__(self, mass, n_particles, density, temperature, path, rotation, variation, center):
        self.mass = mass
        self.n_particles = n_particles
        self.density = density
        self.temperature = temperature
        self.cloud_path = path
        self.rotation = rotation
        self.variation = variation
        self.center = center

    @classmethod
    def from_configfile(cls, config_file_path):
        json_data=open(config_file_path)
        data = json.load(json_data)
        mass = data["mass"]
        n_particles = data["n_particles"]
        density = data["density"]
        temperature = data["temperature"]
        cloud_path = data["path"]
        rotation = data["rotation"]
        variation = data["variation"]
        center = data["center"]
        return cls(mass, n_particles, density, temperature, cloud_path, rotation, variation, center)

def _generate_random_positions_from_a_to_b(a, b, n_particles):
    """ Generates 3D positions from a to b using: (b - a) * random_sample() + a
    Assumes that a is less than b

    :param a: lower limit of range
    :param b: upper limit of range
    :param n_particles: number of particles
    :return: numpy array of 3D vector between a and b
    """
    return (b - a) * np.random.random_sample((n_particles,3)) + a


def _generate_sphere_position_distribution(radius, center, n_particles):

    """Generate particle distribution in the shape of s sphere

    :param radius: radius of sphere
    :param center: 3D vector for the center of the sphere (numpy array)
    :param n_particles: number of particles
    :return: numpy array of n_particles x 3 elements with 3D position
    """
    containing_sphere = Sphere(radius, center)
    preliminary_positions = _generate_random_positions_from_a_to_b(-radius, radius, n_particles)

    preliminary_positions = [tuple(p) for p in preliminary_positions if containing_sphere.contains_point(p)]
    current_len = len(preliminary_positions)

    while current_len < n_particles:
        extension_list = [tuple(p) for p in _generate_random_positions_from_a_to_b(-radius, radius, n_particles - current_len)
                          if containing_sphere.contains_point(p)]
        preliminary_positions.extend(extension_list)
        preliminary_positions = list(set(preliminary_positions))    # Ensure uniqueness
        current_len = len(preliminary_positions)

    return np.array(preliminary_positions)


def generate_positions(radius, center, n_particles, dist_type='sphere'):
    if dist_type == "sphere":
        return _generate_sphere_position_distribution(radius, center, n_particles)


def generate_velocity_distribution(n_particles, rotation):
    return np.repeat(0., n_particles*3).reshape(n_particles, 3)

def generate_mass_distribution(total_mass, n_particles, max_variation=0):
    """ Generate mass distribution

    :param total_mass: total mass to distribute
    :param n_particles: number of particles
    :param max_variation: maximum variation (from 0 to 1) in particle mass
    :return: numpy array with mass distribution
    """
    particle_mass = total_mass / n_particles
    masses_list = np.repeat(particle_mass, n_particles)

    if max_variation:
        # Determine if the variation will sum or substract from the particle mass
        signs = np.array([x if x != 0 else 1 for x in np.random.randint(-1,high=2,size=n_particles)])
        # Variation percentage
        variations = np.array([x if x <= max_variation else 0 for x in np.random.random(n_particles)]) * signs
        mass_variations = masses_list * variations
        # Masses with the variation
        masses_list = masses_list + mass_variations
        # Need to ajust if the total mass is not the desired one
        masses_list = _adjust_mass(masses_list, total_mass)

    return masses_list


def _adjust_mass(masses_list, total_mass):
    """ If the sum of masses_list is different than total_mass, need to adjust

    :param masses_list: The list of masses to adjust
    :param total_mass:  The correct value of total mass for the distribution
    :return: Adjusted list of masses
    """
    difference = total_mass - sum(masses_list)
    particle_difference = np.abs(difference) / len(masses_list)

    if difference == 0:
        return masses_list
    elif difference > 0:
        # The list has less mass, need to add to each particle
        masses_list = masses_list + np.repeat(particle_difference, len(masses_list))
    else:
        # The list has more mass, need to substract, but careful if the particle
        # difference is greater than the particle mass
        masses_list = np.array([x - particle_difference if x - particle_difference > 0 else x for x in masses_list])
        # If still different mass, need to call recursively
        if total_mass != sum(masses_list):
            return _adjust_mass(masses_list, total_mass)

    return masses_list


def _write_values(masses, positions, velocities, density, temperature, path):
    with open(path, 'wb') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',')
        datawriter.writerow('x,y,z,vx,vy,vz,mass,rho,temp'.split(','))
        for i, mass in enumerate(masses):
            row = []
            row.extend(positions[i].tolist())
            row.extend(velocities[i].tolist())
            row.append(mass)
            row.append(density)
            row.append(temperature)
            datawriter.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Generates a distribution of particles '
                                                 'representing an interstellar gas cloud')
    parser.add_argument("-m", "--mass", help="Total mass of cloud (in solar masses)", default=1.)
    parser.add_argument("-np", "--nparticles", help="Number of particles", default=1000)
    parser.add_argument("-d", "--rho", help="Mean density in gr/cm^3", default=1E-20)
    parser.add_argument("-t", "--temperature", help="Temperature of the cloud (in Kelvin)", default=10)
    # parser.add_argument("-s", "--shape", help="Shape of the cloud: sphere, disc, cilinder", default='sphere')
    parser.add_argument("-p", "--path", help="Path", default='./data/egraspy_cloud.csv')
    parser.add_argument("-r", "--rotation", help="Initial percentage of rotational "
                                                 "energy with respect to gravitational energy",
                        default=0)
    parser.add_argument("-var", "--variation", help="Maximum allowed variation percentage in "
                                                    "particles mass", default=0)
    parser.add_argument("-cfg", "--config", help="Path to a config file containing all the parameters. "
                                                 "This file must be in json format. "
                                                 "If given, it will override any other parameter")
    parser.add_argument("-v", "--verbose", help="Displays debuging messages", default=False)
    args = parser.parse_args()

    generate_cloud(args)


def load_particles_from_file(path):
    particle_list = []
    raw_data = np.loadtxt(path, delimiter=',', skiprows=1)
    for p in raw_data:
        particle_list.append(Particle.from_nparray(p))
    return particle_list


def get_max_distance(particles):
    return get_max_distance_positions([p.position for p in particles])

def get_max_distance_positions(positions):
    return max([np.linalg.norm(i) for i in [[r[0], r[1], r[2]] for r in positions]])

def generate_cloud(args, write_file=True):

    if args.config:
        reader = ParameterReader.from_configfile(args.config)
    else:
        # noinspection PyArgumentList
        reader = ParameterReader(args.mass, args.nparticles, args.rho, args.temperature,
                                 args.path, args.rotation, args.variation, DEFAULT_CENTER)

    # Assume for now spherical distribution centered on 0,0,0
    # Since mass comes in solar masses, and density in gr/cm³, need to convert to kg and kg/m³
    # result is expresses in parsecs
    radius = calculate_radius(reader.mass * SUN_MASS, reader.density*1000.0) / parsec
    positions = generate_positions(radius, reader.center, reader.n_particles)
    masses = generate_mass_distribution(reader.mass, reader.n_particles, reader.variation)
    velocities = generate_velocity_distribution(reader.n_particles, reader.rotation)
    if write_file:
        _write_values(masses, positions, velocities, reader.density, reader.temperature, reader.cloud_path)

    # For testing purposes only?
    particles_list = []

    for i, mass in enumerate(masses):
        p = Particle(positions[i][0], positions[i][1], positions[i][2],
                     velocities[i][0], velocities[i][1], velocities[i][2],
                     reader.density, mass)

        particles_list.append(p)

    return particles_list

if __name__ == "__main__":
    sys.exit(main())
