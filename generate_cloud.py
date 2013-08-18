"""Script to generate an initial distribution of particles representing an interstellar gas cloud
The cloud is stored in a csv (comma separated value) file with the following format:
x, y, z, vx, vy, vz, mass, density, temperature

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse
import numpy as np

def read_params(path):
    return


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
        masses_list = adjust_mass(masses_list, total_mass)

    return masses_list


def adjust_mass(masses_list, total_mass):
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
            return adjust_mass(masses_list, total_mass)

    return masses_list

def main(argv=None):
    parser = argparse.ArgumentParser(description='Generates a distribution of particles '
                                                 'representing an interstellar gas cloud')
    parser.add_argument("-m", "--mass", help="Total mass of cloud (in solar masses)", default=1)
    parser.add_argument("-np", "--nparticles", help="Number of particles", default=1000)
    parser.add_argument("-d", "--rho", help="Mean density", default=1E-20)
    parser.add_argument("-t", "--temperature", help="Temperature of the cloud (in Kelvin)", default=10)
    parser.add_argument("-s", "--shape", help="Shape of the cloud: sphere, disc, cilinder", default='sphere')
    parser.add_argument("-p", "--path", help="Path", default='./egraspy_cloud.csv')
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

if __name__ == "__main__":
    sys.exit(main())
