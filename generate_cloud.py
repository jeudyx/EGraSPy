"""Script to generate an initial distribution of particles representing an interstellar gas cloud
The cloud is stored in a csv (comma separated value) file with the following format:
x, y, z, vx, vy, vz, mass, density, temperature

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse


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
