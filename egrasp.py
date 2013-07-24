"""Main script from where the simulation is started

"""

__author__ = 'Jeudy Blanco - jeudyx@gmail.com'

import sys
import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(description='Starts the simulation')
    parser.add_argument("file", help="Path of the file with the cloud data")
    parser.add_argument("-t", "--theta", help="Barnes-Hut theta parameter", default=0.7)
    parser.add_argument("-s", "--savefrequency", help="How often do you want to save the state of the system?", default=100)
    parser.add_argument("-i", "--startfrom", help="From which iteration you want to start?", default=0)
    parser.add_argument("-j", "--maxiter", help="How many iterations you want to run?", default=1)
    parser.add_argument("-v", "--verbose", help="Displays debuging messages", default=False)
    parser.add_argument("-cfg", "--config", help="Path to a config file containing all the parameters. "
                                                 "This file must be in json format. "
                                                 "If given, it will override any other optional parameter")
    args = parser.parse_args()

if __name__ == "__main__":
    sys.exit(main())
