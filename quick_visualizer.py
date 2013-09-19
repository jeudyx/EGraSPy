import numpy as np
from generate_cloud import load_particles_from_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import scipy.constants
import sys


def visualize_particles(particles):
    x = np.array([p.position[0] for p in particles]) / scipy.constants.au
    y = np.array([p.position[1] for p in particles]) / scipy.constants.au
    z = np.array([p.position[2] for p in particles]) / scipy.constants.au

    fig = plt.figure("Particles")

    ax = plt.axes(projection='3d')

    ax.plot(x, y, z, '.')

    plt.show()


def main(path):
    particles = load_particles_from_file(path)
    visualize_particles(particles)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except IndexError:
        print 'Need one parameter with the path of the file containing particles to visualize'
