=======
EGraSPy
=======

Evolución Gravitacional de Sistema de Partículas en Python (Gravitational Evolution of Particles System, in python) GrEPS?

Based on original project in Fortran (https://github.com/jeudyx/EGraSP)

Executable scripts:

 * egrasp: main simulation
 * generate_cloud : creates and stores the initial distribution of particles
 * results_visualization


Modules:

 * astro_constants: constantes not available in scipy
 * barneshut: implementation of the Barnes-Hut approximation to gravitational interactions
 * integration: implementation of Leapfrog integration
 * octree: implementation of construction of the octree
 * physics: implementation of equations and formulae
 * structures: class definition for all the data structures used here.
 * neighbour_detection: Neighbour detection algorithm

----The content of this file is under construction----

Dependencies:
 * numpy
 * scipy
 * mpi4py
 * matplotlib
 * argparse


Units:
 * kg for mass
 * meters for distance
 * gr/cm³ for density
 * Kelvin for temperature
 * velocity in meters per second

Implemented algorithms:
 * Barnes-Hut for gravitational interactions
 * Closest Neighbours detection

Integration method:
 * Leap-frog

Parallelization: MPI (with mpi4py)

    * For parallel execution with mpi: mpirun -np 4 -host localhost python egrasp.py -cfg ./params/test_run.json (in this case, run in localhost in 4 processors)
    * SET LD_LIBRARY_PATH in bashrc to ~/libraries/mpich2-1.4.1p1/lib
    * In /etc/hosts, have a reference to the machine name to 127.0.0.1
    * Install mpich2 with ./configure CFLAGS=-fPIC --enable-shared