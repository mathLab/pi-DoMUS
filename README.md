# pi-DoMUS [![Build Status](https://travis-ci.org/mathLab/pi-DoMUS.svg?branch=master)](https://travis-ci.org/mathLab/pi-DoMUS)
Parallel-Deal.II MUlti-physics Solver

# Required Packages
- Trilinos  (https://trilinos.org)
- deal.II   (www.dealii.org)
- deal2lkit (https://github.com/mathlab/deal2lkit)
- Sundials  (http://computation.llnl.gov/casc/sundials/main.html)


# Installation Instructions

Compile and install pidomus:

    git clone git@github.com:mathLab/pi-DoMUS.git

    cd pi-DoMUS
    mkdir build
    cd build

    cmake -DCMAKE_INSTALL_PREFIX=/some/where/pidomus ..
    make
    make install

and then start working with one of its examples by e.g. copying it

    cp -r /some/where/pidomus/examples/name-of-example /some/other/place/my-app
    cd /some/other/place/my-app
    cmake -DPIDOMUS_DIR=/some/where/pidomus .
    make run

if you have an environment variable called `PIDOMUS_DIR` it will look into that directory and you do not have to specify it as option (`-DPIDOMUS_DIR`).
