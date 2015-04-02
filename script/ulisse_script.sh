#!/bin/bash

cd "$PBS_O_WORKDIR/../build/"

. /home/mathlab/gnu.conf /home/mathlab/gnu/

make distclean
cmake ..
make

mpirun -np 20 ./step-32 > output.dat

