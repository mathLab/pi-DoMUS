#!/bin/bash

cd $PBS_O_WORKDIR

module load intel/14.0
module load openmpi/1.8.3/intel/14.0
make
mpirun -np 20 ./step-32 > output.sh

