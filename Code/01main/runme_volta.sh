#!/bin/bash

# runme.sh is a shell script for executing GPU MPI application
# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh
#------------------------------------------------------------------

# load the reqired modules:
module load cuda/12.2.2
module load openmpi/4.1.8-gcc-8.5.0

mpirun=$(which mpirun)

# remove what should be removed
rm a.out
# rm *.res *.inf a.out

# How many mpi procs should run
if [ $# -lt 1 ]; then
    echo $0: usage: runme.sh nprocs
    exit 1
fi
nprocs=$1
#export CUDA_VISIBLE_DEVICES=6,7
#nvcc -arch=sm_52 --compiler-bindir mpic++ --compiler-options -O3  MPI_EV_ZY.cu -DD_x=$2 -DD_y=$3 -DD_z=$4 -DNPARS1=$5 -DNBX=$6 -DNBY=$7 -DNBZ=$8
nvcc -arch=sm_70 --compiler-bindir mpic++ --compiler-options -O3  MPI_EV_ZY.cu -DD_x=$2 -DD_y=$3 -DD_z=$4 -DNPARS1=$5 -DNBX=$6 -DNBY=$7 -DNBZ=$8

# run3

run_cmd="-np $nprocs -rf gpu_rankfile_volta --mca pml ob1 --mca btl self,vader,tcp"

echo $mpirun $run_cmd

$mpirun $run_cmd ./a.out