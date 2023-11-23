#!/bin/bash
#SBATCH -N 32                 # 4 nodes total
#SBATCH --ntasks=512           # 8 total tasks
#SBATCH --cpus-per-task=2   # 16 CPU cores assigned to each task
#SBATCH --job-name=celeb-logic-sweep
#SBATCH --mem=250000
#SBATCH --array=0-9         # We'll have 11 values of alpha (0.0 ... 0.1)
#SBATCH --time=24:00:00 
#SBATCH --output=logs/%j_%a.log
#SBATCH --error=logs/%j_%a.log

# --- Prolog (load modules, env, etc.) ---
. /beegfs1/urz/utils/slurmProlog.sh

module load mpi/openmpi-4.1
module load cuda/toolkit-11.6.0 python/python-3.11.3

# Create a hostfile (if needed by your MPI setup)
HOSTFILE=outputs/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE

# Activate your Python virtual environment
source $SCRATCH/$USER/venv311-openmpi/bin/activate

# Define alpha_list as an array of 0.00 -> 0.10 in steps of 0.01
alpha_list=($(seq 1 10))

# Based on the array index, pick the alpha
alpha="${alpha_list[$SLURM_ARRAY_TASK_ID]}"

echo "Running job $SLURM_ARRAY_TASK_ID with alpha=$alpha"

# Now run your Python script using mpirun.
# We pass two Hydra overrides:
#  1) alpha=$alpha
#  2) hydra.run.dir=./outputs/multirun-${alpha}
#
# This ensures each run ends up in a unique subdirectory like:
#    ./outputs/multirun-0.00, ./outputs/multirun-0.01, ...

source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun --np 512 -map-by slot:PE=2 --bind-to hwthread python3 -u evo-mining.py hydra.run.dir=./outputs/logic-sweep-${alpha}  objective._target_=evo.LogicFitness ${@}

# --- Epilog ---
. /beegfs1/urz/utils/slurmEpilog.sh
