#!/bin/bash
#SBATCH -N 32                 # 4 nodes total
#SBATCH --ntasks=128           # 8 total tasks
#SBATCH --cpus-per-task=2   # 16 CPU cores assigned to each task
#SBATCH --job-name=ood-mining
#SBATCH --mem=80000
#SBATCH --array=0-10         # We'll have 11 values of alpha (0.0 ... 0.1)
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
alpha_list=($(seq 0 0.01 0.1))

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
mpirun -np $SLURM_NTASKS \
  --map-by slot:PE=$SLURM_NTASKS_PER_NODE --bind-to core \
  python3 -u evo-mining.py \
    alpha=$alpha \
    hydra.run.dir=./outputs/alpha-sweep-${alpha}

# --- Epilog ---
. /beegfs1/urz/utils/slurmEpilog.sh