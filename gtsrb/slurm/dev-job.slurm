#!/bin/bash
#SBATCH -N 1                 # 4 nodes total
#SBATCH --ntasks=2           # 8 total tasks
#SBATCH --cpus-per-task=2   # 16 CPU cores assigned to each task
#SBATCH --job-name=gtsrb-mining
#SBATCH --mem=80000          # 80 GB of memory per node
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log

. /beegfs1/urz/utils/slurmProlog.sh  # output settings, check node healthy

echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list
module load mpi/openmpi-4.1 # message passing interface
module load cuda/toolkit-11.6.0 python/python-3.11.3 # for GPU-usage
echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list

[ "$SLURM_NNODES" ] && [ $SLURM_NNODES -lt 4 ] && mpidebug="--report-bindings"

HOSTFILE=outputs/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE # one entry per host
source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun -np $SLURM_NTASKS --map-by slot:PE=$SLURM_NTASKS_PER_NODE --bind-to core python3 -u evo-mining.py population_size=10 n_generations=1 log_every=1 ${@}

. /beegfs1/urz/utils/slurmEpilog.sh  # output final state, cleanup
