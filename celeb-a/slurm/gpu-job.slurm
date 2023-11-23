#!/bin/bash
#SBATCH -N 1                 # 4 nodes total
#SBATCH --ntasks=8           # 32 nodes x 16 tasks at 4 gpus total tasks
#SBATCH --cpus-per-task=1   # 16 CPU cores assigned to each task
#SBATCH --job-name=celeb-a-mining
#SBATCH --mem=80000          # 80 GB of memory per node
#SBATCH --time 24:00:00       # set 1h walltime (=maximum runtime), see sinfo
#SBATCH --output=logs/%j.log
#SBATCH --gres=gpu:a30:1     # alloc nodes with one A30-GPU (since Nov24)
#SBATCH --error=logs/%j.log

. /beegfs1/urz/utils/slurmProlog.sh  # output settings, check node healthy

echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list
module load mpi/openmpi-4.1 # message passing interface
module load cuda/toolkit-11.6.0 python/python-3.11.3 # for GPU-usage
echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list

[ "$SLURM_NNODES" ] && [ $SLURM_NNODES -lt 4 ] && mpidebug="--report-bindings"

HOSTFILE=outputs/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE # one entry per host

export OMP_NUM_THREADS=2
echo "Number of Tasks per node: $SLURM_NTASKS_PER_NODE"

source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun --np 8 -map-by slot:PE=2 --bind-to hwthread python3 -u evo-mining.py ${@} device="cuda"
. /beegfs1/urz/utils/slurmEpilog.sh  # output final state, cleanup
