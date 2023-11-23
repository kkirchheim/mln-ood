#!/bin/bash
#SBATCH -N 32                 # 4 nodes total
#SBATCH --ntasks=256           # 32 nodes x 16 tasks at 4 gpus total tasks
#SBATCH --cpus-per-task=1   # 16 CPU cores assigned to each task
#SBATCH --job-name=celeb-a-mining
#SBATCH --mem=250000          # 80 GB of memory per node
#SBATCH --time 24:00:00       # set 1h walltime (=maximum runtime), see sinfo
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

export OMP_NUM_THREADS=2
echo "Number of Tasks per node: $SLURM_NTASKS_PER_NODE"

# used to be 640
source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun --np 256 -map-by slot:PE=2 --bind-to hwthread python3 -u evo-mining.py ${@}
# source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun --oversubscribe --use-hwthread-cpus --np 256 -map-by slot:PE=2 --bind-to hwthread python3 -u evo-mining.py ${@}
# source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun -np $SLURM_NTASKS --map-by slot:PE=$SLURM_NTASKS_PER_NODE --bind-to core python3 -u evo-mining.py ${@}
# --oversubscribe -
. /beegfs1/urz/utils/slurmEpilog.sh  # output final state, cleanup
