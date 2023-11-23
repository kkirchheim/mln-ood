#!/bin/bash
#SBATCH -N 2                 # 4 nodes total
#SBATCH --ntasks=32           # 8 total tasks
#SBATCH --cpus-per-task=4   # 16 CPU cores assigned to each task
#SBATCH --job-name=gtsrb-mining
#SBATCH --mem=80000          # 80 GB of memory per node
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log
#SBATCH -p short 

. /beegfs1/urz/utils/slurmProlog.sh  # output settings, check node healthy

echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list
module load mpi/openmpi-4.1 # message passing interface
module load cuda/toolkit-11.6.0 python/python-3.11.3 # for GPU-usage
echo "DEBUG: LOADEDMODULES=$LOADEDMODULES" # module list

[ "$SLURM_NNODES" ] && [ $SLURM_NNODES -lt 4 ] && mpidebug="--report-bindings"

HOSTFILE=outputs/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE # one entry per host

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=False
export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,compact,1,0

source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun -np $SLURM_NTASKS --bind-to none  --map-by slot:PE=$SLURM_CPUS_PER_TASK python3 -u evo-mining.py population_size=10 n_generations=1 log_every=1 ${@}

. /beegfs1/urz/utils/slurmEpilog.sh  # output final state, cleanup
