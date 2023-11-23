#!/bin/bash
#SBATCH -N 1                 # 4 nodes total
#SBATCH --ntasks=2           # 8 total tasks
#SBATCH --cpus-per-task=2   # 16 CPU cores assigned to each task
#SBATCH --job-name=ood-mining
#SBATCH --mem=80000          # 80 GB of memory per node
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.log
#SBATCH -p short

. /beegfs1/urz/utils/slurmProlog.sh  # output settings, check node healthy

HOSTFILE=outputs/slurm-$SLURM_JOBID.hosts
scontrol show hostnames $SLURM_JOB_NODELIST > $HOSTFILE # one entry per host

source $SCRATCH/$USER/venv311-openmpi/bin/activate && mpirun -np $SLURM_NTASKS --map-by slot:PE=$SLURM_NTASKS_PER_NODE --bind-to core python3 -u evo-mining.py population_size=10 n_generations=1 log_every=1 ${@}

. /beegfs1/urz/utils/slurmEpilog.sh  # output final state, cleanup
