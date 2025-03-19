#!/bin/bash -l
#SBATCH --job-name=MD_Wrapper
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --time=02:00:00
#SBATCH --account=project_XXXX

export PATH="/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin:$PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python XXX