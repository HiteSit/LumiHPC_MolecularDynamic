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
#SBATCH --account=project_465000973

cat << EOF > select_gpu
#!/bin/bash

# Export the Python Path
export PATH="/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin:$PATH"

# Distribute the GPU for the ranks equally
export ROCR_VISIBLE_DEVICES=\$((SLURM_LOCALID % SLURM_GPUS_PER_NODE))

# Only assign GPUs to ranks > 0 (workers)
if [ \$SLURM_PROCID -gt 0 ]; then
    # Adjust the calculation to account for rank 0 not using a GPU
    worker_id=\$((SLURM_PROCID - 1))
    export ROCR_VISIBLE_DEVICES=\$((worker_id % SLURM_GPUS_PER_NODE))
else
    # For rank 0, set an invalid GPU ID or leave it unset
    export ROCR_VISIBLE_DEVICES=""
fi

exec \$*
EOF

chmod +x ./select_gpu

cat << EOF > md_config_full_apo.toml
[mode]
type = "apo"

[paths]
protein_folder = "./Apo_Structures"
protein_glob_pattern = "*.cif"

[system]
delta_pico = 0.002
rerun = false

[nvt]
steps = 400
dcd_save = 50
log_save = 1

[npt]
steps = 400
dcd_save = 50
log_save = 1

[md]
steps = 150000          # 150 ns
dcd_save = 100          # 100 ps
log_save = 10           # 10 ps
EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --output=MD_Wrapper_%t.log --error=MD_Wrapper_%t.error ./select_gpu python MPI_MD_Wrapper.py md_config_full_apo.toml

rm -rf ./select_gpu
rm -rf ./md_config_full_apo.toml