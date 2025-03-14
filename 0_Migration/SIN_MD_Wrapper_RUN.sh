#!/bin/bash -l
#SBATCH --job-name=MD_Wrapper
#SBATCH --output=MD_Wrapper.log
#SBATCH --error=MD_Wrapper.error
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --time=00:30:00
#SBATCH --account=project_465000973

# Set up the conda environment
export PATH="/users/fuscoric/Documents/Enviroments/cheminf_container/bin:$PATH"

# List of ligand paths
run_dict=(
    "./Apo_Structures/9bf9_Dimer_LAG3-MHCII.cif:Dimer_LAG3-MHCII:0"
    "./Apo_Structures/9bf9_Mono_LAG3-MHCII.cif:Mono_LAG3-MHCII:1"
    "./Apo_Structures/9bf9_Dimer_LAG3.cif:Dimer_LAG3:2"
    "./Apo_Structures/9bf9_Mono_LAG3.cif:Mono_LAG3:3"
)

export ROCR_VISIBLE_DEVICES="0,1,2,3"

for run in "${run_dict[@]}"; do
    IFS=':' read -ra params <<< "$run"
    protein_path="${params[0]}"
    ligand_path="${params[1]}"
    gpu_id="${params[2]}"
    
    python SIN_MD_Wrapper.py "$protein_path" "$ligand_path" "$gpu_id" &
done
wait