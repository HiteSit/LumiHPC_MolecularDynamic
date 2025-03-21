#!/usr/bin/env python3

import os
import argparse
from typing import Dict, Any, Optional

"""
HPC Job Script Generator
=======================

A flexible toolkit for generating HPC job scripts for molecular dynamics simulations and analysis.

This module provides functions to create customized SLURM batch scripts for running 
MD simulations and analysis jobs on HPC systems. It handles all the necessary SBATCH 
parameters, environment setup, and GPU distribution logic.

Key Features:
------------
- Generate complete SBATCH scripts for MD simulations with custom parameters
- Generate SBATCH scripts for analysis runs with custom parameters
- Support for TOML configuration for MD simulations
- Proper GPU distribution logic for MPI workloads
- Command-line interface for easy integration into workflows

Usage Modes:
-----------
1. As a command-line tool:
   ```
   python Call_Runner.py --type md --wrapper-path /path/to/MPI_MD_Wrapper.py --output job.sh
   python Call_Runner.py --type analysis --analysis-script analysis.py --script-args "--input data.dcd" 
   ```

2. By importing the functions in other Python scripts:
   ```python
   from Call_Runner import generate_md_runner

   script = generate_md_runner(
       job_name="MD_Simulation",
       nodes=2,
       wrapper_path="/path/to/MPI_MD_Wrapper.py"
   )
   ```

Functions:
---------
- generate_md_runner: Create a script for running MD simulations
- generate_analysis_runner: Create a script for running analysis jobs
- main: Command-line interface entry point

Dependencies:
------------
- Required: os, argparse
- Optional: toml (for loading TOML configuration files)
"""


def generate_md_runner(
    # SBATCH parameters
    job_name: str = "MD_Wrapper",
    output: str = "/dev/null",
    error: str = "/dev/null",
    partition: str = "dev-g",
    nodes: int = 1,
    ntasks_per_node: int = 2,
    gpus_per_node: int = 2,
    cpus_per_task: int = 7,
    time: str = "02:00:00",
    account: str = "project_XXXX",
    # Application specific parameters
    toml_config: Dict[str, Any] = None,
    wrapper_path: str = "MPI_MD_Wrapper.py",
    python_path: str = "/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin",
) -> str:
    """
    Generate a SLURM batch script for running molecular dynamics simulations.
    
    This function creates a complete SLURM script with all necessary SBATCH parameters,
    environment setup, GPU distribution logic, and TOML configuration for MD simulations.
    
    Parameters:
    -----------
    job_name : str, default="MD_Wrapper"
        Name of the SLURM job
    output : str, default="/dev/null"
        Path for STDOUT redirection
    error : str, default="/dev/null"
        Path for STDERR redirection
    partition : str, default="dev-g"
        SLURM partition to use
    nodes : int, default=1
        Number of nodes to allocate
    ntasks_per_node : int, default=2
        Number of MPI tasks per node
    gpus_per_node : int, default=2
        Number of GPUs per node
    cpus_per_task : int, default=7
        Number of CPU cores per task
    time : str, default="02:00:00"
        Job time limit in format HH:MM:SS
    account : str, default="project_XXXX"
        Project account to charge
    toml_config : Dict[str, Any], optional
        TOML configuration as a nested dictionary. You only need to provide 
        the sections you want to customize. Common usage patterns:
        
        ```python
        # Basic APO mode
        custom_config = {
            "mode": {"type": "apo"},
            "paths": {"protein_folder": "./MyProteins"}
        }
        
        # Ligand mode with custom settings
        custom_config = {
            "mode": {"type": "lig"},
            "paths": {
                "ligand_folder": "./H3_Ligands",
                "ligand_glob_pattern": ".*.sdf",
                "fixed_receptor_path": "./LAG3_Moloc.pdb"
            },
            "md": {"steps": 10000}
        }
        
        # Legacy mode with specific runs
        custom_config = {
            "mode": {"type": "legacy"},
            "legacy": {
                "runs": {
                    "Run1": {"protein": "./protein1.pdb", "ligand": "./ligand1.sdf"},
                    "Run2": {"protein": "./protein2.pdb", "ligand": "APO"}
                }
            }
        }
        ```
        
        Any unspecified sections will use the default values.
    wrapper_path : str, default="MPI_MD_Wrapper.py"
        Path to the MPI MD wrapper script
    python_path : str, default="/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin"
        Path to the Python environment
        
    Returns:
    --------
    str
        Complete SLURM batch script as a string
        
    Notes:
    ------
    The TOML configuration focuses on the most commonly changed parameters:
    1. The mode (apo, lig, legacy)
    2. The paths specific to each mode
    3. The MD steps for production runs
    
    You only need to specify the sections you want to customize, and all other
    settings will use sensible defaults.
    """
    # Define the default TOML configuration with all standard sections
    default_config = {
        "mode": {
            "type": "apo"  # Default to APO mode
        },
        "paths": {
            # APO mode settings
            "protein_folder": "./Apo_Structures",
            "protein_glob_pattern": "*.cif",
            # LIG mode settings
            "ligand_folder": "./Ligands",
            "ligand_glob_pattern": "*.sdf",
            "fixed_receptor_path": "./receptor.pdb"
        },
        "system": {
            "delta_pico": 0.002,
            "rerun": False
        },
        "nvt": {
            "steps": 400,
            "dcd_save": 50,
            "log_save": 1,
            "temps_list": [50, 100, 150, 200, 250, 300, 301]
        },
        "npt": {
            "steps": 400,
            "dcd_save": 50,
            "log_save": 1,
            "rests_list": [1000000000, 100000, 1000, 100, 10, 1],
            "atoms_to_restraints": ["CA"]
        },
        "md": {
            "steps": 150000,
            "dcd_save": 100,
            "log_save": 10
        }
    }
    
    # If user provided any configuration, merge it with the default
    merged_config = default_config.copy()
    
    if toml_config is not None:
        # Handle special case where md is provided as a single value
        if "md" in toml_config and not isinstance(toml_config["md"], dict):
            # Convert simple value to proper dict
            md_steps = toml_config["md"]
            toml_config["md"] = {"steps": int(md_steps)}
        
        # Handle each top-level section separately for proper merging
        for section, values in toml_config.items():
            if section in merged_config and isinstance(values, dict) and isinstance(merged_config[section], dict):
                # Update existing section with user values
                merged_config[section].update(values)
            else:
                # Replace or add entire section
                merged_config[section] = values
        
        # Special handling for mode-specific paths
        # If mode has been explicitly set, filter out paths that aren't relevant to that mode
        if "mode" in toml_config and "type" in toml_config["mode"]:
            mode_type = toml_config["mode"]["type"]
            
            # For legacy mode, completely remove the paths section
            if mode_type == "legacy":
                if "paths" in merged_config:
                    del merged_config["paths"]
            else:
                mode_specific_paths = {}
                
                # Copy over all user-specified paths first
                if "paths" in toml_config:
                    mode_specific_paths.update(toml_config["paths"])
                
                # Keep only relevant default paths that haven't been overridden
                if mode_type == "apo":
                    # For APO mode, only keep protein-related paths
                    keys_to_keep = ["protein_folder", "protein_glob_pattern"]
                    for key in keys_to_keep:
                        if key not in mode_specific_paths and key in merged_config["paths"]:
                            mode_specific_paths[key] = merged_config["paths"][key]
                elif mode_type == "lig":
                    # For LIG mode, only keep ligand-related paths
                    keys_to_keep = ["ligand_folder", "ligand_glob_pattern", "fixed_receptor_path"]
                    for key in keys_to_keep:
                        if key not in mode_specific_paths and key in merged_config["paths"]:
                            mode_specific_paths[key] = merged_config["paths"][key]
                
                # Update the paths section with only the relevant paths
                merged_config["paths"] = mode_specific_paths
    
    # Create temporary script content
    script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --account={account}

cat << EOF > select_gpu
#!/bin/bash

# Export the Python Path
export PATH="{python_path}:$PATH"

# Distribute the GPU for the ranks equally
export ROCR_VISIBLE_DEVICES=$((SLURM_LOCALID % SLURM_GPUS_PER_NODE))

# Only assign GPUs to ranks > 0 (workers)
if [ $SLURM_PROCID -gt 0 ]; then
    # Adjust the calculation to account for rank 0 not using a GPU
    worker_id=$((SLURM_PROCID - 1))
    export ROCR_VISIBLE_DEVICES=$((worker_id % SLURM_GPUS_PER_NODE))
else
    # For rank 0, set an invalid GPU ID or leave it unset
    export ROCR_VISIBLE_DEVICES=""
fi

exec $*
EOF

chmod +x ./select_gpu

cat << EOF > md_config.toml
"""
    
    # Add TOML configuration - ensure all sections from merged_config are included
    for section, values in merged_config.items():
        # Special handling for legacy.runs which is a nested structure
        if section == "legacy" and isinstance(values, dict) and "runs" in values:
            # For legacy mode, we only want the legacy.runs section, not a separate legacy section
            script_content += "[legacy.runs]\n"
            for run_name, run_config in values["runs"].items():
                if isinstance(run_config, dict):
                    # Format: Run1 = { protein = "path.pdb", ligand = "path.sdf" }
                    # Make sure to follow TOML inline table format exactly
                    pairs = []
                    for k, v in run_config.items():
                        if isinstance(v, str):
                            pairs.append(f'{k} = "{v}"')
                        else:
                            pairs.append(f'{k} = {v}')
                    run_str = ", ".join(pairs)
                    script_content += f'{run_name} = {{ {run_str} }}\n'
            script_content += "\n"
        else:
            script_content += f"[{section}]\n"
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, bool):
                        script_content += f"{key} = {str(value).lower()}\n"
                    elif isinstance(value, str):
                        script_content += f'{key} = "{value}"\n'
                    elif isinstance(value, list):
                        # Format: temps_list = [50, 100, 150, 200, 250, 300, 301]
                        list_str = str(value).replace("'", '"')  # Ensure proper TOML format for lists
                        script_content += f"{key} = {list_str}\n"
                    else:
                        script_content += f"{key} = {value}\n"
            script_content += "\n"
    
    script_content += f"""EOF

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun --output=MD_Wrapper_%t.log --error=MD_Wrapper_%t.error ./select_gpu python {wrapper_path} md_config.toml

rm -rf ./select_gpu
rm -rf ./md_config.toml
"""
    
    return script_content


def generate_analysis_runner(
    # SBATCH parameters
    job_name: str = "MD_Analysis",
    output: str = "/dev/null",
    error: str = "/dev/null",
    partition: str = "dev-g",
    nodes: int = 1,
    ntasks_per_node: int = 2,
    gpus_per_node: int = 2,
    cpus_per_task: int = 7,
    time: str = "02:00:00",
    account: str = "project_XXXX",
    # Application specific parameters
    analysis_script: str = "analysis_script.py",
    python_path: str = "/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin",
    script_args: str = "",
) -> str:
    """
    Generate a SLURM batch script for running molecular dynamics analysis.
    
    This function creates a complete SLURM script with all necessary SBATCH parameters,
    environment setup, and command execution for analysis jobs.
    
    Parameters:
    -----------
    job_name : str, default="MD_Analysis"
        Name of the SLURM job
    output : str, default="/dev/null"
        Path for STDOUT redirection
    error : str, default="/dev/null"
        Path for STDERR redirection
    partition : str, default="dev-g"
        SLURM partition to use
    nodes : int, default=1
        Number of nodes to allocate
    ntasks_per_node : int, default=2
        Number of MPI tasks per node
    gpus_per_node : int, default=2
        Number of GPUs per node
    cpus_per_task : int, default=7
        Number of CPU cores per task
    time : str, default="02:00:00"
        Job time limit in format HH:MM:SS
    account : str, default="project_XXXX"
        Project account to charge
    analysis_script : str, default="analysis_script.py"
        Path to the analysis Python script
    python_path : str, default="/scratch/project_465001750/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin"
        Path to the Python environment
    script_args : str, default=""
        Command-line arguments to pass to the analysis script
        
    Returns:
    --------
    str
        Complete SLURM batch script as a string
        
    Notes:
    ------
    The generated script is simpler than the MD script as it doesn't need
    GPU selection logic or TOML configuration. It just sets up the environment
    and runs the analysis script with the provided arguments.
    
    Example:
    --------
    >>> from Call_Runner import generate_analysis_runner
    >>> script = generate_analysis_runner(
    ...     job_name="RMSD_Analysis",
    ...     analysis_script="HPC_MD/Analysis_Lig.py",
    ...     script_args="--input trajectory.dcd --output results/"
    ... )
    >>> with open("analysis_job.sh", "w") as f:
    ...     f.write(script)
    """
    script_content = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={output}
#SBATCH --error={error}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --account={account}

export PATH="{python_path}:$PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
srun python {analysis_script} {script_args}
"""
    
    return script_content


def main() -> None:
    """
    Command-line interface for the HPC job script generator.
    
    This function parses command-line arguments and generates the appropriate
    SLURM batch script based on the requested job type (MD or analysis).
    
    Parameters:
    -----------
    None (uses command-line arguments)
    
    Returns:
    --------
    None
    
    Notes:
    ------
    The generated script is either printed to stdout or written to a file,
    depending on the value of the --output parameter.
    
    For MD jobs, a TOML configuration can be provided via the --toml-config parameter.
    If provided, the configuration is loaded from the file and used instead of the defaults.
    
    Example Usage:
    -------------
    # Generate an MD job script with custom parameters
    python Call_Runner.py --type md --job-name CustomMD --nodes 2 --wrapper-path /path/to/MPI_MD_Wrapper.py
    
    # Generate an analysis job script and print to stdout
    python Call_Runner.py --type analysis --analysis-script Analysis_Lig.py --output -
    
    # Generate an MD job script with a custom TOML configuration
    python Call_Runner.py --type md --toml-config simulation.toml --output md_job.sh
    """
    parser = argparse.ArgumentParser(description="Generate HPC job scripts for MD simulations and analysis")
    parser.add_argument("--type", choices=["md", "analysis"], required=True, help="Type of job to generate")
    parser.add_argument("--output", default="job_script.sh", help="Output file for the job script")
    
    # Add arguments for all parameters
    # SBATCH common parameters
    parser.add_argument("--job-name", default=None, help="Name of the job")
    parser.add_argument("--slurm-output", default=None, help="Path for stdout")
    parser.add_argument("--slurm-error", default=None, help="Path for stderr")
    parser.add_argument("--partition", default=None, help="Slurm partition")
    parser.add_argument("--nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Tasks per node")
    parser.add_argument("--gpus-per-node", type=int, default=None, help="GPUs per node")
    parser.add_argument("--cpus-per-task", type=int, default=None, help="CPUs per task")
    parser.add_argument("--time", default=None, help="Time limit (HH:MM:SS)")
    parser.add_argument("--account", default=None, help="Project account")
    parser.add_argument("--python-path", default=None, help="Path to Python environment")
    
    # MD specific arguments
    parser.add_argument("--toml-config", default=None, help="Path to TOML config file")
    parser.add_argument("--wrapper-path", default=None, help="Path to MPI_MD_Wrapper.py")
    
    # Analysis specific arguments
    parser.add_argument("--analysis-script", default=None, help="Path to analysis script")
    parser.add_argument("--script-args", default="", help="Arguments to pass to the analysis script")
    
    args = parser.parse_args()
    
    # Prepare arguments for function call
    kwargs = {}
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None and arg_name not in ["type", "output", "toml_config", "slurm_output", "slurm_error"]:
            # Convert argument names to function parameter names
            if arg_name == "slurm_output":
                kwargs["output"] = arg_value
            elif arg_name == "slurm_error":
                kwargs["error"] = arg_value
            else:
                kwargs[arg_name.replace("-", "_")] = arg_value
    
    # Load TOML config if provided
    if args.toml_config and args.type == "md":
        try:
            import toml
            with open(args.toml_config, 'r') as f:
                kwargs["toml_config"] = toml.load(f)
        except (ImportError, FileNotFoundError) as e:
            print(f"Error loading TOML configuration: {e}")
            return
    
    # Generate script based on type
    if args.type == "md":
        script_content = generate_md_runner(**kwargs)
    else:
        script_content = generate_analysis_runner(**kwargs)
    
    # Write script to file or print to stdout
    if args.output == "-":
        print(script_content)
    else:
        with open(args.output, 'w') as f:
            f.write(script_content)
        os.chmod(args.output, 0o755)  # Make executable
        print(f"Script written to {args.output}")


if __name__ == "__main__":
    main() 