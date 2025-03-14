# LumiHPC_MolecularDynamic

A high-performance molecular dynamics simulation package optimized for LUMI supercomputer using OpenMM and MPI parallelization.

## Overview

LumiHPC_MolecularDynamic is a specialized package designed to run large-scale molecular dynamics simulations on LUMI (Large Unified Modern Infrastructure) supercomputer. The package leverages OpenMM for molecular dynamics calculations and MPI for parallel processing across multiple nodes and GPUs, enabling efficient utilization of HPC resources.

The package supports:
- Protein-only (APO) simulations
- Protein-ligand complex simulations
- Parallel execution of multiple simulations using MPI
- Analysis of simulation results

## Requirements

- Python 3.8+
- OpenMM 7.7+
- mpi4py
- OpenFF Toolkit
- PDBFixer
- Parmed
- PyTraj
- ROCM (for AMD GPUs on LUMI) or CUDA (for NVIDIA GPUs)
- tomli

## Installation

It's recommended to use the provided Singularity container on LUMI, which includes all required dependencies:

```bash
# Using the pre-configured Singularity environment
export PATH="/scratch/project_XXXXX/Singularity_Envs/cheminf_rocm/env_cheminf_rocm/bin:$PATH"
```

If you prefer to install the dependencies manually:

```bash
# Create and activate a conda environment
conda create -n md_env python=3.8
conda activate md_env

# Install OpenMM and related packages
conda install -c conda-forge openmm openff-toolkit pdbfixer parmed pytraj mpi4py

# Install additional dependencies
pip install tomli cloudpickle
```

## Directory Structure

- `HPC_MD/`: Core molecular dynamics implementation
  - `Vanilla_MD.py`: Main MD simulation class
  - `Analysis_Lig.py`: Analysis tools for ligand-protein interactions
  - `Calc_MMPBSA.py`: MM-PBSA free energy calculations

- `Wrappers_HPC/`: Scripts for running on HPC environments
  - `MPI_MD_Wrapper.py`: MPI-enabled wrapper for parallel simulations
  - `MPI_MD_Wrapper_RUN.sh`: Slurm submission script for LUMI

- `Wrappers_Local/`: Scripts for running on local machines
  - `LOCAL_MPI_MD_Wrapper.py`: Local MPI wrapper
  - `LOCAL_MPI_MD_Wrapper_RUN.sh`: Local execution script

- `examples/`: Example input files
  - Sample protein structures (PDB, CIF)
  - Sample ligand structures (SDF)

# Core Molecular Dynamics Components

## Running Simulations with Vanilla_MD.py

The core of this package is the `Run_MD` class in `Vanilla_MD.py`, which provides a simple interface to run complete molecular dynamics simulations. This wrapper handles all aspects of the simulation process, including:

- System preparation with force field assignment
- Energy minimization
- Temperature and pressure equilibration with appropriate restraints
- Production molecular dynamics
- Output file generation

### How to Run a Complete Simulation

The most straightforward way to run a simulation is using the `equilibration_production` function:

```python
from HPC_MD.Vanilla_MD import Run_MD

# Define simulation settings as dictionaries
system_settings = {
    "receptor_path": "protein.pdb",  # Path to protein structure
    "ligand_path": "ligand.sdf",     # Path to ligand (use "APO" for protein-only)
    "delta_pico": 0.002,            # Time step in picoseconds
    "rerun": False,                 # Set to True to continue from checkpoint
    "output_dir": "output_folder"   # Output directory
}

# Settings for temperature equilibration
nvt_settings = {
    "temps_list_simulating": [100, 200, 300],  # Gradually heat the system
    "steps": 3000,                            # Total steps for heating
    "dcd_save": 100,                          # Save trajectory every 100 steps
    "log_save": 10                            # Save log every 10 steps
}

# Settings for pressure equilibration with decreasing restraints
npt_settings = {
    "rests_list_decreasing": [100, 10, 1, 0.1],  # Gradually release restraints
    "atoms_to_restraints": {"CA", "C", "N"},     # Restrain backbone atoms
    "steps": 4000,
    "dcd_save": 100,
    "log_save": 10
}

# Settings for production MD
md_settings = {
    "steps": 50000,    # Total production steps (100 ns at 0.002 ps/step)
    "dcd_save": 100,   # Save trajectory every 100 steps
    "log_save": 10     # Save log every 10 steps
}

# Run the complete workflow
Run_MD.equilibration_production(system_settings, nvt_settings, npt_settings, md_settings)
```

### Continuing a Simulation from Checkpoint

To restart a simulation from a previously saved checkpoint:

```python
# Same settings as before, but with rerun=True
system_settings = {
    "receptor_path": "protein.pdb",
    "ligand_path": "ligand.sdf",
    "delta_pico": 0.002,
    "rerun": True,                  # Enable restart from checkpoint
    "output_dir": "output_folder"
}

# Run will automatically detect and load the checkpoint file
Run_MD.equilibration_production(system_settings, nvt_settings, npt_settings, md_settings)
```

### Output Files

The simulation generates the following key files in the output directory:

- `system.prmtop`: Full system topology (with water)
- `system_noWAT.prmtop`: System topology with water removed (for analysis)
- `Minimized.pdb`: Structure after energy minimization
- `Step1_Nvt.dcd`: Trajectory from NVT equilibration
- `Step1_Nvt.log`: Energy data from NVT equilibration
- `Step2_Npt.dcd`: Trajectory from NPT equilibration 
- `Step2_Npt.log`: Energy data from NPT equilibration
- `Step3_Md_Rep0.dcd`: Production MD trajectory
- `Step3_Md_Rep0.log`: Production MD energy data
- `Step3_Md_Rep0.chk`: Checkpoint file for restart

## MM-PBSA Calculations with Calc_MMPBSA.py

The `Calc_MMPBSA.py` module enables binding free energy calculations using the MM-PBSA approach. It can be used for individual trajectories or for batch processing of multiple systems.

### Running MM-PBSA for a Single System

```python
from HPC_MD.Calc_MMPBSA import Calc_MMPBSA

# Initialize MM-PBSA calculation for a single system
mmpbsa = Calc_MMPBSA(
    MMPBSA_dir="./MMPBSA_output",          # Output directory
    system_prmtop="./system.prmtop",       # System topology file with water
    traj_dcd="./Step3_Md_Rep0.dcd",        # Trajectory from production MD
    percentage=0.1                         # Analyze 10% of frames
)

# Run the calculation
mmpbsa.run_MMPBSA()

# This will create files in the MMPBSA_output directory:
# - mmpbsa.in: Input file for MMPBSA.py
# - receptor.prmtop: Topology for the receptor only
# - ligand.prmtop: Topology for the ligand only
# - complex.prmtop: Topology for the complex without water
# - MMPBSA.csv: Results in CSV format
```

### Batch Processing Multiple Systems

For multiple ligand-protein complexes, use the `Wrapper_MMPBSA` class:

```python
from HPC_MD.Calc_MMPBSA import Wrapper_MMPBSA

# Define paths for multiple systems
systems_dict = {
    "ligand1": {
        "PRMTOP": "./ligand1/system.prmtop",
        "DCD": "./ligand1/Step3_Md_Rep0.dcd"
    },
    "ligand2": {
        "PRMTOP": "./ligand2/system.prmtop",
        "DCD": "./ligand2/Step3_Md_Rep0.dcd"
    },
    "ligand3": {
        "PRMTOP": "./ligand3/system.prmtop",
        "DCD": "./ligand3/Step3_Md_Rep0.dcd"
    }
}

# Initialize the wrapper
mmpbsa_wrapper = Wrapper_MMPBSA(systems_dict)

# Run calculations and get results as DataFrame
results_df = mmpbsa_wrapper()

# The results_df will contain:
# - Ligand names
# - GB binding energies (Generalized Born)
# - PB binding energies (Poisson-Boltzmann)

# Print and analyze results
print(results_df)

# Sort by binding energy
sorted_results = results_df.sort_values(by='GB')
print("Ligands ranked by binding affinity (GB):")
print(sorted_results)
```

## Trajectory Analysis with Analysis_Lig.py

The `Analysis_Lig.py` module provides powerful tools for analyzing MD trajectories, including RMSD/RMSF calculations, PCA, clustering, and visualization.

### Basic Usage

```python
from HPC_MD.Analysis_Lig import find_matching_directories, create_analyzer_dict
from HPC_MD.Analysis_Lig import plot_RMSF_inplace, plot_RMSD_inplace, plot_PCA_inplace

# Find directories containing MD simulation results
md_dirs = ["ligand1", "ligand2", "ligand3"]  # Exact names or regex patterns
file_paths = find_matching_directories(md_dirs)

# Create analyzer objects for each directory
analyzer_dict = create_analyzer_dict(file_paths, overwrite=False)

# Now you can use various plot functions

# Plot RMSF (Root Mean Square Fluctuation)
plot_RMSF_inplace(analyzer_dict)

# Plot RMSD (Root Mean Square Deviation)
rmsd_data, rmsd_summary = plot_RMSD_inplace(analyzer_dict)
print(rmsd_summary)  # Print summary statistics

# Plot PCA (Principal Component Analysis)
plot_PCA_inplace(analyzer_dict)

# Plot Radius of Gyration
from HPC_MD.Analysis_Lig import plot_Radius_inplace
plot_Radius_inplace(analyzer_dict)

# Plot Gaussian Network Model
from HPC_MD.Analysis_Lig import plot_Gaussian_inplace
plot_Gaussian_inplace(analyzer_dict)
```

### Interactive Trajectory Visualization

The package also includes tools for interactive visualization of trajectories using NGLView:

```python
from HPC_MD.Analysis_Lig import TrajectoryViewer

# Create a viewer for a specific simulation
viewer = TrajectoryViewer(analyzer_dict, "ligand1")

# Display the interactive visualization
view = viewer()
view  # Display in Jupyter notebook

# The visualization includes:
# - Protein backbone as cartoon, colored by residue index
# - Ligand in licorice representation
# - Binding site residues (within 5.5Å of ligand) in licorice representation
```

### Creating Trajectory Archives

For sharing or backup purposes, you can create ZIP archives of processed trajectories:

```python
from HPC_MD.Analysis_Lig import create_trajectory_archive

# Create a ZIP archive containing key files from all simulations
create_trajectory_archive("trajectories.zip", analyzer_dict)

# The archive will contain for each directory:
# - Topology file (PRMTOP)
# - Trajectory file (XTC and DCD)
# - PDB structure file
# - Cluster representative structures
```

## Configuration

The simulation is configured using TOML files. Here's an example configuration:

```toml
[mode]
type = "apo"  # Options: "apo" for protein-only, "lig" for protein-ligand

[paths]
protein_folder = "./Apo_Structures"
protein_glob_pattern = "*.cif"
# For lig mode: fixed_receptor_path = "./path/to/receptor.pdb"

[system]
delta_pico = 0.002  # Time step in picoseconds
rerun = false       # Whether to continue from checkpoint

[nvt]
steps = 400         # NVT equilibration steps
dcd_save = 50       # Save trajectory every N steps
log_save = 1        # Save log every N steps

[npt]
steps = 400         # NPT equilibration steps
dcd_save = 50
log_save = 1

[md]
steps = 150000      # Production MD steps (150 ns with 0.002 ps timestep)
dcd_save = 100      # Save trajectory every 100 steps (0.2 ns)
log_save = 10       # Save log every 10 steps (0.02 ns)
```

## Running Simulations on LUMI

### 1. Prepare your job script

Modify the provided `MPI_MD_Wrapper_RUN.sh` script to suit your needs:

```bash
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
#SBATCH --account=project_XXXXX

# Update the account number and other parameters as needed
```

### 2. Prepare your configuration file

Create a TOML configuration file (as shown in the Configuration section) or use the one generated by the script.

### 3. Submit your job

```bash
sbatch Wrappers_HPC/MPI_MD_Wrapper_RUN.sh
```

## Running APO Simulations

For protein-only simulations:

1. Place your protein structures in a directory (e.g., `./Apo_Structures`)
2. Create a configuration with `mode.type = "apo"`
3. Specify the protein folder and glob pattern in the configuration
4. Run the MPI wrapper:

```bash
python Wrappers_HPC/MPI_MD_Wrapper.py your_config.toml
```

## Running Protein-Ligand Simulations

For protein-ligand complex simulations:

1. Prepare your receptor (protein) and ligand files
2. Create a configuration with `mode.type = "lig"`
3. Specify the paths to your receptor and ligand files
4. Run the MPI wrapper:

```bash
python Wrappers_HPC/MPI_MD_Wrapper.py your_config.toml
```

## Parallel Analysis with MPI_Analysis.py

The package includes a powerful MPI-based analysis framework that allows you to process multiple simulation directories in parallel. This is especially useful when you need to analyze large numbers of trajectories on HPC systems.

### Basic Usage

The MPI_Analysis script uses the master-worker pattern to distribute analysis tasks across multiple processors:

```bash
# Basic usage with MPI
mpirun -n <num_processes> python MPI_Analysis.py <dir_pattern1> [<dir_pattern2> ...] [--overwrite]

# Example: analyze all directories matching 'complex_*' using 4 processes
mpirun -n 4 python MPI_Analysis.py 'complex_*'

# Example: analyze multiple directory patterns
mpirun -n 8 python MPI_Analysis.py 'complex_*' 'protein_*'

# Example: use overwrite flag to regenerate analysis files
mpirun -n 4 python MPI_Analysis.py 'complex_*' --overwrite
```

### How It Works

1. The master process (rank 0) finds all directories matching the specified patterns
2. Work items are dynamically distributed to worker processes
3. Each worker processes one directory at a time and reports results back to the master
4. As workers complete their tasks, the master assigns new directories until all work is done
5. A progress bar shows the overall completion status
6. Summary statistics are provided upon completion

### Key Features

- **Dynamic Load Balancing**: Automatically distributes work based on worker availability
- **Error Handling**: Continues processing other directories even if some fail
- **Progress Tracking**: Shows real-time progress with tqdm
- **Summary Reports**: Provides detailed success/failure statistics

### Required Files

For each simulation directory, the following files must exist:

- `*.dcd`: Original trajectory with water
- `system.prmtop`: System topology file with water
- `system_noWAT.prmtop`: System topology without water
- `*_noWAT.xtc`: Trajectory without water in XTC format
- `*_noWAT.dcd`: Trajectory without water in DCD format
- `Minimized_noWAT.pdb`: Structure file without water
- `Clusters.pdb`: Cluster representative structures

### Using the Processed Data

After running the MPI_Analysis script, you can use the generated files with the analysis functions from Analysis_Lig:

```python
from HPC_MD.Analysis_Lig import find_matching_directories, create_analyzer_dict

# Create analyzer dictionary using the same directory patterns
md_dirs = ["complex_*", "protein_*"]
file_paths = find_matching_directories(md_dirs)
analyzer_dict = create_analyzer_dict(file_paths)

# Now you can use this dictionary with all the analysis functions
# described in the "Trajectory Analysis" section
```

### Running on LUMI or Other HPC Systems

For HPC environments, create a job script:

```bash
#!/bin/bash
#SBATCH --job-name=MD_Analysis
#SBATCH --output=analysis_%j.out
#SBATCH --error=analysis_%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=04:00:00
#SBATCH --account=project_XXXXX

# Load required modules
module load cray-python

# Run the MPI analysis
srun python MPI_Analysis.py 'complex_*' 'protein_*'
```

Save the script as `run_analysis.sh` and submit with:

```bash
sbatch run_analysis.sh
```

## Output Files

The simulation generates several output files:

- `*.dcd`: Trajectory files in DCD format
- `*.log`: Log files with energy information
- `*.chk`: Checkpoint files for restart
- `*.prmtop`: AMBER parameter/topology files
- `*.nc`: NetCDF format trajectory files (for analysis)
- `*.pdb`: Structure files

## Advanced Features

### Positional Restraints

The system supports various types of restraints:
- Backbone restraints for protein stability
- Water restraints during equilibration
- Custom restraints on specific atom selections

### Checkpointing and Restart

Simulations can be restarted from checkpoints by setting `rerun = true` in the configuration.

### Multi-GPU Scaling

The MPI wrapper automatically distributes simulations across available GPUs, with each simulation assigned to a specific GPU based on the MPI rank.

## Performance Considerations

- For optimal performance on LUMI, use 1 MPI rank per GPU
- Set `OMP_NUM_THREADS` to match the number of CPUs per task
- Use mixed precision (`Precision: 'mixed'`) for better performance

## License

[Specify your license information here]