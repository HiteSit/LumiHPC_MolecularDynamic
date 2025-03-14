import os
import glob
import sys
import tomli
import subprocess
from pathlib import Path
from mpi4py import MPI
import parmed as pmd
import pytraj as pt
from src.Vanilla_MD import Run_MD

def md_wrapper(config, input_path, dirpath=None, gpu_id=None, is_legacy=False, legacy_run_name=None, legacy_ligand_path=None, output_dir=None):
    """
    Run MD simulation based on configuration
    
    Args:
        config: Dictionary containing configuration settings
        input_path: Path to protein file (APO) or ligand file (LIG)
        dirpath: Directory path for output (only used in APO mode)
        gpu_id: GPU ID to use for computation
        is_legacy: Flag indicating if this is a legacy mode run
        legacy_run_name: Name of the legacy run (used as directory name)
        legacy_ligand_path: Path to the ligand file in legacy mode
        output_dir: Full path to output directory (used in legacy mode)
    """
    delta_pico = config.get('system', {}).get('delta_pico', 0.002)
    
    # Configure settings based on mode
    mode_type = config.get('mode', {}).get('type', '').lower()
    
    if is_legacy:
        # Legacy mode - input_path is protein path, dirpath is run name, legacy_ligand_path is ligand path or "APO"
        receptor_path = input_path
        ligand_path = legacy_ligand_path
        is_apo_mode = ligand_path == "APO"
        item_type = "protein-only" if is_apo_mode else "protein-ligand complex"
        
        # If output_dir is provided, use it for the MD run
        output_path = output_dir if output_dir else os.getcwd()
    else:
        # Standard modes (APO or LIG)
        is_apo_mode = mode_type == "apo"
        if is_apo_mode:
            receptor_path = input_path
            ligand_path = dirpath
            item_type = "protein"
            output_path = os.getcwd()
        else:  # LIG mode
            receptor_path = config.get('paths', {}).get('fixed_receptor_path', './LAC3_Homology_H_Min_Cut.pdb')
            ligand_path = input_path
            item_type = "ligand"
            output_path = os.getcwd()

    # Common settings for all modes
    system_settings = {
        "receptor_path": receptor_path,
        "ligand_path": ligand_path,
        "delta_pico": delta_pico,
        "rerun": config.get('system', {}).get('rerun', False),
        "gpu_id": gpu_id,
        "output_dir": output_path  # Add output directory to system settings
    }
    
    # NVT settings from config or defaults
    nvt_config = config.get('nvt', {})
    nvt_steps = nvt_config.get('steps', 400)
    nvt_settings = {
        "steps": int(nvt_steps // delta_pico),      
        "dcd_save": int(nvt_config.get('dcd_save', 50) // delta_pico),
        "log_save": int(nvt_config.get('log_save', 1) // delta_pico),
        "temps_list_simulating": nvt_config.get('temps_list', [50, 100, 150, 200, 250, 300, 301])
    }
    
    # NPT settings from config or defaults
    npt_config = config.get('npt', {})
    npt_steps = npt_config.get('steps', 400)
    npt_settings = {
        "steps": int(npt_steps // delta_pico),      
        "dcd_save": int(npt_config.get('dcd_save', 50) // delta_pico),
        "log_save": int(npt_config.get('log_save', 1) // delta_pico),
        "rests_list_decreasing": npt_config.get('rests_list', [1000000000, 100000, 1000, 100, 10, 1]),
        "atoms_to_restraints": set(npt_config.get('atoms_to_restraints', ["CA"]))
    }
    
    # MD settings from config or defaults
    md_config = config.get('md', {})
    md_steps = md_config.get('steps', 150000)
    md_settings = {
        "steps": int(md_steps // delta_pico),     
        "dcd_save": int(md_config.get('dcd_save', 100) // delta_pico),
        "log_save": int(md_config.get('log_save', 10) // delta_pico)
    }

    # For legacy mode, print the run name and output directory
    if is_legacy:
        print(f"Starting the MD on GPU {gpu_id} for {item_type} {input_path} in directory: {output_dir}")
    else:
        print(f"Starting the MD on GPU {gpu_id} for {item_type} {input_path}")
    
    runner = Run_MD()
    runner.equilibration_production(system_settings, nvt_settings, npt_settings, md_settings)

def main():
    # Check if configuration file is provided
    if len(sys.argv) != 2:
        print("Usage: python MPI_MD_Wrapper.py <config_file.toml>")
        sys.exit(1)
    
    # Get current working directory
    original_cwd = os.getcwd()
    
    # Load configuration from file
    config_file = sys.argv[1]
    try:
        with open(config_file, 'rb') as f:
            config = tomli.load(f)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    
    # Validate configuration
    if 'mode' not in config or 'type' not in config['mode']:
        print("Error: 'type' must be specified in the [mode] section ('apo', 'lig', or 'legacy')")
        sys.exit(1)
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get GPU ID from environment variable
    gpu_id = os.environ.get('ROCR_VISIBLE_DEVICES', "0")
    
    # Process based on mode
    mode = config['mode']['type'].lower()
    
    if mode == "apo":
        # APO mode - process protein files
        paths_config = config.get('paths', {})
        protein_folder = paths_config.get('protein_folder', './Proteins')
        protein_glob_pattern = paths_config.get('protein_glob_pattern', '*.cif')
        protein_paths = list(Path(protein_folder).glob(protein_glob_pattern))
        
        if not protein_paths:
            print(f"Error: No protein files found matching pattern '{protein_glob_pattern}' in {protein_folder}")
            sys.exit(1)
        
        # Assert that there are at least as many proteins as MPI ranks
        assert len(protein_paths) >= size, f"Error: Not enough proteins ({len(protein_paths)}) for all MPI ranks ({size}). Each rank must have a protein to process."
        
        protein_path = protein_paths[rank]
        protein_folder = protein_path.stem
        
        print(f"Protein {protein_path} assigned to GPU {gpu_id}\n\n\n\n", flush=True)
        md_wrapper(config, str(protein_path), dirpath=str(protein_folder), gpu_id=gpu_id)
    
    elif mode == "lig":
        # LIG mode - process ligand files
        paths_config = config.get('paths', {})
        ligand_folder = paths_config.get('ligand_folder', './Ligands_To_MD_V2')
        ligand_glob_pattern = paths_config.get('ligand_glob_pattern', 'C*.sdf')
        ligand_paths = glob.glob(f"{ligand_folder}/{ligand_glob_pattern}")
        
        if not ligand_paths:
            print(f"Error: No ligand files found matching pattern '{ligand_glob_pattern}' in {ligand_folder}")
            sys.exit(1)
        
        # Assert that there are at least as many ligands as MPI ranks
        assert len(ligand_paths) >= size, f"Error: Not enough ligands ({len(ligand_paths)}) for all MPI ranks ({size}). Each rank must have a ligand to process."
        
        ligand_path = ligand_paths[rank]
        
        print(f"Ligand {ligand_path} assigned to GPU {gpu_id}\n\n\n\n", flush=True)
        md_wrapper(config, ligand_path, gpu_id=gpu_id)
    
    elif mode == "legacy":
        # Legacy mode - process explicitly defined protein-ligand pairs
        if 'legacy' not in config or 'runs' not in config['legacy']:
            print("Error: Legacy mode requires [legacy.runs] section in configuration")
            sys.exit(1)
            
        # Get the legacy runs dictionary
        legacy_runs = config['legacy']['runs']
        run_names = list(legacy_runs.keys())
        
        if not run_names:
            print("Error: No runs defined in [legacy.runs] section")
            sys.exit(1)
            
        # Validate each run has required fields
        invalid_runs = []
        for run_name in run_names:
            run_config = legacy_runs[run_name]
            if not isinstance(run_config, dict) or 'protein' not in run_config:
                invalid_runs.append(run_name)
        
        if invalid_runs:
            print(f"Error: The following runs are missing required 'protein' field: {', '.join(invalid_runs)}")
            sys.exit(1)
        
        # Check if enough runs for all ranks
        if len(run_names) < size:
            print(f"Error: Not enough legacy runs ({len(run_names)}) for all MPI ranks ({size}). Each rank must have a run to process.")
            sys.exit(1)
            
        # Assign run to current rank
        run_name = run_names[rank]
        run_config = legacy_runs[run_name]
        
        protein_path = run_config['protein']
        # Ligand path can be "APO" for protein-only simulations
        ligand_path = run_config.get('ligand', "APO")
        
        # Create work directory based on run name if it doesn't exist
        work_dir = Path(original_cwd) / run_name
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            
        # Make sure work_dir is an absolute path
        work_dir = work_dir.absolute()
        
        print(f"Legacy run '{run_name}' with protein {protein_path} and ligand {ligand_path} assigned to GPU {gpu_id}\n\n\n\n", flush=True)
        
        # Run MD wrapper with legacy flag, passing the output directory without changing the current directory
        md_wrapper(config, protein_path, dirpath=run_name, gpu_id=gpu_id, is_legacy=True, 
                  legacy_run_name=run_name, legacy_ligand_path=ligand_path, output_dir=str(work_dir))
    
    else:
        print(f"Error: Unknown mode '{mode}'. Must be 'apo', 'lig', or 'legacy'")
        sys.exit(1)

if __name__ == "__main__":
    main() 