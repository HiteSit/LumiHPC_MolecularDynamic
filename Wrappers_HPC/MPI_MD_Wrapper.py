import os
import glob
import sys
import tomli
import subprocess
from pathlib import Path
from mpi4py import MPI
import parmed as pmd
import pytraj as pt
from HPC_MD.Vanilla_MD import Run_MD

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
            # Use dirpath for output_path in APO mode
            output_path = os.path.join(os.getcwd(), dirpath) if dirpath else os.getcwd()
        else:  # LIG mode
            receptor_path = config.get('paths', {}).get('fixed_receptor_path', './LAC3_Homology_H_Min_Cut.pdb')
            ligand_path = input_path
            item_type = "ligand"
            # Use dirpath for output_path in LIG mode
            output_path = os.path.join(os.getcwd(), dirpath) if dirpath else os.getcwd()

    # Common settings for all modes
    system_settings = {
        "receptor_path": receptor_path,
        "ligand_path": ligand_path,
        "delta_pico": delta_pico,
        "rerun": config.get('system', {}).get('rerun', False),
        "gpu_id": str(gpu_id),
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
        "log_save": int(md_config.get('log_save', 10) // delta_pico),
    }

    # For legacy mode, print the run name and output directory
    if is_legacy:
        print(f"Starting the MD on GPU {gpu_id} for {item_type} {input_path} in directory: {output_dir}")
    else:
        print(f"Starting the MD on GPU {gpu_id} for {item_type} {input_path}")
    
    runner = Run_MD()
    runner.equilibration_production(system_settings, nvt_settings, npt_settings, md_settings)

def process_task_item(config, item_path, item_type, gpu_id, comm, rank, item_name=None, is_legacy=False, legacy_ligand_path=None, output_dir=None, dirpath=None):
    """
    Process a single task item (protein, ligand, or legacy run)
    
    Args:
        config: Configuration dictionary
        item_path: Path to the item to process
        item_type: Type of item ('protein', 'ligand', or 'legacy')
        gpu_id: GPU ID to use
        comm: MPI communicator
        rank: Current rank ID
        item_name: Name of the item (used for logging)
        is_legacy: Flag indicating legacy mode
        legacy_ligand_path: Path to ligand for legacy mode
        output_dir: Output directory for results
        dirpath: Directory path (used for proteins in APO mode)
    
    Returns:
        Tuple of (status, message)
    """
    try:
        if item_type == 'protein':
            # For protein mode (APO)
            protein_folder = Path(item_path).stem if dirpath is None else dirpath
            print(f"Worker {rank}: Processing protein {item_path} on GPU {gpu_id}", flush=True)
            md_wrapper(config, str(item_path), dirpath=str(protein_folder), gpu_id=gpu_id)
        elif item_type == 'ligand':
            # For ligand mode (LIG)
            # Create a directory name based on the ligand file name
            ligand_folder = Path(item_path).stem
            print(f"Worker {rank}: Processing ligand {item_path} on GPU {gpu_id}", flush=True)
            md_wrapper(config, item_path, dirpath=str(ligand_folder), gpu_id=gpu_id)
        elif item_type == 'legacy':
            # For legacy mode (protein-ligand pair)
            print(f"Worker {rank}: Processing legacy run '{item_name}' with protein {item_path} and ligand {legacy_ligand_path} on GPU {gpu_id}", flush=True)
            md_wrapper(config, item_path, dirpath=item_name, gpu_id=gpu_id, is_legacy=True, 
                      legacy_run_name=item_name, legacy_ligand_path=legacy_ligand_path, output_dir=output_dir)
        return ("SUCCESS", f"Successfully processed {item_type} {item_path}")
    except Exception as e:
        error_msg = f"ERROR processing {item_type} {item_path}: {str(e)}"
        print(f"Worker {rank}: {error_msg}")
        return ("ERROR", error_msg)

def master_worker_distribution(comm, rank, size, work_items, process_func, item_type, config, gpu_id, original_cwd=None):
    """
    Implements master-worker dynamic load balancing
    
    Args:
        comm: MPI communicator
        rank: Current rank ID
        size: Total number of MPI ranks
        work_items: List of items to be processed
        process_func: Function to process each item
        item_type: Type of items being processed ('protein', 'ligand', or 'legacy')
        config: Configuration dictionary
        gpu_id: GPU ID to use
        original_cwd: Original working directory
    """
    if rank == 0:
        # Master process
        pending_items = list(work_items)  # Queue of items to process
        completed_items = []              # List of completed items
        active_workers = {}               # Maps worker rank to assigned item
        
        print(f"Master starting distribution of {len(pending_items)} {item_type} items to {size-1} workers")
        
        # Initial distribution of tasks
        for worker_rank in range(1, size):
            if pending_items:
                item_to_assign = pending_items.pop(0)
                comm.send(item_to_assign, dest=worker_rank, tag=1)  # Tag 1: New task
                active_workers[worker_rank] = item_to_assign
                print(f"Master: Assigned initial {item_type} '{item_to_assign}' to worker {worker_rank}")
            else:
                comm.send("DONE", dest=worker_rank, tag=1)  # No more work
        
        # Continue distributing work as workers finish their tasks
        while active_workers:
            # Wait for any worker to report
            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)  # Tag 2: Task complete
            worker_rank = status.Get_source()
            
            if worker_rank in active_workers:
                completed_items.append(active_workers[worker_rank])
                print(f"Master: Worker {worker_rank} completed {item_type} '{active_workers[worker_rank]}'")
                del active_workers[worker_rank]
            
            # Assign new work to the worker if available
            if pending_items:
                item_to_assign = pending_items.pop(0)
                comm.send(item_to_assign, dest=worker_rank, tag=1)
                active_workers[worker_rank] = item_to_assign
                print(f"Master: Assigned {item_type} '{item_to_assign}' to worker {worker_rank}")
            else:
                comm.send("DONE", dest=worker_rank, tag=1)
                print(f"Master: No more work for worker {worker_rank}, sending DONE signal")
        
        print(f"Master: All {len(completed_items)} {item_type} items have been processed")
        return completed_items
    else:
        # Worker process
        while True:
            # Request work from master
            item = comm.recv(source=0, tag=1)
            
            # Check if there's more work to do
            if item == "DONE":
                print(f"Worker {rank}: Received DONE signal, exiting")
                break
            
            # Process the assigned item based on the mode
            if item_type == 'protein':
                # Process protein
                protein_path = item
                protein_folder = Path(protein_path).stem
                result = process_func(config, protein_path, 'protein', gpu_id, comm, rank, dirpath=protein_folder)
            elif item_type == 'ligand':
                # Process ligand
                ligand_path = item
                # Create ligand folder from basename
                ligand_folder = Path(ligand_path).stem
                result = process_func(config, ligand_path, 'ligand', gpu_id, comm, rank, dirpath=ligand_folder)
            elif item_type == 'legacy':
                # Process legacy run
                run_name = item
                run_config = config['legacy']['runs'][run_name]
                protein_path = run_config['protein']
                ligand_path = run_config.get('ligand', "APO")
                work_dir = Path(original_cwd) / run_name
                work_dir = work_dir.absolute()
                result = process_func(config, protein_path, 'legacy', gpu_id, comm, rank, 
                                    item_name=run_name, is_legacy=True, 
                                    legacy_ligand_path=ligand_path, output_dir=str(work_dir))
            
            # Report completion to master
            comm.send(f"{item}:{result[0]}", dest=0, tag=2)

def main():
    # Initialize MPI first
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Get current working directory
    original_cwd = os.getcwd()
    
    # Root rank will read and broadcast configuration
    if rank == 0:
        # Check if configuration file is provided
        if len(sys.argv) != 2:
            print("Usage: python MPI_MD_Wrapper.py <config_file.toml>")
            sys.exit(1)
        
        # Load configuration from file
        config_file = sys.argv[1]
        try:
            with open(config_file, 'rb') as f:
                config = tomli.load(f)
                config_valid = True
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            config_valid = False
            config = None
            
        # Validate configuration
        if config_valid and ('mode' not in config or 'type' not in config['mode']):
            print("Error: 'type' must be specified in the [mode] section ('apo', 'lig', or 'legacy')")
            config_valid = False
    else:
        config = None
        config_valid = None
    
    # Broadcast the config validity status
    config_valid = comm.bcast(config_valid, root=0)
    
    # Exit if config is invalid
    if not config_valid:
        if rank == 0:
            print("Configuration error - exiting all MPI processes")
        sys.exit(1)
        
    # Broadcast the configuration to all ranks
    config = comm.bcast(config, root=0)
    
    # Get GPU ID from environment variable
    gpu_id = os.environ.get('ROCR_VISIBLE_DEVICES')
    
    # Process based on mode
    mode = config['mode']['type'].lower()
    
    # Synchronize all processes before file gathering
    comm.Barrier()
    
    if mode == "apo":
        # APO mode - process protein files
        paths_config = config.get('paths', {})
        protein_folder = paths_config.get('protein_folder', './Proteins')
        protein_glob_pattern = paths_config.get('protein_glob_pattern', '*.cif')
        
        # Only root process collects all files to ensure consistency
        if rank == 0:
            protein_paths = list(Path(protein_folder).glob(protein_glob_pattern))
            if not protein_paths:
                print(f"Error: No protein files found matching pattern '{protein_glob_pattern}' in {protein_folder}")
                protein_paths = []
        else:
            protein_paths = None
        
        # Broadcast the gathered protein paths
        protein_paths = comm.bcast(protein_paths, root=0)
        
        # Exit if no files found
        if not protein_paths:
            if rank == 0:
                print("No protein files found - exiting all MPI processes")
            sys.exit(1)
            
        # Convert Path objects to strings for serialization
        protein_paths_str = [str(p) for p in protein_paths]
            
        # Use dynamic load balancing for proteins
        completed = master_worker_distribution(
            comm, rank, size, protein_paths_str, process_task_item, 
            'protein', config, gpu_id, original_cwd
        )
    
    elif mode == "lig":
        # LIG mode - process ligand files
        paths_config = config.get('paths', {})
        ligand_folder = paths_config.get('ligand_folder', './Ligands_To_MD_V2')
        ligand_glob_pattern = paths_config.get('ligand_glob_pattern', 'C*.sdf')
        
        # Only root process collects all files to ensure consistency
        if rank == 0:
            ligand_paths = glob.glob(f"{ligand_folder}/{ligand_glob_pattern}")
            if not ligand_paths:
                print(f"Error: No ligand files found matching pattern '{ligand_glob_pattern}' in {ligand_folder}")
                ligand_paths = []
        else:
            ligand_paths = None
        
        # Broadcast the gathered ligand paths
        ligand_paths = comm.bcast(ligand_paths, root=0)
        
        # Exit if no files found
        if not ligand_paths:
            if rank == 0:
                print("No ligand files found - exiting all MPI processes")
            sys.exit(1)
            
        # Use dynamic load balancing for ligands
        completed = master_worker_distribution(
            comm, rank, size, ligand_paths, process_task_item, 
            'ligand', config, gpu_id, original_cwd
        )
    
    elif mode == "legacy":
        # Legacy mode - process explicitly defined protein-ligand pairs
        if 'legacy' not in config or 'runs' not in config['legacy']:
            if rank == 0:
                print("Error: Legacy mode requires [legacy.runs] section in configuration")
            sys.exit(1)
            
        # Get the legacy runs dictionary
        legacy_runs = config['legacy']['runs']
        run_names = list(legacy_runs.keys())
        
        # Only root process validates the runs
        if rank == 0:
            if not run_names:
                print("Error: No runs defined in [legacy.runs] section")
                valid_runs = False
            else:
                # Validate each run has required fields
                invalid_runs = []
                for run_name in run_names:
                    run_config = legacy_runs[run_name]
                    if not isinstance(run_config, dict) or 'protein' not in run_config:
                        invalid_runs.append(run_name)
                
                if invalid_runs:
                    print(f"Error: The following runs are missing required 'protein' field: {', '.join(invalid_runs)}")
                    valid_runs = False
                else:
                    valid_runs = True
        else:
            valid_runs = None
            
        # Broadcast validation result
        valid_runs = comm.bcast(valid_runs, root=0)
        
        if not valid_runs:
            if rank == 0:
                print("Legacy runs configuration error - exiting all MPI processes")
            sys.exit(1)
            
        # Broadcast the run names
        run_names = comm.bcast(run_names, root=0)
        
        # Create all work directories first (only on rank 0)
        if rank == 0:
            print(f"Found {len(run_names)} legacy runs to process with {size} MPI ranks")
            # Create directories for all runs
            for run_name in run_names:
                work_dir = Path(original_cwd) / run_name
                if not work_dir.exists():
                    work_dir.mkdir(parents=True, exist_ok=True)
        
        # Wait for all directories to be created
        comm.Barrier()
        
        # Use dynamic load balancing for legacy runs
        completed = master_worker_distribution(
            comm, rank, size, run_names, process_task_item, 
            'legacy', config, gpu_id, original_cwd
        )
    
    else:
        if rank == 0:
            print(f"Error: Unknown mode '{mode}'. Must be 'apo', 'lig', or 'legacy'")
        sys.exit(1)
    
    # Final message only from rank 0
    if rank == 0:
        print("All processing completed successfully")

if __name__ == "__main__":
    main() 