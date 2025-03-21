#!/usr/bin/env python
import os
import sys
import time
from pathlib import Path
from mpi4py import MPI
from tqdm import tqdm

from HPC_MD.Analysis_Lig import MD_Analyzer, find_matching_directories

def process_directory(dirname, files, overwrite=False):
    """
    Process a single directory to create an MD_Analyzer and required files.
    
    Parameters:
    -----------
    dirname : str
        Name of the directory to process
    files : dict
        Dictionary with file paths
    overwrite : bool, optional
        Whether to overwrite existing processed files
        
    Returns:
    --------
    tuple
        (status, message) indicating the result of processing
    """
    try:
        start_time = time.time()
        
        # Create analyzer (this triggers file creation even if we don't use the object)
        analyzer = MD_Analyzer(dirname, files["DCD_WAT"], files["PRMTOP_WAT"], overwrite=overwrite)
        
        # Extract paths
        dcd_wat = Path(files["DCD_WAT"])
        prmtop_wat = Path(files["PRMTOP_WAT"])
        
        # Generate related paths
        prmtop_nowat = Path(str(prmtop_wat).replace("system.prmtop", "system_noWAT.prmtop"))
        xtc_nowat = Path(str(dcd_wat).replace(".dcd", "_noWAT.xtc"))
        dcd_nowat = Path(str(dcd_wat).replace(".dcd", "_noWAT.dcd"))
        pdb_file = Path(dirname) / "Minimized_noWAT.pdb"
        cluster_file = Path(dirname) / "Clusters.pdb"
        
        # Verify files exist
        assert dcd_wat.exists(), f"DCD_WAT file not found: {dcd_wat}"
        assert prmtop_wat.exists(), f"PRMTOP_WAT file not found: {prmtop_wat}"
        assert prmtop_nowat.exists(), f"PRMTOP_noWAT file not found: {prmtop_nowat}"
        assert xtc_nowat.exists(), f"XTC_noWAT file not found: {xtc_nowat}"
        assert dcd_nowat.exists(), f"DCD_noWAT file not found: {dcd_nowat}"
        assert pdb_file.exists(), f"PDB_noWAT file not found: {pdb_file}"
        assert cluster_file.exists(), f"CLUSTER file not found: {cluster_file}"
        
        elapsed_time = time.time() - start_time
        return ("SUCCESS", f"Successfully processed {dirname} in {elapsed_time:.2f}s")
        
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        return ("ERROR", f"Error processing {dirname}: {str(e)}\n{tb_str}")

def master_worker_distribution(comm, rank, size, work_items, overwrite=False):
    """
    Implements master-worker dynamic load balancing for processing directories
    
    Parameters:
    -----------
    comm : MPI.Comm
        MPI communicator
    rank : int
        Current rank ID
    size : int
        Total number of MPI ranks
    work_items : dict
        Dictionary with directory names as keys and subdictionaries as values
    overwrite : bool, optional
        Whether to overwrite existing processed files
        
    Returns:
    --------
    tuple
        (completed_items, results) - Lists of completed items and their results
        (only meaningful for rank 0)
    """
    if rank == 0:
        # Master process
        pending_items = list(work_items.keys())  # Queue of directory names
        total_items = len(pending_items)
        completed_items = []                    # List of completed directory names
        active_workers = {}                     # Maps worker rank to assigned directory name
        results = {}                           # Store results from each directory
        
        print(f"Master: Starting distribution of {len(pending_items)} directories to {size-1} workers...")
        
        # Set up progress bar
        pbar = tqdm(total=total_items, desc="Processing directories")
        
        # Initial distribution of tasks
        for worker_rank in range(1, size):
            if pending_items:
                dirname = pending_items.pop(0)
                # Send both the directory name and its file paths
                item_data = (dirname, work_items[dirname])
                comm.send(item_data, dest=worker_rank, tag=1)  # Tag 1: New task
                active_workers[worker_rank] = dirname
            else:
                comm.send(("DONE", None), dest=worker_rank, tag=1)  # No more work
        
        # Continue distributing work as workers finish their tasks
        while active_workers:
            # Wait for any worker to report
            status = MPI.Status()
            result = comm.recv(source=MPI.ANY_SOURCE, tag=2, status=status)  # Tag 2: Task complete
            worker_rank = status.Get_source()
            
            if worker_rank in active_workers:
                dirname = active_workers[worker_rank]
                completed_items.append(dirname)
                results[dirname] = result
                pbar.update(1)
                
                # Print status of completed work
                status_msg, details = result
                if status_msg == "SUCCESS":
                    tqdm.write(f"Worker {worker_rank}: {details}")
                else:
                    tqdm.write(f"Worker {worker_rank}: FAILED - {details}")
                
                del active_workers[worker_rank]
            
            # Assign new work to the worker if available
            if pending_items:
                dirname = pending_items.pop(0)
                item_data = (dirname, work_items[dirname])
                comm.send(item_data, dest=worker_rank, tag=1)
                active_workers[worker_rank] = dirname
            else:
                comm.send(("DONE", None), dest=worker_rank, tag=1)
        
        pbar.close()
        print(f"Master: All {len(completed_items)} directories have been processed")
        
        # Count successful and failed tasks
        success_count = sum(1 for r in results.values() if r[0] == "SUCCESS")
        fail_count = len(results) - success_count
        print(f"Summary: {success_count} successful, {fail_count} failed")
        
        # Return results for potential further processing
        return completed_items, results
    else:
        # Worker process
        processed_count = 0
        
        while True:
            # Request work from master
            item_data = comm.recv(source=0, tag=1)
            dirname, files_data = item_data
            
            # Check if there's more work to do
            if dirname == "DONE":
                print(f"Worker {rank}: Completed {processed_count} tasks, now exiting")
                break
            
            # Process the assigned directory
            result = process_directory(dirname, files_data, overwrite)
            processed_count += 1
            
            # Report completion to master
            comm.send(result, dest=0, tag=2)
        
        return [], {}  # Empty results for worker ranks

def mpi_process_directories(md_output_dir, overwrite=False):
    """
    Find directories matching patterns and process them in parallel using MPI.
    This function focuses solely on file generation and does not return a data structure.
    
    Parameters:
    -----------
    md_output_dir : list
        List of directory names or regex patterns to match
    overwrite : bool, optional
        Whether to overwrite existing processed files
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Check that we have at least one worker
    if size < 2 and rank == 0:
        print("Warning: Running with only one MPI rank. No parallelization will occur.")
    
    # Only the master rank (0) finds directories
    if rank == 0:
        start_time = time.time()
        print(f"Finding directories matching patterns: {md_output_dir}")
        file_paths = find_matching_directories(md_output_dir)
        
        if not file_paths:
            print("Error: No matching directories found!")
            # Broadcast empty result to workers
            for worker_rank in range(1, size):
                comm.send(("DONE", None), dest=worker_rank, tag=1)
            return
        
        print(f"Found {len(file_paths)} directories to process")
        prep_time = time.time() - start_time
        print(f"Time to find directories: {prep_time:.2f}s")
    else:
        file_paths = None
    
    # Synchronize all processes before starting work
    comm.Barrier()
    
    # Perform the processing using master-worker distribution
    if rank == 0:
        print(f"Starting parallel processing with {size} MPI ranks...")
        start_time = time.time()
    
    completed_items, results = master_worker_distribution(comm, rank, size, file_paths, overwrite)
    
    # Only rank 0 reports final stats
    if rank == 0:
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s")
        
        # Report which directories were processed successfully
        success_dirs = [d for d in completed_items if results.get(d, ("ERROR", ""))[0] == "SUCCESS"]
        print(f"Successfully processed {len(success_dirs)} directories:")
        for i, dirname in enumerate(success_dirs):
            if i < 10 or i >= len(success_dirs) - 10:
                print(f"  - {dirname}")
            elif i == 10:
                print(f"  ... ({len(success_dirs) - 20} more) ...")
        
        # Report failures if any
        failed_dirs = [d for d in completed_items if results.get(d, ("ERROR", ""))[0] != "SUCCESS"]
        if failed_dirs:
            print(f"Failed to process {len(failed_dirs)} directories:")
            for dirname in failed_dirs:
                status, message = results[dirname]
                print(f"  - {dirname}: {message.split(os.linesep)[0]}")

def main():
    """
    Main function to run MPI-based analysis
    """
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Parse command line arguments
    if rank == 0:
        import argparse
        parser = argparse.ArgumentParser(description='Run MPI-based analysis on molecular dynamics data')
        parser.add_argument('patterns', nargs='+', help='Directory patterns to match (e.g., "complex_*")')
        parser.add_argument('--overwrite', action='store_true', help='Overwrite existing processed files')
        
        # Show help message if no arguments provided
        if len(sys.argv) == 1:
            parser.print_help()
            # Send DONE signal to all workers
            for worker_rank in range(1, size):
                comm.send(("DONE", None), dest=worker_rank, tag=1)
            return
            
        args = parser.parse_args()
        md_output_dir = args.patterns
        overwrite = args.overwrite
        
        if overwrite:
            print("Overwrite mode enabled")
    else:
        md_output_dir = None
        overwrite = None
    
    # Broadcast values to all processes
    md_output_dir = comm.bcast(md_output_dir, root=0)
    overwrite = comm.bcast(overwrite, root=0)
    
    # Run MPI-based processing - does not return anything
    mpi_process_directories(md_output_dir, overwrite)
    
    # Final message only from rank 0
    if rank == 0:
        print("\nMPI processing complete. Files have been generated in parallel.")
        print("\nNext steps:")
        print("1. Use these files with your normal workflow")
        print("2. Create analyzer dictionary manually with:")
        print("   analyzer_dict = create_analyzer_dict(find_matching_directories(md_output_dir), overwrite=False)")

if __name__ == "__main__":
    """
    Usage:
    
    mpirun -n <num_processes> python Wrappers_HPC/MPI.py <dir_pattern1> [<dir_pattern2> ...] --overwrite
    
    """
    main() 