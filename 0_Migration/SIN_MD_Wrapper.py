import os
import sys
import shutil
import subprocess

import parmed as pmd
import pytraj as pt

from src.Vanilla_MD import Run_MD

def md_wrapper(protein_path, ligand_path, gpu_id:str):

    delta_pico = 0.002

    system_settings = {
        "receptor_path": protein_path,
        "ligand_path": ligand_path,
        "delta_pico": delta_pico,
        "rerun": False,
        "gpu_id": str(gpu_id)
    }
    
    nvt_settings = {
        "steps": int(400 // delta_pico),      
        "dcd_save":int(50 // delta_pico),
        "log_save":int(1 // delta_pico),
        "temps_list_simulating":[50, 100, 150, 200, 250, 300, 301]
    }
    
    npt_settings = {
        "steps": int(400 // delta_pico),      
        "dcd_save": int(50 // delta_pico),
        "log_save": int(1 // delta_pico),
        "rests_list_decreasing":[1000000000, 100000, 1000, 100, 10, 1],
        "atoms_to_restraints":{"CA"}
    }
    
    md_settings = {
        "steps": int(150000 // delta_pico),     
        "dcd_save":int(100 // delta_pico),
        "log_save":int(10 // delta_pico)
    }

    print("Starting the MD")
    runner = Run_MD()
    runner.equilibration_production(system_settings, nvt_settings, npt_settings, md_settings)

if __name__ == "__main__":
    protein_path = sys.argv[1]
    ligand_path = sys.argv[2]
    gpu_id = sys.argv[3]
    md_wrapper(protein_path, ligand_path, gpu_id)

