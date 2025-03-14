#!/usr/bin/env python
# Example of using Wrapper_MMPBSA with analyzer_dict from Analysis_Lig

import os
from pathlib import Path
import pandas as pd
from typing import Dict

from HPC_MD.Calc_MMPBSA import Wrapper_MMPBSA
from HPC_MD.Analysis_Lig import find_matching_directories, create_analyzer_dict

def run_mmpbsa_analysis():
    """
    Example of using Wrapper_MMPBSA with analyzer_dict from Analysis_Lig
    """
    # 1. Find directories and create analyzer_dict
    # This would normally be done via the Analysis_Lig module
    md_output_dirs = ["ligand1", "ligand2"]  # Replace with your actual directory patterns
    file_paths = find_matching_directories(md_output_dirs)
    analyzer_dict = create_analyzer_dict(file_paths, overwrite=False)
    
    # 2. Create a simplified dict for Wrapper_MMPBSA
    # The Wrapper_MMPBSA only needs PRMTOP_WAT and DCD_WAT
    mmpbsa_input_dict = {}
    for dirname, data in analyzer_dict.items():
        mmpbsa_input_dict[dirname] = {
            "PRMTOP_WAT": data["PRMTOP_WAT"],
            "DCD_WAT": data["DCD_WAT"]
        }
    
    # 3. Run MMPBSA calculations
    mmpbsa_wrapper = Wrapper_MMPBSA(mmpbsa_input_dict)
    mmpbsa_results_df = mmpbsa_wrapper()
    
    # 4. Print and save results
    print("MMPBSA Results:")
    print(mmpbsa_results_df)
    
    # 5. Save to CSV
    mmpbsa_results_df.to_csv("mmpbsa_results.csv", index=False)
    
    return mmpbsa_results_df

if __name__ == "__main__":
    run_mmpbsa_analysis() 