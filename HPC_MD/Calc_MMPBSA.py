import concurrent
import math
import os
import random
import re
import shutil
import subprocess
from subprocess import Popen, PIPE
import sys

import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import parmed as pmd
import pytraj as pt

def mmpbsa_csv_to_dict(mmpbsa_csv: Path) -> dict:

    # Read the CSV file
    with open(mmpbsa_csv, 'r') as f:
        text = f.read()
    
    # Split the text into sections using a section header followed by a colon and a newline
    section_pattern = r'(?P<section_title>[A-Z ]+):\n'
    sections = re.split(section_pattern, text)

    # Create a dictionary to store the parsed data
    parsed_data = {}

    # Iterate over sections
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip()
        section_content = sections[i + 1].strip()

        # Split section content into subsections
        subsection_pattern = r'(?P<subsection_title>[A-Za-z ]+ Terms)\n'
        subsections = re.split(subsection_pattern, section_content)

        # Initialize the section in the dictionary
        parsed_data[section_title] = {}

        # Iterate over subsections
        for j in range(1, len(subsections), 2):
            subsection_title = subsections[j].strip()
            csv_content = subsections[j + 1].strip()

            # Convert CSV content to a DataFrame
            csv_df = pd.read_csv(StringIO(csv_content))

            # Store the DataFrame in the dictionary
            parsed_data[section_title][subsection_title] = csv_df

    return parsed_data

class Calc_MMPBSA:
    def __init__(self, MMPBSA_dir, system_prmtop, traj_dcd, percentage: int):
        self.run_dir = MMPBSA_dir
        self.system_prmtop = os.path.abspath(system_prmtop)
        self.system_dcd = os.path.abspath(traj_dcd)

        self.percentage = percentage

    @staticmethod
    def create_topology_files(prmtop, complex_prmtop=None, receptor_prmtop=None, ligand_prmtop=None, strip_mask=None,
                              receptor_mask=None, ligand_mask=None, radius_set=None):
        try:
            from MMPBSA_mods.findprogs import which
        except ImportError:
            import os
            amberhome = os.getenv('AMBERHOME') or '$AMBERHOME'
            raise ImportError('Could not import Amber Python modules. Please make sure '
                              'you have sourced %s/amber.sh (if you are using sh/ksh/'
                              'bash/zsh) or %s/amber.csh (if you are using csh/tcsh)' %
                              (amberhome, amberhome))

        # Check for illegal options
        if not prmtop:
            raise ValueError('The prmtop file must be specified.')

        if receptor_mask and ligand_mask:
            raise ValueError('Cannot specify both receptor and ligand masks!')

        if (receptor_mask or ligand_mask) and not (receptor_prmtop and ligand_prmtop):
            raise ValueError('You specified ligand or receptor mask, but '
                             'not a ligand and receptor topology file!')

        if not (ligand_mask or receptor_mask) and (receptor_prmtop or ligand_prmtop):
            raise ValueError('You must provide a ligand or receptor mask '
                             'for a ligand and receptor topology file!')

        if (receptor_prmtop and not ligand_prmtop) or (ligand_prmtop and not receptor_prmtop):
            raise ValueError('You must specify both ligand and receptor topologies '
                             'or neither!')

        if not ligand_mask and not receptor_mask and not strip_mask:
            raise ValueError('You did not specify any masks -- I have nothing to do!')

        if radius_set is not None:
            allowed_radii = ('bondi', 'mbondi', 'mbondi2', 'mbondi3', 'amber6')
            if not radius_set.lower() in allowed_radii:
                raise ValueError('Radius set must be one of %s' % ' '.join(allowed_radii))

        # Now load the unspecified mask (just !(specified_mask))
        if receptor_mask or ligand_mask:
            if not receptor_mask: receptor_mask = '!(%s)' % ligand_mask
            if not ligand_mask: ligand_mask = '!(%s)' % receptor_mask

        parmed = which('parmed', search_path=True)

        # check that necessary parameters are met
        if not parmed:
            raise RuntimeError('Error: parmed cannot be found!')

        # Create the stripped topology file
        if strip_mask:
            print('Stripping %s (solvent) from original topology, output is %s' % (
                strip_mask, complex_prmtop))
            parmed_commands = "strip %s nobox\n" % strip_mask
            if radius_set is not None:
                parmed_commands += "changeRadii %s\n" % radius_set.lower()
            parmed_commands += "parmout %s\n" % complex_prmtop
            parmed_commands += "go\n"
            process = Popen([parmed, '-n', prmtop], stdout=PIPE, stderr=PIPE,
                            stdin=PIPE)
            (output, error) = process.communicate(parmed_commands.encode())
            if process.wait():
                raise RuntimeError('Error: Creating complex topology failed!\n%s\n%s' %
                                   (output.decode().strip(), error.decode().strip()))

            print('Done stripping solvent!\n')
        # If we aren't stripping solvent, our complex is our original prmtop
        else:
            complex_prmtop = prmtop

        # Now create the receptor prmtop
        if receptor_prmtop:
            print('Creating receptor topology file by stripping %s from %s' % (
                ligand_mask, complex_prmtop))

            parmed_commands = "strip %s nobox\n" % ligand_mask
            if radius_set is not None:
                parmed_commands += "changeRadii %s\n" % radius_set.lower()
            parmed_commands += "parmout %s\n" % receptor_prmtop
            parmed_commands += "go\n"
            process = Popen([parmed, '-n', complex_prmtop], stdout=PIPE, stderr=PIPE,
                            stdin=PIPE)
            (output, error) = process.communicate(parmed_commands.encode())
            if process.wait():
                raise RuntimeError('Error: Creating receptor topology failed!\n%s\n%s' %
                                   (output.decode().strip(), error.decode().strip()))

            print('Done creating receptor topology file!\n')

        # Now create the ligand prmtop
        if ligand_prmtop:
            print('Creating ligand topology file by stripping %s from %s' % (
                receptor_mask, complex_prmtop))

            parmed_commands = "strip %s nobox\n" % receptor_mask
            if radius_set is not None:
                parmed_commands += "changeRadii %s\n" % radius_set.lower()
            parmed_commands += "parmout %s\n" % ligand_prmtop
            parmed_commands += "go\n"
            process = Popen([parmed, '-n', complex_prmtop], stdout=PIPE, stderr=PIPE,
                            stdin=PIPE)
            (output, error) = process.communicate(parmed_commands.encode())

            if process.wait():
                raise RuntimeError('Error: Creating ligand topology failed!\n%s\n%s' %
                                   (output.decode().strip(), error.decode().strip()))

            print('Done creating ligand topology file!\n')

    def _frame_checker(self):
        traj = pt.iterload(self.system_dcd, self.system_prmtop)
        n_frames = len(traj)
        percentage = self.percentage
        step_size = math.ceil(1 / percentage)

        startframe, interval, endframe = (5, step_size, n_frames)
        return startframe, interval, endframe

    def _prepare_MMPBSA(self, solvent_mask, ligand_mask):
        receptor_dest = os.path.join(self.run_dir, 'receptor.prmtop')
        ligand_dest = os.path.join(self.run_dir, 'ligand.prmtop')
        complex_dest = os.path.join(self.run_dir, 'complex.prmtop')

        self.create_topology_files(self.system_prmtop, complex_prmtop=complex_dest, receptor_prmtop=receptor_dest,
                                   ligand_prmtop=ligand_dest, strip_mask=solvent_mask, receptor_mask=ligand_mask)

    def _write_mmpbsa_in(self):

        startframe, interval, endframe = self._frame_checker()

        mmpbsa_in = f"""Input file for running PB and GB in serial
&general
   startframe={startframe}, endframe={endframe}, interval={interval}, keep_files=2,
   strip_mask=":HOH,NA,CL",
/
&gb
  igb=2, saltcon=0.100,
/
&pb
  istrng=0.15, fillratio=4.0, radiopt=0
/"""
        with open(os.path.join(self.run_dir, 'mmpbsa.in'), 'w') as f:
            f.write(mmpbsa_in)

    def run_MMPBSA(self):
        # Check for existence of the directory and since the preparator does not overwrite files,
        # we can safely remove the directory
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)

        os.makedirs(self.run_dir, exist_ok=True)

        # Run the preparator
        self._prepare_MMPBSA(solvent_mask=':HOH,NA,CL', ligand_mask=':UNK')
        self._write_mmpbsa_in()

        # write the runner bash script
        runner_path = os.path.join(self.run_dir, 'run_mmpbsa.sh')
        with open(runner_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"MMPBSA.py -O -i ./mmpbsa.in -sp ../system.prmtop -rp receptor.prmtop -lp ligand.prmtop -cp complex.prmtop -y ../Step3_Md_Rep0.dcd -eo MMPBSA.csv")

        subprocess.run(f"bash run_mmpbsa.sh", cwd=self.run_dir, shell=True, check=True)

class Wrapper_MMPBSA:
    """
    A class used to wrap MMPBSA calculations.

    ...

    Attributes
    ----------
    posix_dict : Dict
        A dictionary containing the paths to the PRMTOP and DCD files for each ligand.
        Compatible with analyzer_dict from Analysis_Lig.py
        Example:
        {
            'C4_fake': {
                'PRMTOP_WAT': '/home/hitesit/Python_Packages/Holo_MD/examples/C4_fake/system.prmtop',
                'DCD_WAT': '/home/hitesit/Python_Packages/Holo_MD/examples/C4_fake/Step3_Md_Rep0.dcd'
            },
            'C4_prep': {
                'PRMTOP_WAT': '/home/hitesit/Python_Packages/Holo_MD/examples/C4_prep/system.prmtop',
                'DCD_WAT': '/home/hitesit/Python_Packages/Holo_MD/examples/C4_prep/Step3_Md_Rep0.dcd'
            }
        }

    Methods
    -------
    set_MMPBSA(posix_key: str)
        Sets up the MMPBSA calculation for the given ligand.
    run_MMPBSA()
        Runs the MMPBSA calculations for all ligands in posix_dict.
    """

    def __init__(self, posix_dict: Dict):
        """
        Constructs all the necessary attributes for the Wrapper_MMPBSA object.

        Parameters
        ----------
            posix_dict : Dict
                A dictionary containing the paths to the PRMTOP and DCD files for each ligand.
        """
        self.posix_dict = posix_dict
    
    @staticmethod
    def rationalize_MMPBSA(mmpbsa_dict: Dict) -> pd.DataFrame:
        rows = []
        for ligname, subdict in mmpbsa_dict.items():
            gb = subdict["GENERALIZED BORN"]["DELTA Energy Terms"].mean().to_frame(ligname).T["DELTA TOTAL"].values[0]
            pb = subdict["POISSON BOLTZMANN"]["DELTA Energy Terms"].mean().to_frame(ligname).T["DELTA TOTAL"].values[0]

            row = {
                "Ligand": ligname,
                "GB": gb,
                "PB": pb
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df
    
    def set_MMPBSA(self, posix_key: str):
        # Create MMPBSA directory relative to the file paths, not the key name
        base_dir = Path(self.posix_dict[posix_key]["PRMTOP_WAT"]).parent
        mmpbsa_dir = base_dir / "MMPBSA"
        mmpbsa_dir.mkdir(exist_ok=True)

        prmtop = self.posix_dict[posix_key]["PRMTOP_WAT"]
        dcd = self.posix_dict[posix_key]["DCD_WAT"]

        mmpbsa_calc = Calc_MMPBSA(str(mmpbsa_dir), prmtop, dcd, percentage=0.1)
        mmpbsa_calc.run_MMPBSA()

        mmpbsa_csv_result: Path = mmpbsa_dir / "MMPBSA.csv"
        assert mmpbsa_csv_result.exists()

        # Return both the path and the key (ligand name) for better identification
        return (mmpbsa_csv_result, posix_key)
    
    def run_MMPBSA(self):
        ligands_to_MMPBS = list(self.posix_dict.keys())
        mmpbsa_results: List[Tuple[Path, str]] = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.set_MMPBSA, ligand) for ligand in ligands_to_MMPBS]
            for future in as_completed(futures):
                try:
                    mmpbsa_csv_path, ligname = future.result()
                    mmpbsa_results.append((mmpbsa_csv_path, ligname))
                except concurrent.futures.process.BrokenProcessPool as e:
                    print(f"BrokenProcessPool error occurred: {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")
                else:
                    print(f"Completed successfully")
        
        return mmpbsa_results
    
    def parse_MMPBSA(self, mmpbsa_results: List[Tuple[Path, str]]) -> Dict:
        mmpbsa_dict = {}
        for mmpbsa_csv_path, ligname in mmpbsa_results:
            # Use the provided ligand name instead of parsing the path
            mmpbsa_dict[ligname] = mmpbsa_csv_to_dict(mmpbsa_csv_path)
        
        return mmpbsa_dict
    
    def __call__(self) -> pd.DataFrame:
        mmpbsa_results: List[Tuple[Path, str]] = self.run_MMPBSA()
        mmpbsa_dict: Dict = self.parse_MMPBSA(mmpbsa_results)
        mmpbsa_df: pd.DataFrame = self.rationalize_MMPBSA(mmpbsa_dict)
        return mmpbsa_df