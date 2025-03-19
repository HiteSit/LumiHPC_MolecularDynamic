import os
import re
import sys
import shutil
import cloudpickle
from tqdm import tqdm
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from itertools import chain

import datamol as dm

import pytraj as pt
from pytraj.cluster import kmeans
import parmed as pmd

import MDAnalysis as mda
from MDAnalysis.analysis import rms, align, rms, gnm, pca
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

try:
    import nglview as nv
    HAS_NGLVIEW = True
except ImportError:
    HAS_NGLVIEW = False

"""
Holo_MD Analysis Helpers
========================

This module provides classes and functions for analyzing molecular dynamics simulations.

Data Structure Philosophy:
-------------------------
- Path handling uses pathlib.Path objects throughout, converted to strings when passed to external libraries
- Analyzer objects are created and stored in a nested dictionary structure:
  {
    'directory_name': {
      'PRMTOP_WAT': '/path/to/topology_with_water.prmtop',
      'DCD_WAT': '/path/to/trajectory_with_water.dcd',
      'PRMTOP_noWAT': '/path/to/topology_without_water.prmtop',
      'PDB_noWAT': '/path/to/structure_without_water.pdb',
      'XTC_noWAT': '/path/to/trajectory_without_water.xtc',
      'CLUSTER': '/path/to/clusters.pdb',
      'CLASS': <MD_Analyzer object>
    },
    ...
  }
- Plotting functions accept this dictionary and use the 'CLASS' key to access analyzers
"""

def find_matching_directories(md_output_dir):
    """
    Find directories matching patterns in md_output_dir and locate their PRMTOP and DCD files.
    
    Parameters:
    -----------
    md_output_dir : list
        List of directory names or regex patterns to match
        
    Returns:
    --------
    dict
        Dictionary with directory names as keys and nested dictionaries as values.
        Each nested dictionary initially contains:
        {
            'PRMTOP_WAT': '/path/to/topology.prmtop',
            'DCD_WAT': '/path/to/trajectory.dcd'
        }
        
        This dictionary is typically expanded later with additional file paths
        and an analyzer object stored under the 'CLASS' key.
    """
    file_paths = {}
    prmtop_filename = "system.prmtop"
    dcd_pattern = re.compile(r'.*0.dcd')
    
    for root, dirs, _ in os.walk("."):
        for dir_name in dirs:
            # Check if directory matches any pattern in md_output_dir
            for pattern in md_output_dir:
                if (pattern == dir_name) or (re.match(pattern, dir_name)):
                    dirpath = Path(root) / dir_name
                    file_paths[dir_name] = {"PRMTOP_WAT": "", "DCD_WAT": ""}
                    
                    for file in os.listdir(dirpath):
                        filepath = dirpath / file
                        if file == prmtop_filename:
                            file_paths[dir_name]["PRMTOP_WAT"] = str(filepath)
                        elif dcd_pattern.match(file):
                            file_paths[dir_name]["DCD_WAT"] = str(filepath)
                    break  # Once matched, no need to check other patterns
    
    return file_paths

def create_analyzer_dict(file_paths, overwrite=False):
    """
    Create a dictionary of analyzers and file paths from a dictionary of file paths.
    
    Parameters:
    -----------
    file_paths : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain 'DCD_WAT' and 'PRMTOP_WAT' keys.
    overwrite : bool, optional
        Whether to overwrite existing processed files. Default is False.
        
    Returns:
    --------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary contains the original file paths plus additional paths
        and an 'CLASS' key with the analyzer object.
    """
    analyzer_dict = file_paths.copy()
    for dirname, files in tqdm(file_paths.items(), desc="Creating analyzers"):
        analyzer = MD_Analyzer(dirname, files["DCD_WAT"], files["PRMTOP_WAT"], overwrite=overwrite)
        
        # DCD and PRMTOP
        dcd_wat = Path(files["DCD_WAT"])
        assert dcd_wat.exists()
        
        prmtop_wat = Path(files["PRMTOP_WAT"])
        assert prmtop_wat.exists()
        
        prmtop_nowat = Path(str(prmtop_wat).replace("system.prmtop", "system_noWAT.prmtop"))
        assert prmtop_nowat.exists() 
        
        xtc_nowat = Path(str(dcd_wat).replace(".dcd", "_noWAT.xtc"))
        assert xtc_nowat.exists()
        
        dcd_nowat = Path(str(dcd_wat).replace(".dcd", "_noWAT.dcd"))
        assert dcd_nowat.exists()
        
        dcd_wat_slice = Path(str(dcd_wat).replace(".dcd", "_WAT.dcd"))
        assert dcd_wat_slice.exists()
        
        pdb_file = Path(dirname) / "Minimized_noWAT.pdb"
        assert pdb_file.exists()
        
        cluster_file = Path(dirname) / "Clusters.pdb"
        assert cluster_file.exists()
        
        analyzer_dict[dirname]["PRMTOP_noWAT"] = str(prmtop_nowat)
        analyzer_dict[dirname]["PDB_noWAT"] = str(pdb_file)
        analyzer_dict[dirname]["XTC_noWAT"] = str(xtc_nowat)
        analyzer_dict[dirname]["DCD_noWAT"] = str(dcd_nowat)
        analyzer_dict[dirname]["DCD_WAT"] = str(dcd_wat_slice)
        
        analyzer_dict[dirname]["CLUSTER"] = str(cluster_file)
        analyzer_dict[dirname]["CLASS"] = analyzer
    
    return analyzer_dict

class Pytraj_Analysis():
    """
    Handles basic trajectory analysis using pytraj.
    
    This class processes raw trajectories to:
    1. Remove water and ions
    2. Convert to different formats (XTC, DCD, PDB)
    3. Perform clustering and basic analysis
    
    Parameters:
    -----------
    md_dir : str or Path
        Path to the directory containing the MD simulation files
    traj_path : str or Path
        Path to the trajectory file (DCD format)
    top_path : str or Path
        Path to the topology file (PRMTOP format)
    overwrite : bool, default=False
        Whether to overwrite existing processed files
        
    Attributes:
    -----------
    traj_noWAT : pytraj.Trajectory
        Trajectory with water and ions removed
    """
    def __init__(self, md_dir, traj_path, top_path, overwrite=False):
        self.md_dir = Path(md_dir)
        self.traj_path = Path(traj_path)
        self.top_path = Path(top_path)
        self.overwrite = overwrite

        # General settings
        SLICE = 20
        SLICE_MULTIPLIER = 1.6
        SLICE_DIVIDER = 4
        
        MASK_WAT = "!(:HOH,NA,CL)"
        MASK_CLUSTER = "!@H="
        
        cluster_opts = {"MASK_WAT": MASK_WAT, "MASK_CLUSTER": MASK_CLUSTER, "NUM": 10, "SLICE": SLICE / SLICE_DIVIDER}

        # Define output paths
        self.xtc_path = self.md_dir / "Step3_Md_Rep0_noWAT.xtc"
        self.dcd_path = self.md_dir / "Step3_Md_Rep0_noWAT.dcd"
        self.dcd_wat_slice = self.md_dir / "Step3_Md_Rep0_WAT.dcd"
        
        self.nowat_top_path = self.md_dir / "system_noWAT.prmtop"
        self.pdb_path = self.md_dir / "Minimized_noWAT.pdb"
        self.cluster_path = self.md_dir / "Clusters.pdb"
        
        all_files_exist = (
            self.xtc_path.exists() and 
            self.nowat_top_path.exists() and 
            self.dcd_path.exists() and
            self.dcd_wat_slice.exists() and
            self.pdb_path.exists() and
            self.cluster_path.exists()
        )

        # Handle trajectory loading and processing
        if str(traj_path).endswith(".dcd"):
            if not overwrite:
                self.traj_noWAT = pt.iterload(str(self.dcd_path), str(self.nowat_top_path))            # Load no-water DCD
            else:
                # Process DCD (heavy computation)
                self.traj_WAT = pt.iterload(str(self.traj_path), str(self.top_path), stride=SLICE)     # Load with stride
                self.traj_WAT.autoimage()                                                              # Fix box image
                self.traj_WAT.superpose(mask="@CA", ref=0)                                             # Superpose on first frame
                self.traj_noWAT = self.traj_WAT[MASK_WAT]                                              # Remove water, ions, and ligands
                
                # Write XTC and DCD
                pt.write_traj(str(self.xtc_path), self.traj_noWAT, overwrite=True)
                pt.write_traj(str(self.dcd_path), self.traj_noWAT, overwrite=True)
                
                # Write DCD with super-sliced frames
                SS = int(SLICE * SLICE_MULTIPLIER)
                TMP_TRAJ = self.traj_WAT[::SS]
                pt.write_traj(str(self.dcd_wat_slice), TMP_TRAJ, overwrite=True)
                
                # Write noWAT topology
                pt.save(str(self.nowat_top_path), self.traj_noWAT.top, overwrite=True)
                
                # Write clusters
                self.cluster_traj(cluster_opts)
        else:
            raise ValueError(f"Trajectory file {self.traj_path} is not a DCD file")

        # Convert to PDB the first frame
        pt.write_traj(str(self.pdb_path), self.traj_noWAT, frame_indices=[0], overwrite=True)

    def cluster_traj(self, cluster_opts):
        try:
            mask_wat = cluster_opts["MASK_WAT"]
            mask_cluster = cluster_opts["MASK_CLUSTER"]
            n_clust = cluster_opts["NUM"]
            slice = cluster_opts["SLICE"]
            TMP_TRAJ = pt.iterload(str(self.traj_path), str(self.top_path), stride=slice)[mask_wat]
            TMP_TRAJ.autoimage()
            TMP_TRAJ.superpose(mask="@CA", ref=0)     
            cluster_data = kmeans(TMP_TRAJ, mask=mask_cluster, n_clusters=n_clust)
            centroids = list(cluster_data.centroids)
            self.traj_Cluster = TMP_TRAJ[centroids]
            
            del TMP_TRAJ
    
            # # FIXME: Probably sub-optimal
            # cluster_df = pd.DataFrame({
            #     "Cluster_ID": list(range(len(cluster_data.centroids))),
            #     "Fraction": cluster_data.fraction,
            # })
            
            pt.write_traj(
                str(self.cluster_path),
                self.traj_Cluster,
                options="model",
                overwrite=True
            )
        except Exception as e:
            print(f"Cluster FAIL for {self.md_dir}\nError: {e}")
    
    def PCA(self):
        traj_PCA = self.traj_noWAT
        data_PCA = pt.pca(traj_PCA, mask='!@H=', n_vecs=2)
        data_PCA = pt.pca(traj_PCA, mask='@CA,@N', n_vecs=2)
        
        PCA = data_PCA[0]
        return PCA, traj_PCA
    
    def plot_PCA(self, title="PCA", filepath=None):
        PCA, traj_PCA = self.PCA()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(PCA[0], PCA[1], marker='o', c=range(traj_PCA.n_frames), alpha=0.5)
        
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.grid(False)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Frame")
        
        if filepath is not None:
            fig.savefig(filepath)
        
        plt.show()

class MDA_Analysis():
    """
    Handles molecular dynamics analysis using MDAnalysis.
    
    This class provides functions for analyzing:
    - RMSF (Root Mean Square Fluctuation)
    - RMSD (Root Mean Square Deviation)
    - Radius of gyration
    - Normal mode analysis (Gaussian elastic network)
    - PCA analysis
    
    Parameters:
    -----------
    md_dir : str or Path
        Path to the directory containing the preprocessed files
        (Must contain Step3_Md_Rep0_noWAT.xtc and system_noWAT.prmtop)
        
    Attributes:
    -----------
    universe : MDAnalysis.Universe
        MDAnalysis Universe object containing the topology and trajectory
    """
    def __init__(self, md_dir):
        md_dir_path = Path(md_dir)
        traj_harmon_path = md_dir_path / "Step3_Md_Rep0_noWAT.xtc"
        assert traj_harmon_path.exists()
        
        prmtop_harmon_path = md_dir_path / "system_noWAT.prmtop"
        assert prmtop_harmon_path.exists()
        
        self.md_dir = md_dir_path
        self.universe = mda.Universe(str(prmtop_harmon_path), str(traj_harmon_path))
    
    def calc_rmsf(self):
        self.universe.trajectory[0]
        # Precompute RMSF
        average = align.AverageStructure(self.universe, self.universe, select='protein and name CA', ref_frame=0).run()
        ref = average.results.universe
        aligner = align.AlignTraj(self.universe, ref, select='protein and name CA', in_memory=True).run()
        
        c_alphas = self.universe.select_atoms('protein and name CA')
        R = rms.RMSF(c_alphas).run()
        rmsf = R.results.rmsf
        return rmsf, c_alphas
    
    def plot_rmsf(self):
        rmsf, c_alphas = self.calc_rmsf()
        
        # Plot the chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(c_alphas.resids, rmsf, label=str(self.md_dir))
        ax.legend()
        ax.set_xlabel("Residue")
        ax.set_ylabel("RMSF (Å)")
        
        plt.show()
    
    def calc_rmsd(self):
        # Restart trajectory
        self.universe.trajectory[0]
        rmsd = rms.RMSD(self.universe, select="all", groupselections=["protein and backbone", "resname UNK"]).run().results.rmsd
        return rmsd
    
    def plot_rmsd(self):
        rmsd = self.calc_rmsd()
        
        # Plot the chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rmsd[:, 0], rmsd[:, -2:])
        ax.set_xlabel("Frame")
        ax.set_ylabel("RMSD (Å)")
        
        plt.show()
    
    def gaussian_elastic(self, close=False):
        self.universe.trajectory[0]
        if close == False:
            nma = gnm.GNMAnalysis(self.universe, select='protein and name CA', cutoff=7.0).run()
            nma_res = nma.results
            return nma_res
        else:
            nma = gnm.closeContactGNMAnalysis(self.universe, select='protein and name CA',
                                              cutoff=7.0, weights="size").run()
            nma_res = nma.results
            return nma_res
    
    def radgyr_run(self):
        self.universe.trajectory[0]
        def radgyr(atomgroup, masses, total_mass=None):
            # coordinates change for each frame
            coordinates = atomgroup.positions
            center_of_mass = atomgroup.center_of_mass()
        
            # get squared distance from center
            ri_sq = (coordinates-center_of_mass)**2
            # sum the unweighted positions
            sq = np.sum(ri_sq, axis=1)
            sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
            sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
            sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y
        
            # make into array
            sq_rs = np.array([sq, sq_x, sq_y, sq_z])
        
            # weight positions
            rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
            # square root and return
            return np.sqrt(rog_sq)
        
        protein_sel = self.universe.select_atoms("protein")
        
        rog = AnalysisFromFunction(radgyr, self.universe.trajectory, protein_sel, protein_sel.masses, total_mass=np.sum(protein_sel.masses)).run()
        return rog.results["timeseries"]
    
    def radgyr_run_plot(self):
        rog = self.radgyr_run()
        
        labels = ['all', 'x-axis', 'y-axis', 'z-axis']
        for col, label in zip(rog.T, labels):
            plt.plot(col, label=label)
            
        plt.legend()
        plt.ylabel('Radius of gyration (Å)')
        plt.xlabel('Frame')
        plt.show()
    
    def PCA_mda(self, components=2):
        # Reload the universe
        u = self.universe
        
        # Define the backbone
        selection_str = "backbone"
        selection_md = u.select_atoms(selection_str)
        
        # Run the PCA
        pc = pca.PCA(u, select=selection_str,
                     align=True, mean=None,
                     n_components=None).run()
        
        # Obtain the cumulative variance
        cumulative_variance_df = pd.DataFrame(pc.results.cumulated_variance[:3], columns=["Cumulative Variance"])
        cumulative_variance_df.index = ["PC1", "PC2", "PC3"]
        cumulative_df = cumulative_variance_df.to_dict()["Cumulative Variance"]
        
        # Obtain the transformed frames
        transformed = pc.transform(selection_md, n_components=3)
        transformed_df = pd.DataFrame(transformed, columns=["PC1", "PC2", "PC3"])
        transformed_df["Frames"] = range(u.trajectory.n_frames)
        
        return cumulative_df, transformed_df
    
    def plot_PCA_3D(self, title="PairGrid PCA", filepath=None):
        
        _, transformed_df = self.PCA_mda()
        
        g = sns.PairGrid(transformed_df, hue="Frames", palette=sns.color_palette("viridis", self.universe.trajectory.n_frames))
        g.map(plt.scatter, marker=".")
        
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    
        if filepath is not None:
            g.fig.savefig(filepath)
        
        plt.show()
    
class MD_Analyzer(Pytraj_Analysis, MDA_Analysis):
    """
    Combines Pytraj and MDAnalysis functionality for comprehensive MD analysis.
    
    This class inherits from both Pytraj_Analysis and MDA_Analysis to provide
    a complete set of tools for trajectory processing and analysis.
    
    Parameters:
    -----------
    md_dir : str or Path
        Path to the directory containing the MD simulation files
    traj_path : str or Path
        Path to the trajectory file (DCD format with water)
    top_path : str or Path
        Path to the topology file (PRMTOP format with water)
    overwrite : bool
        Whether to overwrite existing processed files
    
    Usage:
    ------
    This class is typically instantiated for each simulation directory and
    stored in a dictionary under the 'CLASS' key:
    
    file_dict['simulation_dir']['CLASS'] = MD_Analyzer(...)
    """
    def __init__(self, md_dir, traj_path, top_path, overwrite):
        
        Pytraj_Analysis.__init__(self, md_dir, traj_path, top_path, overwrite)
        MDA_Analysis.__init__(self, md_dir)

##########################
### Plotting Functions ###
##########################
"""
Plotting Functions
-----------------
All plotting functions follow the same pattern:
1. Accept a dictionary where keys are directory names and values are dictionaries
2. Each value dictionary must contain a 'CLASS' key with an MD_Analyzer object
3. The directory name is used as the plot label
4. Extract data from the analyzers and create visualizations

Example dictionary structure:
{
  'simulation_1': {
    'PRMTOP_WAT': '/path/to/file.prmtop',
    'DCD_WAT': '/path/to/file.dcd',
    ...
    'CLASS': <MD_Analyzer object>
  },
  'simulation_2': {
    ...
  }
}
"""
  
def plot_RMSF_inplace(analyzer_dict):
    """
    Plot RMSF values for multiple analyzers using a dictionary.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain a 'CLASS' key with the analyzer object.
    """
    fig = go.Figure()
    for label, data in analyzer_dict.items():
        analyzer = data['CLASS']  # Get the analyzer object from the CLASS key
        RMSF, c_alphas = analyzer.calc_rmsf()
        fig.add_trace(go.Scatter(
            x=c_alphas.resids,
            y=RMSF[:300],
            mode='lines',
            name=label
        ))
        
    fig.update_layout(
        title='RMSF Plot',
        xaxis_title='Residue',
        yaxis_title='RMSF (Å)',
        legend_title='MD Directory',
        height=600
    )
    fig.show()

def plot_RMSD_inplace(analyzer_dict):
    """
    Plot RMSD values for multiple analyzers using a dictionary.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain a 'CLASS' key with the analyzer object.
    """
    fig = go.Figure()
    df_RMSD_lst = []
    
    for label, data in analyzer_dict.items():
        analyzer = data['CLASS']  # Get the analyzer object from the CLASS key
        RMSD = analyzer.calc_rmsd()
        fig.add_trace(go.Scatter(
            x=RMSD[:, 0],
            y=RMSD[:, 4],
            mode='lines',
            name=label
        ))
        
        # Save a DF
        df_RMSD = pd.DataFrame({
            "Complex": label,
            "Frame": RMSD[:, 0],
            "RMSD": RMSD[:, 4]
        })
        
        # Calculate mean RMSD for this complex
        mean_rmsd = np.mean(RMSD[:, 4])
        
        # Add mean as a column
        df_RMSD["Mean_RMSD"] = mean_rmsd
        
        df_RMSD_lst.append(df_RMSD)
    
    fig.update_layout(
        title='RMSD Plot',
        xaxis_title='Frame',
        yaxis_title='RMSD (Å)',
        legend_title='MD Directory',
        height=800,
    )
    fig.show()
    
    # Concatenate all dataframes
    df_combined = pd.concat(df_RMSD_lst)
    
    # Create summary dataframe with mean values
    df_summary = df_combined.groupby('Complex')['RMSD'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_summary = df_summary.sort_values('mean')
    
    print("Summary of RMSD values by complex (sorted by mean):")
    print(df_summary)
    
    return df_combined, df_summary

def plot_Radius_inplace(analyzer_dict):
    """
    Plot Radius of Gyration for multiple analyzers using a dictionary.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain a 'CLASS' key with the analyzer object.
    """
    fig = go.Figure()
    for label, data in analyzer_dict.items():
        analyzer = data['CLASS']  # Get the analyzer object from the CLASS key
        radgyr_array = analyzer.radgyr_run()
        frames = list(range(0, len(radgyr_array)))
        fig.add_trace(go.Scatter(
            x=frames,
            y=radgyr_array[:, 0],
            mode='lines',
            name=label
        ))
    
    fig.update_layout(
        title='Radius of Gyration',
        xaxis_title='Frame',
        yaxis_title='RMSD (Å)',
        legend_title='MD Directory',
        height=600
    )
    fig.show()

def plot_Gaussian_inplace(analyzer_dict):
    """
    Plot Gaussian Network Model for multiple analyzers using a dictionary.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain a 'CLASS' key with the analyzer object.
    """
    fig = go.Figure()
    for label, data in analyzer_dict.items():
        analyzer = data['CLASS']  # Get the analyzer object from the CLASS key
        gaussian = analyzer.gaussian_elastic(close=False)
        eigenvalues = gaussian["eigenvalues"]
        time = gaussian["times"]
        
        fig.add_trace(go.Scatter(x=time, y=eigenvalues, mode='lines', name=label))
        
    fig.update_layout(
        title='Gaussian Network Model',
        xaxis_title='Time',
        yaxis_title='Eigenvalue',
        legend_title='MD Directory',
        height=600
    )
    fig.show()

def plot_PCA_inplace(analyzer_dict):
    """
    Plot PCA for multiple analyzers using a dictionary.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain a 'CLASS' key with the analyzer object.
    """
    n_plots = len(analyzer_dict)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig_width = 6 * n_cols  # 6 inches per column
    fig_height = 5 * n_rows  # 5 inches per row

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Handle the case where there's only one plot
    if n_plots == 1:
        axs = np.array([axs])
    
    for ax, (title, data) in zip(axs.flatten(), analyzer_dict.items()):
        analyzer = data['CLASS']  # Get the analyzer object from the CLASS key
        PCA, traj_PCA = analyzer.PCA()
        scatter = ax.scatter(PCA[0], PCA[1], marker='o', c=range(traj_PCA.n_frames), alpha=0.5)
        
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.axhline(y=0, color='gray', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--')
        ax.grid(False)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Frame")
    
    # Handle empty subplots if there are more subplots than analyzers
    for i in range(n_plots, len(axs.flatten())):
        axs.flatten()[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()

####################################
### Trajectory Archive Functions ###
####################################

def create_trajectory_archive(zip_filename, file_paths):
    """
    Create a zip archive containing trajectory files from multiple simulation directories.
    
    Parameters:
    -----------
    zip_filename : str
        Path to the output zip file
    file_paths : dict
        Dictionary containing paths to trajectory files for each run directory
    """
    zip_path = Path(zip_filename)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for run_dir, files in file_paths.items():
            # Add file to the zip file, with the appropriate directory structure
            
            prmtop_nowat = files.get("PRMTOP_noWAT", "")
            prmtop_wat = files.get("PRMTOP_WAT", "")
            xtc = files.get("XTC_noWAT", "")
            dcd = files.get("DCD_noWAT", "")
            dcd_wat = files.get("DCD_WAT", "")
            pdb = files.get("PDB_noWAT", "")
            cluster = files.get("CLUSTER", "")
            
            if prmtop_nowat:
                prmtop_path = Path(prmtop_nowat)
                arcname = str(Path(run_dir) / prmtop_path.name)
                zipf.write(prmtop_nowat, arcname)
            
            if prmtop_wat:
                prmtop_path = Path(prmtop_wat)
                arcname = str(Path(run_dir) / prmtop_path.name)
                zipf.write(prmtop_wat, arcname)
            
            if dcd_wat:
                dcd_path = Path(dcd_wat)
                arcname = str(Path(run_dir) / dcd_path.name)
                zipf.write(dcd_wat, arcname)
            
            if xtc:
                xtc_path = Path(xtc)
                arcname = str(Path(run_dir) / xtc_path.name)
                zipf.write(xtc, arcname)    
            
            if dcd:
                dcd_path = Path(dcd)
                arcname = str(Path(run_dir) / dcd_path.name)
                zipf.write(dcd, arcname)
    
            if pdb:
                pdb_path = Path(pdb)
                arcname = str(Path(run_dir) / pdb_path.name)
                zipf.write(pdb, arcname)
    
            if cluster:
                cluster_path = Path(cluster)
                arcname = str(Path(run_dir) / cluster_path.name)
                zipf.write(cluster, arcname)


###################################
### Trajectory Viewer Functions ###
###################################
        
class TrajectoryViewer:
    """
    Interactive visualization of molecular dynamics trajectories using nglview.
    
    This class provides a convenient interface for visualizing trajectories with
    customized representations, focusing on the ligand and its surrounding residues.
    
    Parameters:
    -----------
    analyzer_dict : dict
        Dictionary with directory names as keys and subdictionaries as values.
        Each subdictionary must contain 'XTC_noWAT' and 'PRMTOP_noWAT' keys
        with paths to the trajectory and topology files.
    dirname : str
        Key in analyzer_dict specifying which trajectory to visualize
        
    Attributes:
    -----------
    analyzer_dict : dict
        Reference to the input dictionary containing file paths
    dirname : str
        Directory name key used to access the specific trajectory
    nv : module
        Reference to the nglview module
        
    Methods:
    --------
    find_neight(traj)
        Identifies residues within 5.5Å of the ligand
    __call__()
        Creates and returns an interactive nglview widget with custom representations
        
    Raises:
    -------
    ImportError
        If nglview is not installed
        
    Notes:
    ------
    The visualization includes:
    1. Protein backbone as cartoon colored by residue index
    2. Ligand as licorice representation (non-hydrogen atoms)
    3. Binding site residues as licorice representation (non-hydrogen atoms)
    """
    def __init__(self, analyzer_dict, dirname):
        if not HAS_NGLVIEW:
            raise ImportError("nglview is required for visualization but not installed. "
                             "Install with: pip install nglview")
            
        self.analyzer_dict = analyzer_dict
        self.dirname = dirname
        self.nv = nv  # Save for later use
    
    @staticmethod
    def find_neight(traj):
        """
        Find residues within 5.5Å of the ligand.
        
        Parameters:
        -----------
        traj : pytraj.Trajectory
            Trajectory object to analyze
            
        Returns:
        --------
        str
            String of residue IDs formatted for nglview selection
            (e.g., "1 or 2 or 3")
        """
        from itertools import chain
        atom_ndx = pt.search_neighbors(traj, mask=":UNK<:6.5")
        
        atom_ndx_2d = []
        for sub in atom_ndx:
            sub_list = list(sub)
            atom_ndx_2d.append(sub_list)
        
        atom_ndx_flatten = list(set(list(chain(*atom_ndx_2d))))

        all_resids = []
        for atom in atom_ndx_flatten:
            resid = traj.top.atom(atom).resid
            all_resids.append(resid)
        
        resids_unique = list(set(all_resids))
        resids_unique_str = " or ".join(map(str, resids_unique))

        return resids_unique_str

    def __call__(self):
        """
        Create and configure an interactive nglview visualization.
        
        Returns:
        --------
        nglview.NGLWidget
            Interactive widget for trajectory visualization
        """
        traj = pt.iterload(self.analyzer_dict[self.dirname]["XTC_noWAT"], 
                          self.analyzer_dict[self.dirname]["PRMTOP_noWAT"])

        N_list = self.find_neight(traj)
        
        view = self.nv.show_pytraj(traj)
        
        view.clear_representations()
        view.representations = [
            {
                "type":"cartoon",
                "params": {"sele":"protein", "color":"residueindex"}
            },
            {
                "type":"licorice",
                "params": {"sele":"(ligand) and not (_H)"}
            },
            {
                "type":"licorice",
                "params": {"sele":f"({N_list}) and not (_H)"}
            },
            # {
            #     "type":"surface",
            #     "params": {"sele":"protein and not ligand", "color":"blue", "wireframe":True, "opacity":0.6, "isolevel":3.}
            # }
        ]
        
        view.center("ligand")
        return view