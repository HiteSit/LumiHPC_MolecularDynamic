from sys import stdout
import numpy as np
import cloudpickle
from pdbfixer import PDBFixer
from pathlib import Path
from typing import Union, Optional, Dict, Any

import parmed as pmd
import pytraj as pt

# OpenMM Application Layer
from openmm import app
from openmm import XmlSerializer
from openmm.app import (
    Modeller, Simulation, PDBFile, DCDReporter, 
    StateDataReporter, CheckpointReporter
)

# OpenMM Library Layer
from openmm import (
    Platform, LangevinIntegrator, MonteCarloBarostat,
    CustomExternalForce
)

# OpenMM Units
from openmm import unit

# OPENFF
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

class Molecular_Dynamics:
    def __init__(self, pdb_path, sdf_path, n_gpu, output_dir=None):
        
        self.pdb_path = Path(pdb_path)
        self.sdf_path = Path(sdf_path) if sdf_path != "APO" else sdf_path
        
        try:
            self.n_gpu = n_gpu
            self.platform = Platform.getPlatformByName('HIP')
            self.proprieties = {'Precision': 'mixed', 'DeviceIndex': self.n_gpu}
        except Exception as e:
            self.n_gpu = n_gpu
            self.platform = Platform.getPlatformByName('CUDA')
            self.proprieties = {'Precision': 'mixed', 'CudaDeviceIndex': self.n_gpu}

        # If sdf_path is "APO" or not a valid file, we're doing APO simulation
        self.is_apo = sdf_path == "APO" or not (isinstance(self.sdf_path, Path) and self.sdf_path.exists() and self.sdf_path.is_file())
        
        # If output_dir is provided, use it for the workdir
        if output_dir:
            self.output_dir = Path(output_dir)
            # Print simulation mode with output directory
            print(f"Running {'APO' if self.is_apo else 'Complex'} simulation in specified directory: {self.output_dir}")
            self.workdir = self.output_dir
        else:
            # Legacy behavior: use ligand_path for workdir name
            if isinstance(self.sdf_path, Path):
                self.workdir = Path(self.sdf_path.name.split(".")[0])
            else:
                self.workdir = Path(sdf_path)  # when sdf_path is "APO"
            
            # Print simulation mode
            print(f"Running {'APO' if self.is_apo else 'Complex'} simulation in directory: {self.workdir}")
            
        # Create the workdir if it doesn't exist
        self.workdir.mkdir(exist_ok=True)

    @staticmethod
    def combine_topology(protein_fixer, lig_top=None, system_generator=None):
        # Create modeller from protein
        modeller = Modeller(protein_fixer.topology, protein_fixer.positions)

        # Add ligand to model if provided
        if lig_top is not None:
            modeller.add(lig_top.to_openmm(), lig_top.get_positions().to_openmm())

        # Add solvent to system
        if system_generator is not None:
            modeller.addSolvent(system_generator.forcefield,
                         model='tip3p',
                         padding=1 * unit.nanometer,
                         ionicStrength=0.15 * unit.molar)

        return modeller

    def _ligand_preparation(self):
        """Process ligand file into OpenMM compatible object"""
        if not self.is_apo:
            # Convert Path to string as Molecule.from_file expects a string path
            ligand_mol = Molecule.from_file(str(self.sdf_path))
            lig_top = ligand_mol.to_topology()
            return ligand_mol, lig_top
        return None, None

    def _protein_preparation(self):
        # Convert Path to string for PDBFixer
        fixer = PDBFixer(str(self.pdb_path))
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        print("Missing residues: ", fixer.missingResidues)
        print("Missing atoms: ", fixer.missingAtoms)
        print("Nonstandard residues: ", fixer.nonstandardResidues)

        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
        
        return fixer
        
    def _parmed_writer(self):
        # Convert OpenMM Topology to Parmed
        parmed_object = pmd.openmm.topsystem.load_topology(self.modeller.topology, self.system, self.modeller.positions)

        # Add bond type
        bond_type = pmd.BondType(1, 1, list=parmed_object.bond_types)
        parmed_object.bond_types.append(bond_type)
        
        for bond in parmed_object.bonds:
            if bond.type is None:
                bond.type = bond_type

        # Save Prmtop
        save_path_prmtop = Path(self.workdir) / "system.prmtop"
        parmed_object.save(str(save_path_prmtop), format="amber", overwrite=True)

        # Save inpcrd
        save_path_inpcrd = Path(self.workdir) / "system.inpcrd"
        parmed_object.save(str(save_path_inpcrd), format="rst7", overwrite=True)

        # Add Radii
        from parmed.tools import changeradii
        amber_object = pmd.amber.AmberParm(str(save_path_prmtop))
        changeradii.amber6(amber_object)
        amber_object.save(str(save_path_prmtop), overwrite=True)

    def system_generator(self, init_temp, delta_pico, system_restart: Optional[Path] = None):
        if system_restart is None:
            # Load the protein and ligand if needed
            ligand_mol, ligand_top = self._ligand_preparation()
            protein_fixer = self._protein_preparation()
            
            # Setup the forcefield with appropriate parameters
            forcefield_kwargs = {
                'constraints': app.HBonds, 'rigidWater': False, 'removeCMMotion': False,
                'hydrogenMass': 4 * unit.amu
            }
            
            if self.is_apo:
                # APO system - no ligand forcefield needed
                system_generator = SystemGenerator(
                    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
                    forcefield_kwargs=forcefield_kwargs
                )
            else:
                # Complex system - include ligand forcefield
                system_generator = SystemGenerator(
                    forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
                    small_molecule_forcefield='gaff-2.11',
                    molecules=[ligand_mol],
                    forcefield_kwargs=forcefield_kwargs
                )
                
            self.modeller = self.combine_topology(protein_fixer, ligand_top, system_generator)
            with open(Path(self.workdir) / "modeller.pkl", "wb") as f:
                cloudpickle.dump(self.modeller, f)
        else:
            with open(Path(self.workdir) / "modeller.pkl", "rb") as f:
                self.modeller = cloudpickle.load(f)
    
        # Create system with or without ligand
        if system_restart is None:
            if self.is_apo:
                self.system = system_generator.create_system(self.modeller.topology)
            else:
                self.system = system_generator.create_system(self.modeller.topology, molecules=ligand_mol)
        else:
            with open(system_restart) as f:
                self.system = XmlSerializer.deserialize(f.read())

        self.integrator = LangevinIntegrator(
            init_temp * unit.kelvin,
            1 / unit.picosecond,
            delta_pico * unit.picoseconds,
        )

        if system_restart is None:
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, platform=self.platform, platformProperties=self.proprieties)
            context = self.simulation.context
            context.setPositions(self.modeller.positions)
        else:
            self.simulation = Simulation(self.modeller.topology, self.system, self.integrator, platform=self.platform, platformProperties=self.proprieties)
            
        system_size = len(list(self.modeller.topology.atoms()))
        print(f"The size is: {system_size}")

    def setup_positional_restraints(self, k:int):
        restraints = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        self.system.addForce(restraints)
        restraints.addGlobalParameter('k', k * unit.kilojoules_per_mole/unit.nanometer)

        restraints.addPerParticleParameter('x0')
        restraints.addPerParticleParameter('y0')
        restraints.addPerParticleParameter('z0')

        return restraints

    def remove_all_restraints(self):
        '''
        Remove the external force that generate the positional restraint
        '''
        for num, force in enumerate(self.system.getForces()):
            if isinstance(force, CustomExternalForce):
                self.system.removeForce(num)

    class Minimization:            
        def minimize(self):
            '''
            Minimize the system
            '''
            min_state = self.simulation.context.getState(getEnergy=True, getPositions=True)
            energy_before_min = min_state.getPotentialEnergy()
            print(f"Initial energy: {energy_before_min}")

            print("Beginning Minimization")
            self.simulation.minimizeEnergy()
            print("End Minimization")

            min_state = self.simulation.context.getState(getEnergy=True, getPositions=True)
            energy_after_min = min_state.getPotentialEnergy()
            print(f"After Minimization: {energy_after_min}")

            save_path = Path(self.workdir) / "Minimized.pdb"
            with open(save_path, "w") as f:
                PDBFile.writeFile(self.simulation.topology, min_state.getPositions(), f)

    class Nvt:
        def restraints_water(self):
            restraints = self.setup_positional_restraints(100000000000000)
            for residue in self.modeller.topology.residues():
                if residue.name not in ["HOH"]:
                    for atom in residue.atoms():
                        restraints.addParticle(atom.index, self.modeller.positions[atom.index])
        
        def setup_reporter(self, basename, nvt_steps, steps_saving, steps_log, append):
            nvt_state = self.simulation.context.getState(getEnergy=True)
            time_elapsed = nvt_state.getTime().value_in_unit(unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = nvt_steps + steps_elapsed

            save_path_dcd = Path(self.workdir) / f"{basename}.dcd"
            save_path_log = Path(self.workdir) / f"{basename}.log"

            self.simulation.reporters.append(DCDReporter(str(save_path_dcd), steps_saving, append=append))
            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(str(save_path_log), steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))       

        def run(self, steps, temp):
            self.integrator.setTemperature(temp)
            print(f"Running NVT at {temp}K for {steps} steps...")
            self.simulation.step(steps)

    class Npt:
        def add_barostat(self):
            # Find and remove any existing barostats
            barostat_indices = []
            for i in range(self.system.getNumForces()):
                if isinstance(self.system.getForce(i), MonteCarloBarostat):
                    barostat_indices.append(i)
                    
            for i in sorted(barostat_indices, reverse=True):
                self.system.removeForce(i)
                
            # Add a new barostat at 1 bar
            self.system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
            
            # Update the simulation context
            self.simulation.context.reinitialize(preserveState=True)

        def restraint_backbone(self, restr_force, atoms_to_res:set):
            # Restraints the atoms
            self.remove_all_restraints()
            restraint = self.setup_positional_restraints(restr_force)
            for atom in self.modeller.topology.atoms():
                if atom.name in atoms_to_res:
                    restraint.addParticle(atom.index, self.modeller.positions[atom.index])
            
            # Restraints the residues
            for residue in self.modeller.topology.residues():
                if residue.name in ["MOL"]:
                    for atom in residue.atoms():
                        restraint.addParticle(atom.index, self.modeller.positions[atom.index])
        
        def setup_reporter(self, basename, npt_steps, steps_saving, steps_log, append):
            npt_state = self.simulation.context.getState(getEnergy=True)
            time_elapsed = npt_state.getTime().value_in_unit(unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = npt_steps + steps_elapsed

            save_path_dcd = Path(self.workdir) / f"{basename}.dcd"
            save_path_log = Path(self.workdir) / f"{basename}.log"

            self.simulation.reporters.append(DCDReporter(str(save_path_dcd), steps_saving, append=append))
            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(str(save_path_log), steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))

        def run(self, steps):
            print(f"Running NPT for {steps} steps...")
            self.simulation.step(steps)

    class Plain_Md():
        def setup_reporter(self, basename, md_steps, steps_saving, steps_log, append):
            md_state = self.simulation.context.getState(getEnergy=True)
            time_elapsed = md_state.getTime().value_in_unit(unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = md_steps + steps_elapsed

            save_path_dcd = Path(self.workdir) / f"{basename}.dcd"
            save_path_log = Path(self.workdir) / f"{basename}.log"
            save_path_checkpoint = Path(self.workdir) / f"{basename}.chk"
        
            self.simulation.reporters.append(DCDReporter(str(save_path_dcd), steps_saving, append=append))

            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(str(save_path_log), steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))  
            
            self.simulation.reporters.append(CheckpointReporter(str(save_path_checkpoint), steps_saving))

        def run(self, basename, steps):
            np.random.seed(33)
            self.simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, np.random.randint(1, 1e6))
            print(f"Running Production MD for {steps} steps...")
            self.simulation.step(steps)

            save_path_checkpoint = Path(self.workdir) / f"{basename}.chk"
            self.simulation.saveCheckpoint(str(save_path_checkpoint))

class Run_MD(Molecular_Dynamics):
    def __init__(self):
        pass

    @staticmethod
    def equilibration_production(system_settings, nvt_settings, npt_settings, md_settings):
        # Check for Rerun
        rerun = system_settings["rerun"]
        
        # Get output directory from system_settings if available
        output_dir = system_settings.get("output_dir")

        if rerun == False:
            ## Initialize the class
            md = Molecular_Dynamics(system_settings["receptor_path"], system_settings["ligand_path"], 
                                  "0", output_dir=output_dir)

            # Create the System
            print("Creating the System")
            delta_pico = system_settings["delta_pico"]
            md.system_generator(nvt_settings["temps_list_simulating"][0], delta_pico)

            # Save PRMTOP
            md._parmed_writer()
            
            # Save XML
            with open(Path(md.workdir) / "system.xml", "w") as f:
                f.write(XmlSerializer.serialize(md.system))

            ## Minimize the System
            print("Minimizing the System")
            md.Minimization.minimize(md)

            ## NVT
            # Restraints the molecules of Water
            print("Setting up NVT with water restraints")
            md.remove_all_restraints()
            md.Nvt.restraints_water(md)

            # Choose temperature gradient
            temps = nvt_settings["temps_list_simulating"]
            partial_steps = nvt_settings["steps"] // len(temps)

            # Setup Reporters
            md.simulation.reporters.clear()
            md.Nvt.setup_reporter(md, "Step1_Nvt", nvt_settings["steps"], nvt_settings["dcd_save"], nvt_settings["log_save"], False)

            # Run NVT
            print("Running NVT")
            for t in temps:
                print(f"Temp = {t}")
                md.Nvt.run(md, partial_steps, t)

            ## NPT
            # Remove all previous restraints
            md.remove_all_restraints()

            # Add barostat
            print("Setting up NPT with protein restraints")
            md.Npt.add_barostat(md)

            # Choose restraints gradient
            restr_list = npt_settings["rests_list_decreasing"]
            partial_steps = npt_settings["steps"] // len(restr_list)

            # Setup reporters
            md.simulation.reporters.clear()
            md.Npt.setup_reporter(md, "Step2_Npt", npt_settings["steps"], npt_settings["dcd_save"], npt_settings["log_save"], False)

            # Run NPT
            for r in restr_list:
                md.Npt.restraint_backbone(md, r, npt_settings["atoms_to_restraints"])
                print(f"Restr = {r}")
                md.Npt.run(md, partial_steps)
                md.remove_all_restraints()

            # Remove all restraints and save the last state
            md.remove_all_restraints()
            md.simulation.reporters.clear()
            final_npt_checkpoint = "step2_last_NVT.chk"
            checkpoint_path = Path(md.workdir) / final_npt_checkpoint
            md.simulation.saveCheckpoint(str(checkpoint_path))

            # MD - Production
            print("Setting up production MD without restraints")
            # Setup the reporters
            md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"], md_settings["log_save"], False)

            # Run MD
            md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])

        elif rerun == True:
            # Initialize the class
            md = Molecular_Dynamics(system_settings["receptor_path"], system_settings["ligand_path"], 
                                  "0", output_dir=output_dir)

            # Create the System
            print("Creating the System")
            delta_pico = system_settings["delta_pico"]
            system_restart_path = Path(md.workdir) / "system.xml"
            md.system_generator(300, delta_pico, system_restart=system_restart_path)

            # Load the checkpoint
            final_md_checkpoint = f"Step3_Md_Rep{md.n_gpu}.chk"
            checkpoint_path = Path(md.workdir) / final_md_checkpoint
            if checkpoint_path.exists():
                print(f"Restarting from {checkpoint_path}")
                md.simulation.loadCheckpoint(str(checkpoint_path))
            else:
                print(f"Warning: Checkpoint file {checkpoint_path} not found. Starting from current state.")

            # Setup reporters
            md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"],
                                     md_settings["log_save"], True)
            md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])