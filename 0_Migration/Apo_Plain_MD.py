import os
from sys import stdout
import numpy as np
from pdbfixer import PDBFixer

import parmed as pmd
import pytraj as pt

# OPENMM
import openmm
from openmm import *

from openmm import app
from openmm.app import *

from openmm import unit as openmm_unit

# OPENFF
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator

class Molecular_Dynamics:
    def __init__(self, pdb_path, sdf_path, n_gpu):
        
        self.pdb_path = pdb_path
        self.sdf_path = sdf_path
        
        self.n_gpu = n_gpu
        self.platform = Platform.getPlatformByName('CUDA')
        self.proprieties = {'Precision': 'mixed', 'CudaDeviceIndex': self.n_gpu}

        self.workdir = os.path.basename(sdf_path).split(".")[0]
        os.makedirs(self.workdir, exist_ok=True)

    @staticmethod
    def combine_topology(protein_fixer, system_generator):
        modeller = Modeller(protein_fixer.topology, protein_fixer.positions)

        modeller.addSolvent(system_generator.forcefield,
                            model='tip3p',
                            padding=1 * openmm_unit.nanometer,
                            ionicStrength=0.15 * openmm_unit.molar)

        return modeller

    def _protein_preparation(self):
        fixer = PDBFixer(self.pdb_path)
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()

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
        save_path_prmtop = os.path.join(self.workdir, "system.prmtop")
        parmed_object.save(save_path_prmtop, format="amber", overwrite=True)

        # Save inpcrd
        save_path_inpcrd = os.path.join(self.workdir, "system.inpcrd")
        parmed_object.save(save_path_inpcrd, format="rst7", overwrite=True)

        # Add Radii
        from parmed.tools import changeradii
        amber_object = pmd.amber.AmberParm(save_path_prmtop)
        changeradii.amber6(amber_object)
        amber_object.save(save_path_prmtop, overwrite=True)

    def system_generator(self, init_temp, delta_pico):
        # Load the protein and ligand
        protein_fixer = self._protein_preparation()

        # Setup teh forcefield
        forcefield_kwargs = {'constraints': app.HBonds, 'rigidWater': False, 'removeCMMotion': False,
                             'hydrogenMass': 4 * openmm_unit.amu}
        system_generator = SystemGenerator(forcefields=['amber/ff14SB.xml', 'amber/tip3p_standard.xml'],
                                           forcefield_kwargs=forcefield_kwargs)

        # Grab the general topology of the complex
        modeller = self.combine_topology(protein_fixer, system_generator)

        self.system = system_generator.create_system(modeller.topology)
        self.integrator = openmm.LangevinIntegrator(
            init_temp * openmm_unit.kelvin,
            1 / openmm_unit.picosecond,
            delta_pico * openmm_unit.picoseconds,
        )

        self.simulation = Simulation(modeller.topology, self.system, self.integrator, platform=self.platform, platformProperties=self.proprieties)
        context = self.simulation.context
        context.setPositions(modeller.positions)

        # Add the topology to self
        self.modeller = modeller

    def setup_positional_restraints(self, k:int):
        restraints = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        self.system.addForce(restraints)
        restraints.addGlobalParameter('k', k*openmm_unit.kilojoules_per_mole/openmm_unit.nanometer)

        restraints.addPerParticleParameter('x0')
        restraints.addPerParticleParameter('y0')
        restraints.addPerParticleParameter('z0')

        return restraints

    def remove_all_restraints(self):
        '''
        Remove the external force that generate the positional restraint
        '''
        for num, force in enumerate(self.system.getForces()):
            if isinstance(force, openmm.CustomExternalForce):
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

            save_path = os.path.join(self.workdir, "Minimized.pdb")
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
            time_elapsed = nvt_state.getTime().value_in_unit(openmm_unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = nvt_steps + steps_elapsed

            save_path_dcd = os.path.join(self.workdir, f"{basename}.dcd")
            save_path_log = os.path.join(self.workdir, f"{basename}.log")

            self.simulation.reporters.append(DCDReporter(save_path_dcd, steps_saving, append=append))
            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(save_path_log, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))       

        def run(self, steps, temp):
            self.integrator.setTemperature(temp)
            self.simulation.step(steps)

    class Npt:
        def add_barostat(self):
            self.system.addForce(MonteCarloBarostat(1*openmm_unit.bar, 300*openmm_unit.kelvin))

        def restraint_backbone(self, restr_force, atoms_to_res:set):
            # Restraints the atoms
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
            time_elapsed = npt_state.getTime().value_in_unit(openmm_unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = npt_steps + steps_elapsed

            save_path_dcd = os.path.join(self.workdir, f"{basename}.dcd")
            save_path_log = os.path.join(self.workdir, f"{basename}.log")

            self.simulation.reporters.append(DCDReporter(save_path_dcd, steps_saving, append=append))
            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(save_path_log, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))

        def run(self, steps):
            self.simulation.step(steps)
    
    class Plain_Md():
        def setup_reporter(self, basename, md_steps, steps_saving, steps_log, append):
            md_state = self.simulation.context.getState(getEnergy=True)
            time_elapsed = md_state.getTime().value_in_unit(openmm_unit.picoseconds)
            steps_elapsed = int(time_elapsed / 0.002)

            total_steps = md_steps + steps_elapsed

            save_path_dcd = os.path.join(self.workdir, f"{basename}.dcd")
            save_path_log = os.path.join(self.workdir, f"{basename}.log")
            save_path_checkpoint = os.path.join(self.workdir, f"{basename}.chk")
        
            self.simulation.reporters.append(DCDReporter(save_path_dcd, steps_saving, append=append))

            self.simulation.reporters.append(StateDataReporter(stdout, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))
            self.simulation.reporters.append(StateDataReporter(save_path_log, steps_log, step=True,
                    potentialEnergy=True, temperature=True, volume=True, remainingTime=True, progress=True, speed=True, totalSteps=total_steps))  
            
            self.simulation.reporters.append(CheckpointReporter(save_path_checkpoint, steps_saving))

        def run(self, basename, steps):

            np.random.seed(33)
            self.simulation.context.setVelocitiesToTemperature(300*openmm_unit.kelvin, np.random.randint(1, 1e6))
            self.simulation.step(steps)

            save_path_checkpoint = os.path.join(self.workdir, f"{basename}.chk")

            self.simulation.saveCheckpoint(save_path_checkpoint)

class Run_MD(Molecular_Dynamics):
    def __init__(self):
        pass

    @staticmethod
    def equilibration_production(system_settings, nvt_settings, npt_settings, md_settings):
        # Check for Rerun
        rerun = system_settings["rerun"]

        if rerun == False:
            # Initialize the class
            md = Molecular_Dynamics(system_settings["receptor_path"], system_settings["ligand_path"], "0")

            # Create the System
            print("Creating the System")
            delta_pico = system_settings["delta_pico"]
            md.system_generator(nvt_settings["temps_list_simulating"][0], delta_pico)

            # Save PRMTOP
            md._parmed_writer()

            ## Minimize the System
            print("Minimizing the System")
            md.Minimization.minimize(md)

            ## NVT
            # Restraints the molecules of Water
            md.Nvt.restraints_water(md)

            # Choose temperature gradient
            temps = nvt_settings["temps_list_simulating"]
            partial_steps = nvt_settings["steps"] // len(temps)

            # Setup Reporters
            md.simulation.reporters.clear()
            md.Nvt.setup_reporter(md, "Step1_Nvt", nvt_settings["steps"], nvt_settings["dcd_save"], nvt_settings["log_save"],
                                  False)

            # Run NVT
            print("Running NVT")
            for t in temps:
                print(f"Temp = {t}")
                md.Nvt.run(md, partial_steps, t)

            ## NPT
            # Remove all previous restraints
            md.remove_all_restraints()

            # Add barostat
            md.Npt.add_barostat(md)

            # Choose restraints gradient
            restr_list = npt_settings["rests_list_decreasing"]
            partial_steps = npt_settings["steps"] // len(restr_list)

            # Setup reporters
            md.simulation.reporters.clear()
            md.Npt.setup_reporter(md, "Step2_Npt", npt_settings["steps"], npt_settings["dcd_save"], npt_settings["log_save"],
                                  False)

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
            md.simulation.saveCheckpoint(os.path.join(md.workdir, final_npt_checkpoint))

            # MD - Production
            # Setup the reporters
            md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"],
                                       md_settings["log_save"], False)

            # Run MD
            md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])

        elif rerun == True:
            # Initialize the class
            md = Molecular_Dynamics(system_settings["receptor_path"], system_settings["ligand_path"],
                                    system_settings["gpu_id"])

            # Create the System
            print("Creating the System")
            delta_pico = system_settings["delta_pico"]
            md.system_generator(300, delta_pico)

            # Load the checkpoint
            final_md_checkpoint = f"Step3_Md_Rep{md.n_gpu}.chk"
            final_md_checkpoint = os.path.join(md.workdir, final_md_checkpoint)
            with open(final_md_checkpoint, 'rb') as f:
                md.simulation.context.loadCheckpoint(f.read())

            # Setup reporters
            md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"],
                                       md_settings["log_save"], True)
            md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])

    # @staticmethod
    # def production(system_settings, md_settings):
    #     # Initialize the class
    #     md = Molecular_Dynamics(system_settings["receptor_path"], system_settings["ligand_path"], system_settings["gpu_id"])
    #
    #     # Create the System
    #     print("Creating the System")
    #     delta_pico = system_settings["delta_pico"]
    #     md.system_generator(300, delta_pico)
    #
    #     rerun = system_settings["rerun"]
    #     if rerun == False:
    #
    #         # Load the NVT checkpoint
    #         final_npt_checkpoint = "step2_last_NVT.chk"
    #         final_npt_checkpoint = os.path.join(md.workdir, final_npt_checkpoint)
    #         with open(final_npt_checkpoint, 'rb') as f:
    #             md.simulation.context.loadCheckpoint(f.read())
    #
    #         # Setup reporters
    #         md.simulation.reporters.clear()
    #         md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"],
    #                                    md_settings["log_save"], False)
    #
    #         # Run MD
    #         md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])
    #
    #     elif rerun == True:
    #
    #         final_md_checkpoint = f"Step3_Md_Rep{md.n_gpu}.chk"
    #         final_md_checkpoint = os.path.join(md.workdir, final_md_checkpoint)
    #
    #         # Retrieve the last checkpoint
    #         with open(final_md_checkpoint, 'rb') as f:
    #             md.simulation.context.loadCheckpoint(f.read())
    #
    #         # Setup reporters
    #         md.Plain_Md.setup_reporter(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"], md_settings["dcd_save"],
    #                                    md_settings["log_save"], True)
    #         md.Plain_Md.run(md, f"Step3_Md_Rep{md.n_gpu}", md_settings["steps"])