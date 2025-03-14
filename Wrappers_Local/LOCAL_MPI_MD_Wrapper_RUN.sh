#!/bin/bash

cat << EOF > local_apo.toml
[mode]
type = "legacy"  # Use "legacy" mode for explicit protein-ligand pairs

[legacy.runs]
Run1 = { protein = "examples/LAG3_1.pdb", ligand = "examples/Ligand_1.sdf" }
Run2 = { protein = "examples/LAG3_2.pdb", ligand = "APO" }

[system]
delta_pico = 0.002
rerun = false

[nvt]
steps = 50
dcd_save = 10
log_save = 1

[npt]
steps = 50
dcd_save = 10
log_save = 1

[md]
steps = 200
dcd_save = 25
log_save = 10
EOF

mpirun -np 1 python LOCAL_MPI_MD_Wrapper.py local_apo.toml

rm -rf ./local_apo.toml