#!/bin/bash
set -e

module --force purge
module load LUMI/24.03
module load lumi-container-wrapper
module load PrgEnv-cray

DIRECTORY="env_cheminf_rocm"

# Check if the directory exists
if [ -d "$DIRECTORY" ]; then
  echo "Directory exists. Removing it"
  rm -r $DIRECTORY
else
  echo "Directory does not exist. Moving on."
fi

mkdir $DIRECTORY

cat << 'EOF' > post_install.sh
#!/bin/bash
module load PrgEnv-cray

mamba remove --force openmm -y
pip install openmm[hip6]

export CC=$(which mpicc)
export MPICC=$(which mpicc)
export CXX=$(which mpicxx)
export FC=$(which mpifort)

pip install mpi4py --no-cache-dir
pip install cloudpickle
pip install plotly
pip install -U jedi-language-server

EOF
chmod +x post_install.sh	

conda-containerize new --mamba \
--post-install ./post_install.sh \
--prefix $DIRECTORY cheminf_rocm.yml

rm post_install.sh

else
echo "Invalid way: $WAY"
echo "Usage: $0 <way>"
echo "  way: 1 or 2"
exit 1
fi