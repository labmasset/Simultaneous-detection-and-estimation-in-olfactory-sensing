#!/bin/bash -l
#SBATCH --job-name=SLAM_geom
#SBATCH --account=def-pmasset
#SBATCH --time=0-11:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=0-39

#SBATCH --output=logs/present_sweep/slam_geom/%A_%a.out

# Load environment modules
module load python
module load openblas

# Configure OpenBLAS threading
export FLEXIBLAS=BLIS
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1


# Go to your project directory
cd ..

# bash setup.sh
source .venv/bin/activate
# Run the common driver with SLAM model and L1 metric
# Default prior is bernoulli, change to --prior kumaraswamy if needed
# Default affinity matrix type is sparse_binary, change if needed
srun python -m mirrored_langevin_rnn.run configs/present_sweep_slam_geom.yaml