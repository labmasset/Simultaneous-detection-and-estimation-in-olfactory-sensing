#!/bin/bash -l
#SBATCH --job-name=SLAM_threshold_sweep
#SBATCH --account=def-pmasset
#SBATCH --time=0-2:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --array=0-39

#SBATCH --output=logs/present_sweep/slam/%A_%a.out

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


cd ..

source .venv/bin/activate
srun python -m mirrored_langevin_rnn.run configs/present_sweep_slam.yaml