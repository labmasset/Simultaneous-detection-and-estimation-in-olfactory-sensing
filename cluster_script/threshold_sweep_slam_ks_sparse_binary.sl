#!/bin/bash -l
#SBATCH --job-name=SLAM_KS_threshold_sweep
#SBATCH --account=def-pmasset
#SBATCH --time=0-5:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --array=0-239

#SBATCH --output=logs/threshold_sweep/auc/slam_ks_sparse_binary/%A_%a.out

# Load environment modules
module load python
module load openblas

# Configure OpenBLAS threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_INTEROP_THREADS=1

cd ..
source .venv/bin/activate

srun python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_binary.yaml
