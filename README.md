# Simultaneous-detection-and-estimation-in-olfactory-sensing

## Environment Setup

Can either use environment.yml to create a conda environment or run `setup.sh` to create a virtual environment.
```bash
conda env create --name SDEO_env -f environment.yml
```

## Reproducing Figures

Notebooks are used to generate the figures from the saved data.

You could download the precomputed data from [Zenodo](~) and place it in the `data/` directory.

Go to the corresponding notebook in `notebooks/` (indicated by the file name) to generate each figure.

You could also run the experiments to generate the data yourself, by following the instructions in the "Running Experiments" section below.

After running the above experiments, data should be saved in the `data/` directory. Then run the notebooks in `notebooks/` to generate the figures.

## Running Experiments

### Local Execution

Use the following commands from the repository root to run each example experiment:

#### Dynamics Demonstration (Figure 3 & Supplementary Figure S1-S3)

Sensing matrix: Dense gamma

```bash
python -m mirrored_langevin_rnn.run configs/simulation_poisson_dense_gamma.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_slam_dense_gamma.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_slam_circuit_dense_gamma.yaml
```

Sensing matrix: Sparse binary

```bash
python -m mirrored_langevin_rnn.run configs/simulation_poisson_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_slam_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_slam_circuit_sparse_binary.yaml
```

Slow-changing concentration dynamics:

```bash
python -m mirrored_langevin_rnn.run configs/simulation_steps_slam_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_steps_slam_circuit_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_steps_poisson_sparse_binary.yaml
```

#### SDEO vs. Non-separated Comparison (Figures 5 and S5)

```bash
# sweeps experiments
python -m mirrored_langevin_rnn.run configs/present_sweep_slam.yaml
python -m mirrored_langevin_rnn.run configs/present_sweep_slam_circuit.yaml
python -m mirrored_langevin_rnn.run configs/present_sweep_poisson.yaml
python -m mirrored_langevin_rnn.run configs/gamma_sweep.yaml
```

#### Capacity Heatmap (Figures 6 and S6)

```bash
# threshold sweep experiments
# Non-separated model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_sparse_binary.yaml
# SDEO Bernoulli model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_sparse_binary.yaml
# SDEO Kumaraswamy model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_binary.yaml
```

#### Sparsity Sweep Heatmap (Figure S7-f)

Update the sparsity in `threshold_sweep_slam_ks_sparse_binary.yaml` from 0.1 to 0.5 and run the command below for each sparsity value.

```bash
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_binary.yaml
```

### Running on a SLURM Cluster

Updating the user information `.sl` files under `cluster_scripts/`. Then submit the jobs using `sbatch`.

## Miscellaneous


```bash
# single simulation debug
python -m mirrored_langevin_rnn.simulator.mld_rnn
```

Useful command for SLURM job inspection:

- print out CPU utilization percentage:

```bash
sacct -j 1215191 -n -P -o JobID,TotalCPU,CPUTimeRAW \
| awk -F'|' '
function tosec(x, d,h,m,s,rest,a){
  d=0; rest=x
  if (index(x,"-")) { split(x,a,"-"); d=a[1]; rest=a[2] }
  n=split(rest,a,":")
  h=(n==3)?a[1]:0; m=(n==3)?a[2]:(n==2)?a[1]:0; s=(n>=2)?a[n]:a[1]
  sub(/\..*$/, "", s)  # drop fractional seconds
  return d*86400 + h*3600 + m*60 + s
}
$3>0 && $1 !~ /\.(batch|extern)$/ {
  tot=tosec($2); pct=(tot/$3)*100
  printf "%s CPU Utilization: %.2f%%\n", $1, pct
  sum_tot+=tot; sum_den+= $3
}
END { if (sum_den>0) printf "Aggregate CPU Utilization: %.2f%%\n", (sum_tot/sum_den)*100 }'
```

- print out total CPU hours for array job:

```bash
sacct -j 1234567 -X -n -o CPUTimeRAW \
| awk '{s+=$1} END {printf "Total CPU hours: %.2f\n", s/3600}'
```
