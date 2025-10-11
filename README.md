# Simultaneous-detection-and-estimation-in-olfactory-sensing

## Running Experiments

Use the following commands from the repository root to run each example experiment:
(Gamma sweep haven't tested)

### Dynamics Demonstration (Figures 3 & Supplementary Figure ~)

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

New step increasing concentration dynamics:

```bash
python -m mirrored_langevin_rnn.run configs/simulation_steps_slam_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_steps_slam_circuit_sparse_binary.yaml &
python -m mirrored_langevin_rnn.run configs/simulation_steps_poisson_sparse_binary.yaml
```

### Separated vs. Non-separated comparison (Figures 4)

```bash
# sweeps experiments
python -m mirrored_langevin_rnn.run configs/present_sweep_slam.yaml
python -m mirrored_langevin_rnn.run configs/present_sweep_slam_circuit.yaml
python -m mirrored_langevin_rnn.run configs/present_sweep_poisson.yaml
python -m mirrored_langevin_rnn.run configs/gamma_sweep.yaml
```

### Capacity Heatmap (Figure 5)

```bash
# threshold sweep experiments
# Poisson model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_poisson_sparse_binary.yaml
# SLAM Bernoulli model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_bernoullil_sparse_binary.yaml
# SLAM Kumaraswamy model
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_dense_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_gamma.yaml
python -m mirrored_langevin_rnn.run configs/threshold_sweep_slam_ks_sparse_binary.yaml
```

### Miscellaneous

```bash
# sparsity sweep experiments
python -m mirrored_langevin_rnn.run configs/sparsity_sweep.yaml

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
