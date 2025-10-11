"""
Utilities for working with experiment result files in HDF5 format.

This module provides functions for loading, merging, and preparing data
from simulation output files.
"""

from __future__ import annotations
from pathlib import Path
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Union

import h5py
import numpy as np

__all__ = [
    "merge_model_batches",
    "_combine_batches",
    "snapshot_time_indices",
    "collect_C_all",
    "collect_theta_all",
    "load_threshold_batches",
    "average_grid",
    "save_merged_grid",
    "_get_n_high",
]

# Configuration & helpers
_log = logging.getLogger(__name__)

# Attribute names used by the SimulationResult dataclass
ATTR_N_HIGH = "n_high"     # number of *high* odours in the scenario
ATTR_REPEAT = "repeat"     # repetition index
ATTR_N_LOW  = "n_low"      # stored for sanity checks only
ATTR_BATCH = "batch_index"


def _get_n_high(g) -> int:
    """
    Return the n_high value for an HDF5 run‑group *g*.

    Priority:
    1.  attribute 'n_high'   (snake case)
    2.  attribute 'nHigh'    (camel case - this is what the saver wrote)
    3.  parse it from the group's name, e.g. '…/nLow=0_nHigh=23_rep=7'
    """
    for cand in ("num_high", "nHigh"):
        if cand in g.attrs:
            return int(g.attrs[cand])

    import re
    m = re.search(r"nHigh=(\d+)", g.name.split("/")[-1])
    if m:
        return int(m.group(1))

    raise KeyError(
        f"Could not find n_high for group {g.name}. "
        "Expected attribute 'nHigh' or a name containing 'nHigh=<int>'."
    )


def _copy_group(src, dest, name):
    """Helper function to copy a group from one HDF5 file to another."""
    dest.create_dataset(name, data=src[:], compression="gzip", compression_opts=4)

def merge_model_batches(model_name: str,
                        config: dict, overwrite: bool = True,
                        logger: logging.Logger = logging.getLogger(__name__)) -> Path:
    """
    Helper function to merge batch files for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model being merged
    config : dict
        Model configuration containing 'input', 'pattern', and 'output' keys
    overwrite : bool, default=True
        Whether to overwrite existing output files

    Returns
    -------
    Path
        Path to the merged output file
    """
    try:
        merged_path = _combine_batches(
            dir_path=config["input"],
            pattern=config["pattern"],
            out_file=config["output"],
            overwrite=overwrite,
        )
        logger.info(f"Merged {model_name} model results: {merged_path}")
        return merged_path
    except FileNotFoundError:
        logger.warning(f"No {model_name} batch files found. Skipping merge.")
        return None
    
def _combine_batches(
    dir_path: Union[str, Path],
    pattern: str,
    out_file: Union[str, Path],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Merge several batch files into a single HDF5 file with the same `/runs/<key>` layout.

    Every run‑group is copied byte‑for‑byte; nothing is averaged here.
    Duplicate run keys (same nLow/nHigh/rep) raise a ValueError so you
    never silently overwrite a simulation.
    
    Parameters
    ----------
    dir_path : str or Path
        Directory containing batch files to merge
    pattern : str
        Glob pattern for batch files, e.g., "slam_naive_results_batch*.h5"
    out_file : str or Path
        Output file path for merged results
    overwrite : bool, default=False
        Whether to overwrite existing output file
        
    Returns
    -------
    Path
        Path to the merged output file
    """
    dir_path = Path(dir_path)
    out_file = Path(out_file)

    if out_file.exists():
        if not overwrite:
            raise FileExistsError(f"{out_file} exists – pass overwrite=True")
        out_file.unlink()

    src_files = sorted(dir_path.glob(pattern))
    if not src_files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {dir_path}")

    with h5py.File(out_file, "w") as fout:
        runs_out = fout.create_group("runs")

        for src in src_files:
            with h5py.File(src, "r") as fin:
                batch_idx = fin.attrs.get(ATTR_BATCH, -1)
                for key, g_src in fin["runs"].items():
                    if key in runs_out:
                        raise ValueError(
                            f"Duplicate run key {key!r} found in {src.name}"
                        )
                    # deep‑copy the entire group, including datasets & attrs
                    fin.copy(g_src, runs_out, name=key)
                # remember which batches were merged (handy for provenance)
                runs_out.attrs.setdefault("merged_batches", np.empty(0, int))
                runs_out.attrs["merged_batches"] = np.append(
                    runs_out.attrs["merged_batches"], batch_idx
                )

        fout.attrs[ATTR_BATCH] = -1  # meaning "combined file"
    
    _log.info(f"Merged {len(src_files)} files into {out_file}")
    return out_file


def snapshot_time_indices(
    h5_path: Union[str, Path],
    time_idx: Iterable[int],
    out_file: Union[str, Path] = "results_snapshot.h5",
    *,
    overwrite: bool = False,
) -> Path:
    """
    Extract specific time indices from *C* (and *Theta*) datasets.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to source HDF5 file
    time_idx : Iterable[int]
        Time indices to extract (zero-based)
    out_file : str or Path, default="results_snapshot.h5"
        Output file path
    overwrite : bool, default=False
        Whether to overwrite existing output file
        
    Returns
    -------
    Path
        Path to the snapshot file
    """
    h5_path = Path(h5_path)
    out_path = Path(out_file) if Path(out_file).is_absolute() else h5_path.with_name(out_file)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists – set overwrite=True to replace it")

    t_sel = np.asarray(list(time_idx), dtype=int)

    with h5py.File(h5_path, "r") as fin, h5py.File(out_path, "w") as fout:
        fout.attrs.update(fin.attrs)  # keep file‑level attributes (e.g. batch_index)
        runs_out = fout.create_group("runs")

        for run_key, g_in in fin["runs"].items():
            g_out = runs_out.create_group(run_key)
            # copy attributes verbatim
            for k, v in g_in.attrs.items():
                g_out.attrs[k] = v
            # slice datasets where appropriate
            for dset_name in g_in:
                d_in = g_in[dset_name]
                if dset_name in {"C", "Theta"} and d_in.ndim >= 2:
                    # preserve leading dimensions, slice on *time* axis (1)
                    sel = [slice(None)] * d_in.ndim
                    sel[1] = t_sel
                    d_out = g_out.create_dataset(
                        dset_name,
                        data=d_in[tuple(sel)],
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    _copy_group(d_in, g_out, dset_name)  # copy as‑is (U, others)

    _log.info(f"Snapshot with {len(t_sel)} time points written to {out_path.name}")
    return out_path


def collect_C_all(
    h5_path: Union[str, Path],
) -> Tuple[np.ndarray, List[int]]:
    """
    Read every run in *h5_path* and return a single 4-D array.
    
    Returns a 4-D array:
        C_all[n_odor, n_time, n_high‑1, rep]
    
    where *rep* is padded with NaNs if some n_high have fewer repetitions.
    The second return value is a list `n_high_vals` (already sorted)
    so you know which slice corresponds to which n_high.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing simulation results
        
    Returns
    -------
    C_all : np.ndarray
        4-D array of concentration estimates
    n_high_vals : List[int]
        Sorted list of n_high values
    """
    with h5py.File(h5_path, "r") as f:
        by_high: Dict[int, List[np.ndarray]] = defaultdict(list)

        for g in f["runs"].values():
            n_high = _get_n_high(g)
            by_high[n_high].append(np.asarray(g["C"]))

    n_high_vals = sorted(by_high)
    # assume at least one run per n_high → take shape from first entry
    sample_C = by_high[n_high_vals[0]][0]
    n_odor, n_time = sample_C.shape

    n_high_max = max(n_high_vals)
    max_rep = max(len(v) for v in by_high.values())

    C_all = np.full(
        (n_odor, n_time, n_high_max, max_rep), np.nan, dtype=sample_C.dtype
    )

    for k in n_high_vals:
        for j, C in enumerate(by_high[k]):
            C_all[:, :, k - 1, j] = C

    return C_all, n_high_vals


def collect_theta_all(
    h5_path: Union[str, Path],
) -> Tuple[np.ndarray, List[int]]:
    """
    Read Theta values from every run in *h5_path* and return a single 4-D array.
    
    Returns a 4-D array:
        Theta_all[n_odor, n_time, n_high‑1, rep]
    
    Similar to collect_C_all_h5 but for Theta values.
    
    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing simulation results
        
    Returns
    -------
    Theta_all : np.ndarray
        4-D array of Theta values
    n_high_vals : List[int]
        Sorted list of n_high values
    """
    with h5py.File(h5_path, "r") as f:
        by_high = defaultdict(list)
        for g in f["runs"].values():
            n_high = _get_n_high(g)
            if "Theta" in g:
                by_high[n_high].append(np.asarray(g["Theta"]))

    n_high_vals = sorted(by_high)
    sample = by_high[n_high_vals[0]][0]
    n_odor, n_time = sample.shape
    n_high_max = max(n_high_vals)
    max_rep = max(len(v) for v in by_high.values())

    theta_all = np.full((n_odor, n_time, n_high_max, max_rep),
                        np.nan, dtype=sample.dtype)
    for k in n_high_vals:
        for j, th in enumerate(by_high[k]):
            theta_all[:, :, k - 1, j] = th

    return theta_all, n_high_vals


def load_threshold_batches(
    out_dir: Path,
    pattern: str = "threshold_results_batch*.h5",
    fill_value: float = np.nan,
    dtype: Optional[np.dtype] = None,
) -> Tuple[np.ndarray, List[Path]]:
    """
    Read all threshold sweep batch files and return stacked array of grids.
    
    Parameters
    ----------
    out_dir : Path
        Directory containing batch files
    pattern : str, default="threshold_results_batch*.h5"
        Glob pattern for batch files
    fill_value : float, default=np.nan
        Fill value for padding smaller grids
    dtype : np.dtype, optional
        Data type for output array
        
    Returns
    -------
    grids : np.ndarray
        Stacked array of all grids (n_batch, max_nSens, max_nOdor)
    files : List[Path]
        Sorted list of files that were loaded
    """
    files = sorted(out_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No batch files found in {out_dir}")

    # Load once, capture shapes & dtypes
    raw_grids = []
    shapes = []
    dtypes = []
    for p in files:
        with h5py.File(p, "r") as f:
            g = f["grid"][()]  # read all at once
        raw_grids.append(g)
        shapes.append(g.shape)
        dtypes.append(g.dtype)

    # Determine common dtype that CAN hold NaNs
    if dtype is None:
        dtype = np.result_type(fill_value, *dtypes)  # usually float64

    # Compute the maximum grid size
    max_rows = max(r for r, _ in shapes)
    max_cols = max(c for _, c in shapes)

    # Allocate the final 3‑D block
    grids = np.full(
        (len(files), max_rows, max_cols), fill_value, dtype=dtype
    )

    # Copy each grid into the big block
    for i, (g, (r, c)) in enumerate(zip(raw_grids, shapes)):
        grids[i, :r, :c] = g.astype(dtype, copy=False)
        _log.debug(f"{files[i].name} → original {(r, c)}, padded to {(max_rows, max_cols)}")

    _log.info(f"Loaded {len(files)} threshold batch files into shape: {grids.shape}")
    return grids, files


def average_grid(grids: np.ndarray) -> np.ndarray:
    """
    Element-wise nan-mean of all batch grids.
    
    Parameters
    ----------
    grids : np.ndarray
        Stacked array of grids (n_batch, nSens, nOdor)
        
    Returns
    -------
    np.ndarray
        Averaged grid
    """
    return np.nanmean(grids, axis=0)


def save_merged_grid(
    out_dir: Path,
    avg_grid: np.ndarray,
    template_attrs_file: Path,
    output_name: str = "threshold_results_merged_average.h5"
) -> Path:
    """
    Save averaged grid to HDF5, using metadata from template file.
    
    Parameters
    ----------
    out_dir : Path
        Directory to save output file
    avg_grid : np.ndarray
        Averaged grid to save
    template_attrs_file : Path
        Template file containing axis metadata
    output_name : str, default="threshold_results_merged_average.h5"
        Name of output file
        
    Returns
    -------
    Path
        Path to saved file
    """
    save_path = out_dir / output_name
    with h5py.File(template_attrs_file, "r") as tmpl, \
         h5py.File(save_path, "w") as f:
        f.create_dataset("grid", data=avg_grid)
        # Copy axis attribute metadata so plotting code has labels
        f.attrs.update({
            "nOdor_values": list(tmpl.attrs["nOdor_values"]),
            "nSens_values": list(tmpl.attrs["nSens_values"]),
        })
    
    _log.info(f"Saved merged grid to {save_path}")
    return save_path 