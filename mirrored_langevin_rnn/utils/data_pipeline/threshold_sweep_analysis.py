"""
Analysis and visualization utilities for threshold sweep experiments.

This module provides functions for analyzing threshold sweep data,
including grid averaging, heatmap generation and visualization with
contour lines.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

__all__ = [
    "load_threshold_batch_files",
    "plot_threshold_heatmap",
    "merge_threshold_batches",
    "compute_average_grid"
]


def load_threshold_batch_files(
    out_dir: Path,
    pattern: str = "threshold_results_batch*.h5",
) -> Tuple[
    np.ndarray,          # shape (n_batch, nSens_all, nOdor_all)
    List[Path],          # the files, in sorted order
    Sequence[int],       # full nSens axis
    Sequence[int],       # full nOdor axis
]:
    """
    Read every batch file in *out_dir* matching *pattern* and align them
    into a common grid.

    Parameters
    ----------
    out_dir : Path
        Directory containing batch files
    pattern : str, default="threshold_results_batch*.h5"
        Glob pattern to match batch files

    Returns
    -------
    grids : np.ndarray
        Stacked array of all grids (n_batch, nSens_all, nOdor_all)
    files : List[Path]
        Sorted list of files that were loaded
    nSens_all : Sequence[int]
        The sorted union of every file's nSens_values
    nOdor_all : Sequence[int]
        The sorted union of every file's nOdor_values
    """
    files = sorted(out_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No batch files found in {out_dir!r}")

    # 1) collect each file's local axes and raw grid
    raws = []
    sens_lists = []
    odor_lists = []
    for p in files:
        with h5py.File(p, "r") as f:
            grid = f["grid"][:]
            local_sens = list(f.attrs["num_osn_values"])
            local_odor = list(f.attrs["num_potential_odors_values"])
        raws.append(grid)
        sens_lists.append(local_sens)
        odor_lists.append(local_odor)

    # 2) form the global parameter axes
    nSens_all = tuple(sorted(set().union(*sens_lists)))
    nOdor_all = tuple(sorted(set().union(*odor_lists)))
    S = len(nSens_all)
    O = len(nOdor_all)

    padded_grids = []
    # 3) for each file, build an S×O array and insert its data
    for p, grid, local_sens, local_odor in zip(files, raws, sens_lists, odor_lists):
        G = np.full((S, O), np.nan, dtype=grid.dtype)

        # find where each local row/col sits in the global axes
        row_idx = [nSens_all.index(s) for s in local_sens]
        col_idx = [nOdor_all.index(o) for o in local_odor]

        # broadcast-safe insert
        G[np.ix_(row_idx, col_idx)] = grid

        padded_grids.append(G)

    # 4) stack into (n_batch, S, O)
    grids = np.stack(padded_grids, axis=0)
    return grids, files, nSens_all, nOdor_all


def plot_threshold_heatmap(
    avg_grid: np.ndarray,
    n_odor_values: List[int],
    n_sens_values: List[int],
    *,
    ax=None,
    cmap: str = "Blues",
    sigma: float = 1,
    contour_levels: Tuple[float, ...] = (10, 20, 30),
    highlight_level: float = 20,
    annot: bool = False,
    figsize: Tuple[float, float] = (6, 6),
    highlight_color: str = "darkorange",
):
    """
    Draw a heat-map with contour overlays.

    Parameters
    ----------
    avg_grid : 2-D ndarray
        Values to visualize (nSens, nOdor)
    n_odor_values : list of int or float
        X-axis tick values (dictionary sizes, nOdor)
    n_sens_values : list of int or float
        Y-axis tick values (sensor counts, nSens)
    ax : matplotlib.axes.Axes, optional
        Destination axes; a new one is created if omitted
    cmap : str, default="Blues"
        Matplotlib colormap for pcolormesh
    sigma : float, default=1
        Gaussian smoothing parameter (0 for no smoothing)
    contour_levels : tuple of float, default=(10, 20, 30)
        Levels for contour lines
    highlight_level : float, default=20
        Level to highlight with yellow contour
    annot : bool, default=False
        Whether to annotate cells with their values
    figsize : tuple of float, default=(6, 6)
        Figure size when creating a new figure

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    # Setup
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Build a mesh for pcolormesh
    X, Y = np.meshgrid(n_odor_values, n_sens_values)

    # Get a copy of the colormap and set 'bad' color for NaNs
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    
    cmap_obj.set_bad(color='magenta')

    # plot the original grid without smoothing
    mesh = ax.pcolormesh(
        X,
        Y,
        avg_grid,
        shading="auto",
        cmap=cmap_obj,  # Use the modified colormap object
        norm=LogNorm(),  # Emphasizes multiplicative differences
    )
    
    # compute the contour levels with smoothed grid
    if sigma > 0:
        smoothed_grid = gaussian_filter(avg_grid, sigma=sigma)
    else:
        smoothed_grid = avg_grid.copy()

    # Draw normal contour lines (black)
    normal_lvls = [lvl for lvl in contour_levels if abs(lvl - highlight_level) > 1e-6]
    if normal_lvls:
        cs = ax.contour(
            X,
            Y,
            smoothed_grid,
            levels=normal_lvls,
            colors="k",
            linewidths=2,
        )

    # Draw highlighted contour (yellow)
    cs_hi = ax.contour(
        X,
        Y,
        smoothed_grid,
        levels=[highlight_level],
        colors=highlight_color,  # Use a darker yellow for visibility
        linewidths=3,
    )
    ax.clabel(cs_hi, inline=True, fontsize=22, fmt="%g")
    
    # Axis styling
    ax.set_xscale("log")  # Dictionary size is usually log‑scaled
    
    # Thicker bottom/left spines, hide top/right
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    # Ticks styling
    ax.tick_params(
        direction='in',
        length=16,
        width=2,
        labelsize=24,
        pad=10
    )
    ax.minorticks_off()
    
    # Custom ticks
    ax.set_xticks([1000, 2000, 4000, 8000, 16000])
    ax.set_xlim([1000, 16000])
    ax.set_ylim(100, 800)  # Y-axis limits based on nSens_values
    ax.set_xticklabels(["1000", "", "4000", "", "16000"])
    ax.set_yticks([100, 275, 450, 625, 800])
    ax.set_yticklabels(["100", "", "450", "", "800"])

    # Optional annotations
    if annot:
        for (i, j), val in np.ndenumerate(smoothed_grid):
            ax.text(
                n_odor_values[j],
                n_sens_values[i],
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    # fig.tight_layout()
    return fig, ax


def merge_threshold_batches(
    out_dir: Path, 
    pattern: str = "threshold_results_batch*.h5",
    output_file: str = "threshold_results_merged.h5",
    merge_method: str = "first_valid"
) -> Tuple[np.ndarray, Path]:
    """
    Merge multiple threshold grid batch files into a single complete grid.
    
    This function takes partial grids from batch files and combines them into a single
    complete grid. Each batch file may contain a different subset of the parameter space.
    
    Parameters
    ----------
    out_dir : Path
        Directory containing batch files
    pattern : str, default="threshold_results_batch*.h5"
        Glob pattern to match batch files
    output_file : str, default="threshold_results_merged.h5"
        Output filename
    merge_method : str, default="first_valid"
        Method to use when merging overlapping values:
        - "first_valid": Take the first non-NaN value encountered
        - "max": Take the maximum value
        - "min": Take the minimum value
        - "mean": Take the mean of all non-NaN values
        
    Returns
    -------
    merged_grid : np.ndarray
        The merged grid
    output_path : Path
        Path to the saved file
    """
    # Load all batch files and align them to a common grid
    grids, files, n_sens_all, n_odor_all = load_threshold_batch_files(out_dir, pattern)
    
    # Create a merged grid based on the specified method
    if merge_method == "first_valid":
        # Initialize with NaNs
        merged_grid = np.full((len(n_sens_all), len(n_odor_all)), np.nan, dtype=np.float32)
        
        # For each position, take the first non-NaN value encountered across all batches
        for i in range(len(n_sens_all)):
            for j in range(len(n_odor_all)):
                valid_values = grids[:, i, j][~np.isnan(grids[:, i, j])]
                if len(valid_values) > 0:
                    merged_grid[i, j] = valid_values[0]
    
    elif merge_method == "max":
        # Take the maximum value at each position (ignoring NaNs)
        merged_grid = np.nanmax(grids, axis=0)
    
    elif merge_method == "min":
        # Take the minimum value at each position (ignoring NaNs)
        merged_grid = np.nanmin(grids, axis=0)
    
    elif merge_method == "mean":
        # Take the mean of all non-NaN values at each position
        merged_grid = np.nanmean(grids, axis=0)
    
    else:
        raise ValueError(f"Unknown merge method: {merge_method}")
    
    # Save the merged grid
    output_path = out_dir / output_file
    with h5py.File(output_path, "w") as f:
        f.create_dataset("grid", data=merged_grid)
        f.attrs.update({
            "nOdor_values": list(n_odor_all),
            "nSens_values": list(n_sens_all),
        })
    
    return merged_grid, output_path


def compute_average_grid(
    out_dir: Path,
    pattern: str = "threshold_results_batch*.h5",
    output_file: str = "threshold_results_average.h5"
) -> Tuple[np.ndarray, Path]:
    """
    Compute the average grid from multiple threshold batch files.
    
    Parameters
    ----------
    out_dir : Path
        Directory containing batch files
    pattern : str, default="threshold_results_batch*.h5"
        Glob pattern to match batch files
    output_file : str, default="threshold_results_average.h5"
        Output filename
        
    Returns
    -------
    avg_grid : np.ndarray
        Averaged grid
    output_path : Path
        Path to the saved file
    """
    # Load all batch files
    grids, files, n_sens_all, n_odor_all = load_threshold_batch_files(out_dir, pattern)
    
    # Compute average (ignoring NaNs)
    avg_grid = np.nanmean(grids, axis=0)
    
    # Save the averaged grid
    output_path = out_dir / output_file
    with h5py.File(output_path, "w") as f:
        f.create_dataset("grid", data=avg_grid)
        f.attrs.update({
            "nOdor_values": list(n_odor_all),
            "nSens_values": list(n_sens_all),
        })
        
    return avg_grid, output_path 