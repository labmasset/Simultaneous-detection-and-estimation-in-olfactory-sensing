"""
Analysis and plotting utilities for present sweep experiments.

This module provides functions for analyzing and visualizing the results
of present sweep experiments, including L1 error curves, accuracy heatmaps, 
and other metrics.
"""

import logging
from typing import List, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from mirrored_langevin_rnn.utils.visualization.styling import add_vertical_markers, hide_labels

__all__ = [
    "compute_L1_curve",
    "plot_L1_curves",
    "compute_accuracy_heatmap",
    "plot_accuracy_heatmap",
    "compute_rankacc_heatmap",
    "compute_binarized_auc_heatmap",
    "compute_auc_heatmap",
    "compute_rank_accuracy_curve",
    "compute_auc_curve",
    "compute_binarized_presence_auc_curve",
    "plot_presence_assessed_curves",
]





def compute_L1_curve(
    h5_path: Union[str, Path],
    snapshot_idx: int,
    *,
    true_conc: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean ± std L1-error across repetitions for each *n_high*.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing present sweep results
    snapshot_idx : int
        Time index to analyze
    true_conc : float, default=40.0
        Ground truth concentration value

    Returns
    -------
    n_high_arr : np.ndarray
        Array of n_high values
    mean_err : np.ndarray
        Mean L1 error for each n_high
    sem_err : np.ndarray
        1.96 * Standard error of L1 error for each n_high (95% CI)
    """
    from collections import defaultdict
    import h5py
    from .data_io import _get_n_high

    # Aggregate runs by n_high
    def _aggregate_by_n_high(h5_path):
        with h5py.File(h5_path, "r") as f:
            by_high = defaultdict(list)
            for g in f["runs"].values():
                n_high = _get_n_high(g)
                C = np.asarray(g["C"])
                by_high[n_high].append(C)
        n_high_vals = np.array(sorted(by_high))
        return n_high_vals, by_high

    # Get data grouped by n_high
    n_high_arr, groups = _aggregate_by_n_high(h5_path)

    # Initialize output arrays
    mean_err = np.zeros_like(n_high_arr, dtype=float)
    sem_err = np.zeros_like(n_high_arr, dtype=float)

    # Calculate L1 error for each n_high
    for i, n_high in enumerate(n_high_arr):
        errs = []
        for C in groups[n_high]:
            if snapshot_idx >= C.shape[1]:
                raise IndexError(
                    f"snapshot_idx {snapshot_idx} exceeds time dimension {C.shape[1]}")
            est = C[:n_high, snapshot_idx]
            errs.append(np.mean(np.abs(est - true_conc)))
        errs = np.asarray(errs)
        mean_err[i] = errs.mean()
        sem_err[i] = 1.96 * errs.std(ddof=1) / np.sqrt(len(errs)) if len(errs) > 1 else 0.0

    return n_high_arr, mean_err, sem_err


def plot_L1_curves(
    h5_paths: List[Union[str, Path]],
    labels: List[str],
    *,
    snapshot_idx: int,
    true_conc: float = 40.0,
    show_std: bool = True,
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (6, 6),
):
    """
    Plot L1 error curves for multiple models backed by HDF5 files.

    Parameters
    ----------
    h5_paths : List[str or Path]
        Paths to HDF5 files to plot
    labels : List[str]
        Labels for each curve
    snapshot_idx : int
        Time index to analyze
    true_conc : float, default=40.0
        Ground truth concentration value
    show_std : bool, default=True
        Whether to show standard deviation bands
    colors : List[str], optional
        Colors for each curve
    figsize : Tuple[float, float], default=(6, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each curve with std shading
    for path, label, color in zip(h5_paths, labels, colors):
        n_high, mu, sigma = compute_L1_curve(
            path, snapshot_idx, true_conc=true_conc)
        if show_std:
            ax.fill_between(n_high, mu - sigma, mu + sigma,
                            color=color, alpha=0.25, linewidth=0)
        ax.plot(n_high, mu, marker="None", linewidth=2,
                color=color, label=label)

    # Style the plot
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2)

    # Configure ticks
    ax.tick_params(
        which='both',
        direction='in',
        length=16,
        width=2,
        labelsize=24,
        pad=10
    )

    # Set y-limit
    ax.set_ylim([0, 50])
    ax.set_xlim([1, 100])
    # Set ticks
    ax.set_xticks([1, 25, 50, 75, 100])
    ax.set_xticklabels(["1", "", "50", "", "100"])
    ax.set_yticks([0, 12.5, 25, 37.5, 50])
    ax.set_yticklabels(["0", "", "25", "", "50"])
    # normalize aspect ratio by x / y
    ax.set_aspect(n_high.shape[0] / 50)

    fig.tight_layout()
    return fig, ax


def compute_accuracy_heatmap(
    C_all: np.ndarray,
    c_truth: float,
    *,
    tolerance: float = 10.0,
    t_start: int = 0,
    n_timepoints: int = 75,
) -> np.ndarray:
    """
    Compute accuracy heatmap showing proportion of correct concentration estimates.

    Parameters
    ----------
    C_all : np.ndarray
        4D array of concentration estimates (n_odor, n_time, n_high, n_rep)
    c_truth : float
        Ground truth concentration value
    tolerance : float, default=10.0
        Acceptable deviation from c_truth
    t_start : int, default=0
        Starting time index
    n_timepoints : int, default=75
        Number of time points to include

    Returns
    -------
    np.ndarray
        Heatmap of accuracy values (n_high, n_timepoints)
    """
    n_odor, n_t, n_high_max, n_rep = C_all.shape
    if t_start + n_timepoints > n_t:
        raise ValueError("Requested time window exceeds C_all length")

    heatmap = np.zeros((n_high_max, n_timepoints), dtype=np.float64)
    t_slice = slice(t_start, t_start + n_timepoints)

    for k in range(1, n_high_max + 1):
        est = C_all[:k, t_slice, k - 1, :]
        correct = np.abs(est - c_truth) <= tolerance
        num_correct_per_t = correct.sum(axis=(0, 2))
        total = k * n_rep
        heatmap[k - 1] = num_correct_per_t / total

    return heatmap


def compute_l1_heatmap(
    C_all: np.ndarray,
    c_truth: float,
    *,
    tolerance: float = 10.0,
    t_start: int = 0,
    n_timepoints: int = 75,
    type: str = "concentration_l1"
) -> np.ndarray:

    n_odor, n_t, n_high_max, n_rep = C_all.shape
    if t_start + n_timepoints > n_t:
        raise ValueError("Requested time window exceeds C_all length")

    heatmap = np.zeros((n_high_max, n_timepoints), dtype=np.float64)
    t_slice = slice(t_start, t_start + n_timepoints)

    for k in range(1, n_high_max + 1):
        est = C_all[:k, t_slice, k - 1, :]
        l1_error = np.abs(est - c_truth)
        l1_error_mean = np.mean(l1_error, axis=2)
        l1_error_mean = np.mean(l1_error_mean, axis=0)
        heatmap[k - 1] = l1_error_mean

    return heatmap


def plot_accuracy_heatmap(
    heatmap: np.ndarray,
    *,
    sigma: float = 1,
    contour_levels: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    title: Optional[str] = None,
    t_start: int = 0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "Blues",
    figsize: Tuple[float, float] = (7, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    highlight_level: float = 0.5,
    ver_markers: Optional[List[float]] = None,
    aspect_ratio: float = 1
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot accuracy heatmap with contour lines.

    Parameters
    ----------
    heatmap : np.ndarray
        Heatmap data (n_high, n_timepoints)
    sigma : float, default=1
        Gaussian smoothing sigma
    contour_levels : Tuple[float, ...], default=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        Contour levels
    title : str, optional
        Plot title
    t_start : int, default=0
        Starting time index
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    cmap : str, default="Blues"
        Colormap
    figsize : Tuple[float, float], default=(6, 6)
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    n_high_levels, n_timepoints = heatmap.shape
    time_axis = np.arange(t_start, t_start + n_timepoints)
    odor_axis = np.arange(1, n_high_levels + 1)

    X, Y = np.meshgrid(time_axis, odor_axis)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    smoothed = heatmap

    # Main heatmap
    if vmin and vmax:
        mesh = ax.pcolormesh(X, Y, smoothed, shading='nearest',
                             cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # mesh = ax.pcolormesh(X, Y, heatmap, shading='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        mesh = ax.pcolormesh(X, Y, smoothed, shading='nearest', cmap=cmap)
    ax.set_xlim(time_axis[0], time_axis[-1])

    smoothed = gaussian_filter(heatmap, sigma=sigma)

    if ver_markers is not None:
        add_vertical_markers(ax, ver_markers,
                             color="#E13960", linestyle='--',
                             linewidth=4, alpha=1)
    # Contour lines
    other_levels = [lev for lev in contour_levels if lev != highlight_level]
    contours = ax.contour(
        X, Y, smoothed, levels=other_levels, colors="k", linewidths=3)
    contour_yellow = ax.contour(X, Y, smoothed, levels=[
                                highlight_level], colors="gold", linewidths=3)

    # Style axes
    ax.set_ylim([1, 100])
    ax.set_yticks([1, 25, 50, 75, 100])
    ax.set_yticklabels(["1", "", "50", "", "100"])
    ax.set_xlim([1, 75])
    ax.set_xticks([1, 18.25, 37.5, 55.25, 75])
    ax.set_xticklabels(["1", "", "375", "", "750"])
    ax.set_aspect(n_timepoints / n_high_levels * aspect_ratio)

    # Style spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    # Labels if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=26, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=26, fontweight="bold")

    # Tick styling
    ax.tick_params(
        which='both',
        direction='in',
        length=16,
        width=2,
        labelsize=24,
        pad=10
    )
    fig.tight_layout()
    return fig, ax


def plot_concentration_heatmap(
    heatmap: np.ndarray,
    heatmap_contour: np.ndarray = None,
    *,
    sigma: float = 0,
    contour_levels: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    title: Optional[str] = None,
    t_start: int = 0,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: str = "Blues",
    figsize: Tuple[float, float] = (7, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    highlight_level: float = 0.8,
    ver_markers: Optional[np.ndarray] = None,
    aspect_ratio: float = 1
) -> Tuple[plt.Figure, plt.Axes]:
    n_high_levels, n_timepoints = heatmap.shape
    time_axis = np.arange(t_start, t_start + n_timepoints)
    odor_axis = np.arange(1, n_high_levels + 1)

    X, Y = np.meshgrid(time_axis, odor_axis)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    smoothed = heatmap

    norm = None
    if (vmin is not None) and (vmax is not None):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(
        X, Y, smoothed,
        shading='nearest',
        cmap=cmap,
        norm=norm,
        edgecolors='face',     # avoids contrasting seams
        linewidth=0,           # no strokes
        antialiased=False      # reduces sub-pixel gaps in viewers
    )
    ax.set_xlim(time_axis[0], time_axis[-1])

    if ver_markers is not None:
        add_vertical_markers(ax, ver_markers,
                             color="#E13960", linestyle='--',
                             linewidth=4, alpha=1)

    if heatmap_contour is not None:
        smoothed = gaussian_filter(heatmap_contour, sigma=sigma)

        # Contour lines
        other_levels = [
            lev for lev in contour_levels if lev != highlight_level]
        contours = ax.contour(
            X, Y, smoothed, levels=other_levels, colors="k", linewidths=2)
        contour_yellow = ax.contour(X, Y, smoothed, levels=[highlight_level],
                                    colors="gold", linewidths=3)

    # Style axes
    ax.set_ylim([1, 100])
    ax.set_yticks([1, 25, 50, 75, 100])
    ax.set_yticklabels(["1", "", "50", "", "100"])
    ax.set_xlim([1, 75])
    ax.set_xticks([1, 18.25, 37.5, 55.25, 75])
    ax.set_xticklabels(["1", "", "375", "", "750"])
    ax.set_aspect(n_timepoints / n_high_levels * aspect_ratio)

    # Style spines
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    # Labels if provided
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=26, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=26, fontweight="bold")

    # Tick styling
    ax.tick_params(
        which='both',
        direction='in',
        length=16,
        width=2,
        labelsize=24,
        pad=10
    )
    fig.tight_layout()
    return fig, ax


def compute_rankacc_heatmap(
    C_all: np.ndarray,
    *,
    t_start: int = 0,
    n_timepoints: int = 40,
) -> np.ndarray:
    """
    Compute rank-based accuracy heatmap.

    Heat-map[k-1, t] = fraction of repetitions for which *all* of the
    first k odors are among the top-k ranked concentrations at time t.

    Parameters
    ----------
    C_all : np.ndarray
        4D array of concentration estimates (n_odor, n_time, n_high, n_rep)
    t_start : int, default=0
        Starting time index
    n_timepoints : int, default=40
        Number of time points to include

    Returns
    -------
    np.ndarray
        Heatmap of rank accuracy values (n_high, n_timepoints)
    """
    n_odor, n_t, n_high_max, n_rep = C_all.shape
    if t_start + n_timepoints > n_t:
        raise ValueError("time window exceeds data length")

    heatmap = np.zeros((n_high_max, n_timepoints), dtype=np.float64)
    t_slice = slice(t_start, t_start + n_timepoints)

    for k in range(1, n_high_max + 1):
        present = np.arange(k)   # the "true" odors
        # shape: (n_odor, n_timepoints, n_rep)
        est = C_all[:, t_slice, k - 1, :]

        for ti in range(n_timepoints):
            corrects = []
            for r in range(n_rep):
                last = est[:, ti, r]               # length = n_odor
                ranks = rankdata(-last, method="average")
                top_k = np.argsort(ranks)[:k]      # indices of highest k
                # only count as correct if *all* present odors are found
                correct = np.all(np.isin(present, top_k))
                corrects.append(float(correct))
            heatmap[k - 1, ti] = float(np.nanmean(corrects))

    return heatmap

def compute_binarized_auc_heatmap(
    C_all: np.ndarray,
    *,
    t_start: int = 0,
    n_timepoints: int = 40,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Compute ROC AUC heatmap from binarized presence derived from concentrations.

    For each k = 1..n_high_max and time t in the window:
      - If threshold is None:
          Binarize by rank: set top-k concentrations to 1, others 0.
      - Else:
          Binarize by threshold: set 1 where C >= threshold, else 0.
      - Compare with ground-truth presence (first k odors) using ROC AUC,
        averaged across repetitions.

    Parameters
    ----------
    C_all : np.ndarray
        4D array of concentration estimates (n_odor, n_time, n_high, n_rep)
    t_start : int, default=0
        Starting time index
    n_timepoints : int, default=40
        Number of time points to include
    threshold : float, optional
        If provided, binarize by C >= threshold; otherwise use top-k by rank.

    Returns
    -------
    np.ndarray
        Heatmap of AUC values (n_high, n_timepoints). May contain NaNs if AUC
        is undefined for a cell (e.g., all positives or all negatives).
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata

    n_odor, n_t, n_high_max, n_rep = C_all.shape
    if t_start + n_timepoints > n_t:
        raise ValueError("time window exceeds data length")

    heatmap = np.full((n_high_max, n_timepoints), np.nan, dtype=np.float64)
    t_slice = slice(t_start, t_start + n_timepoints)

    for k in range(1, n_high_max + 1):
        present_idx = np.arange(k)
        # est shape: (n_odor, n_timepoints, n_rep)
        est = C_all[:, t_slice, k - 1, :]

        # Ground-truth labels: first k odors are present
        y_true = np.zeros(n_odor, dtype=int)
        y_true[:k] = 1

        for ti in range(n_timepoints):
            aucs = []
            for r in range(n_rep):
                last = est[:, ti, r]

                if threshold is None:
                    # Rank-based binarization: top-k -> 1
                    ranks = rankdata(-last, method="average")  # highest -> rank 1
                    top_k = np.argsort(ranks)[:k]
                    y_score = np.zeros(n_odor, dtype=int)
                    y_score[top_k] = 1
                else:
                    # Threshold-based binarization
                    y_score = (last >= threshold).astype(int)

                try:
                    auc = roc_auc_score(y_true, y_score)
                    aucs.append(auc)
                except ValueError:
                    # Happens if y_true has one class or y_score degenerate in a way sklearn rejects
                    # (e.g., k == 0 or k == n_odor). Skip this repetition.
                    continue

            if len(aucs) > 0:
                heatmap[k - 1, ti] = float(np.nanmean(aucs))

    return heatmap

def compute_auc_heatmap(
    theta_all: np.ndarray,
    *,
    t_start: int = 0,
    n_timepoints: int = 40,
) -> np.ndarray:
    """
    Compute ROC-AUC heatmap for theta values.

    Heat-map entry (k-1, t) = mean ROC-AUC over reps for n_high=k
    at time index t_start + t.

    Parameters
    ----------
    theta_all : np.ndarray
        4D array of theta values (n_odor, n_time, n_high, n_rep)
    t_start : int, default=0
        Starting time index
    n_timepoints : int, default=40
        Number of time points to include

    Returns
    -------
    np.ndarray
        Heatmap of AUC values (n_high, n_timepoints)
    """
    n_odor, n_t, n_high_max, n_rep = theta_all.shape
    if t_start + n_timepoints > n_t:
        raise ValueError("time window exceeds data length")

    heatmap = np.zeros((n_high_max, n_timepoints), dtype=np.float64)
    t_slice = slice(t_start, t_start + n_timepoints)

    for k in range(1, n_high_max + 1):
        labels = np.zeros(n_odor, dtype=bool)
        labels[:k] = True
        est = theta_all[:, t_slice, k - 1, :]            # n_odor × time × rep
        for ti, t in enumerate(range(*t_slice.indices(n_t))):
            aucs = []
            for r in range(n_rep):
                scores = est[:, ti, r]
                # sklearn will error if all scores identical or only one label
                try:
                    auc = roc_auc_score(labels, scores)
                except ValueError:
                    auc = np.nan
                aucs.append(auc)
            heatmap[k - 1, ti] = np.nanmean(aucs)
    return heatmap


def compute_rank_accuracy_curve(
    h5_path: Union[str, Path],
    snapshot_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rank accuracy curve for Poisson model at a specific timepoint.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing Poisson model results
    snapshot_idx : int
        Time index to analyze

    Returns
    -------
    n_high_arr : np.ndarray
        Array of n_high values
    mean_acc : np.ndarray
        Mean rank accuracy for each n_high
    std_acc : np.ndarray
        Standard deviation of rank accuracy for each n_high
    """
    import h5py
    from collections import defaultdict
    from .data_io import _get_n_high

    def _aggregate_by_n_high(h5_path):
        with h5py.File(h5_path, "r") as f:
            by_high = defaultdict(list)
            for g in f["runs"].values():
                n_high = _get_n_high(g)
                C = np.asarray(g["C"])
                by_high[n_high].append(C)
        n_high_vals = np.array(sorted(by_high))
        return n_high_vals, by_high

    n_high_vals, by_high = _aggregate_by_n_high(h5_path)
    n_high_arr = np.array(sorted(by_high.keys()))
    mean_acc = np.zeros(len(n_high_arr))
    std_acc = np.zeros(len(n_high_arr))

    for i, n_high in enumerate(n_high_arr):
        accuracies = []
        for C in by_high[n_high]:
            if snapshot_idx >= C.shape[1]:
                continue

            # Get concentration estimates at the specified timepoint
            est = C[:, snapshot_idx]  # shape: (n_odor,)

            # Compute rank-based accuracy
            present = np.arange(n_high)  # true odor indices
            ranks = rankdata(-est, method="average")  # highest → rank 1
            top_k = np.argsort(ranks)[:n_high]  # indices of top k

            # Check if all present odors are in top k
            correct = np.all(np.isin(present, top_k))
            accuracies.append(float(correct))

        if accuracies:
            accuracies = np.asarray(accuracies)
            mean_acc[i] = accuracies.mean()
            std_acc[i] = accuracies.std(ddof=1) if len(accuracies) > 1 else 0.0

    return n_high_arr, mean_acc, std_acc


def compute_auc_curve(
    h5_path: Union[str, Path],
    snapshot_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute AUC curve for SLAM model at a specific timepoint.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing SLAM model results (with theta values)
    snapshot_idx : int
        Time index to analyze

    Returns
    -------
    n_high_arr : np.ndarray
        Array of n_high values
    mean_auc : np.ndarray
        Mean AUC for each n_high
    sem_auc : np.ndarray
        1.96 * Standard error of AUC for each n_high
    """
    import h5py
    from collections import defaultdict
    from .data_io import _get_n_high

    def _aggregate_by_n_high(h5_path):
        with h5py.File(h5_path, "r") as f:
            by_high = defaultdict(list)
            for g in f["runs"].values():
                n_high = _get_n_high(g)
                theta = np.asarray(g["Theta"])
                by_high[n_high].append(theta)
        n_high_vals = np.array(sorted(by_high))
        return n_high_vals, by_high

    n_high_vals, by_high = _aggregate_by_n_high(h5_path)
    n_high_arr = np.array(sorted(by_high.keys()))
    mean_auc = np.zeros(len(n_high_arr))
    sem_auc = np.zeros(len(n_high_arr))

    for i, n_high in enumerate(n_high_arr):
        aucs = []
        for theta in by_high[n_high]:
            if snapshot_idx >= theta.shape[1]:
                continue

            # Get theta values at the specified timepoint
            scores = theta[:, snapshot_idx]  # shape: (n_odor,)

            # Create binary labels (first n_high are true)
            n_odor = len(scores)
            labels = np.zeros(n_odor, dtype=bool)
            labels[:n_high] = True

            # Compute AUC
            try:
                auc = roc_auc_score(labels, scores)
                aucs.append(auc)
            except ValueError:
                # Skip if AUC cannot be computed (e.g., all scores identical)
                continue

        if aucs:
            aucs = np.asarray(aucs)
            mean_auc[i] = aucs.mean()
            sem_auc[i] = 1.96 * aucs.std(ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0.0

    return n_high_arr, mean_auc, sem_auc

def compute_binarized_presence_auc_curve(
    h5_path: Union[str, Path],
    snapshot_idx: int,
    threshold: Optional[float] = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC AUC curve from a binarized presence estimate derived from concentrations.

    For each run at the given timepoint:
      - If threshold is None:
          Rank odors by estimated concentration and set the top-n_present to 1, others to 0.
      - Else:
          Set 1 for odors with concentration >= threshold, else 0.

    The binarized vector is compared to the ground-truth presence (first n_present odors).

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file containing model results with concentration estimates `C`
    snapshot_idx : int
        Time index to analyze
    threshold : float, optional
        If provided, binarize by `C >= threshold`; otherwise use top-n_present by rank

    Returns
    -------
    n_high_arr : np.ndarray
        Array of n_present values
    mean_auc : np.ndarray
        Mean AUC for each n_present
    std_auc : np.ndarray
        Standard deviation of AUC for each n_present
    """
    import h5py
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score
    from scipy.stats import rankdata
    from .data_io import _get_n_high

    def _aggregate_by_n_high(h5_path):
        with h5py.File(h5_path, "r") as f:
            by_high = defaultdict(list)
            for g in f["runs"].values():
                n_high = _get_n_high(g)
                C = np.asarray(g["C"])
                by_high[n_high].append(C)
        n_high_vals = np.array(sorted(by_high))
        return n_high_vals, by_high

    n_high_vals, by_high = _aggregate_by_n_high(h5_path)
    n_high_arr = np.array(sorted(by_high.keys()))
    mean_auc = np.zeros(len(n_high_arr))
    std_auc = np.zeros(len(n_high_arr))

    for i, n_high in enumerate(n_high_arr):
        aucs = []
        for C in by_high[n_high]:
            if snapshot_idx >= C.shape[1]:
                continue

            est = C[:, snapshot_idx]  # shape: (n_odor,)

            if threshold is None:
                # Rank-based binarization: top-n_high -> 1, else 0
                ranks = rankdata(-est, method="average")  # highest -> rank 1
                top_k = np.argsort(ranks)[:n_high]
                y_score = np.zeros_like(est, dtype=int)
                y_score[top_k] = 1
            else:
                # Threshold-based binarization
                y_score = (est >= threshold).astype(int)

            # Ground truth: first n_high odors are present
            n_odor = len(est)
            y_true = np.zeros(n_odor, dtype=int)
            y_true[:n_high] = 1

            try:
                auc = roc_auc_score(y_true, y_score)
                aucs.append(auc)
            except ValueError:
                # Raised if y_true has only one class (shouldn't happen if 0 < n_high < n_odor)
                continue

        if aucs:
            aucs = np.asarray(aucs)
            mean_auc[i] = aucs.mean()
            std_auc[i] = aucs.std(ddof=1) if len(aucs) > 1 else 0.0

    return n_high_arr, mean_auc, std_auc

def plot_presence_assessed_curves(
    model_data_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    labels: List[str],
    timepoint: int,
    *,
    colors: Optional[List[str]] = None,
    style: Optional[object] = None,
    figsize: Tuple[float, float] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    show_std: bool = True,
    ylim: Tuple[float, float] = (0.5, 1.0),
    y_ticks: List[float] = [0.5, 0.625, 0.75, 0.875, 1.0],
    y_ticklabels: List[str] = ["0.5", "", "0.75", "", "1.0"],
    xlim: Tuple[float, float] = (1, 100),
    x_ticks: List[int] = [1, 25, 50, 75, 100],
    x_ticklabels: List[str] = ["1", "", "50", "", "100"],
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot model comparison at a specific timepoint for multiple models.

    Parameters
    ----------
    model_data_list : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of (n_high_arr, mean_values, std_values) tuples for each model
    labels : List[str]
        Labels for each model
    timepoint : int
        The timepoint index being compared
    colors : List[str], optional
        Colors for each model curve. If None, uses default color cycle
    style : PlotStyle, optional
        Plot styling configuration
    figsize : tuple, default=(8, 6)
        Figure size (width, height)
    save_path : str or Path, optional
        If provided, save figure to this path
    show_std : bool, default=True
        Whether to show standard deviation bands

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    from ..visualization import apply_style

    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if len(model_data_list) != len(labels):
        raise ValueError(
            "Number of model datasets must match number of labels")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each model
    for i, (data, label) in enumerate(zip(model_data_list, labels)):
        n_high_arr, mean_values, std_values = data
        color = colors[i % len(colors)]

        # Plot with error bands if requested
        if show_std:
            ax.fill_between(n_high_arr, mean_values - std_values,
                            mean_values + std_values, color=color, alpha=0.25, linewidth=0)

        ax.plot(n_high_arr, mean_values, marker="None", linewidth=2,
                color=color, label=label)

    # Apply styling
    if style is not None:
        apply_style(ax, style)

    # Style the plot similar to plot_L1_curves_h5
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(2)

    # Configure ticks
    ax.tick_params(
        which='both',
        direction='in',
        length=16,
        width=2,
        labelsize=24,
        pad=10
    )

    ax.set_xlabel("Number of Odours Present")
    ax.set_ylabel("Performance")
    ax.set_title(f"Model Comparison at Time t={timepoint}")

    # Set y-axis limits, ticks, and labels
    ax.set_ylim(ylim)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)

    # Set x-axis ticks and labels
    ax.set_xlim(xlim)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Add legend
    ax.legend()
    ax.set_aspect(n_high_arr.shape[0] / (ylim[1] - ylim[0]))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")

    return fig, ax

