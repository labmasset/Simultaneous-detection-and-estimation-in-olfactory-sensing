# accuracy_metrics.py
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def compute_accuracy_heatmap(
    C_all: np.ndarray,
    c_truth: float,
    *,
    tolerance: float = 10.0,
    t_start: int = 0,
    n_timepoints: int = 40
) -> np.ndarray:
    """
    Convert a (nOdor, nT, nHigh, repetition) tensor into an
    (nHigh, n_timepoints) heat-map of inference accuracy.

    The first *k* rows (odour indices 0..k-1) of *C_all*
    are assumed to be the high/contributing odours for the
    scenario with k high odours.

    Parameters
    ----------
    C_all : ndarray
        Shape (nOdor, nT, nHigh, repetition).
    c_truth : float
        Ground-truth concentration for each high odour.
    tolerance : float, optional
        Acceptable deviation (±) around *c_truth* for a
        prediction to count as correct. Default is 10.
    t_start : int, optional
        First MATLAB time index to score (Python index
        t_start - 1). Default 100 to match your earlier code.
    n_timepoints : int, optional
        Number of consecutive time points to score.
        Default 300, giving indices 100..399 (MATLAB)
        or 99..398 (Python).

    Returns
    -------
    heatmap : ndarray
        Shape (nHigh, n_timepoints) of mean accuracy
        averaged over repetitions.
    """
    n_odor, n_t, n_high_max, n_rep = C_all.shape
    logger.debug(
        "compute_accuracy_heatmap t_start=%d n_timepoints=%d n_t=%d n_high_max=%d n_rep=%d",
        t_start,
        n_timepoints,
        n_t,
        n_high_max,
        n_rep,
    )
    if t_start + n_timepoints > n_t:
        raise ValueError("Requested time window exceeds C_all length")

    # Pre‑allocate result
    heatmap = np.zeros((n_high_max, n_timepoints), dtype=np.float64)

    # Slice the time window once for efficiency
    t_slice = slice(t_start, t_start + n_timepoints)

    # Loop over each condition: k high odours present
    for k in range(1, n_high_max + 1):
        # Estimated concentrations for the *k* high odours
        # Dimensions: (k, n_timepoints, n_rep)
        est = C_all[:k, t_slice, k - 1, :]    # note k‑1 because 0‑based axis
        # Boolean mask of correct estimates (broadcasted)
        correct = np.abs(est - c_truth) <= tolerance
        # Count correct predictions across odours and reps
        num_correct_per_t = correct.sum(axis=(0, 2))
        # Total possible predictions at each t
        total = k * n_rep
        heatmap[k - 1] = num_correct_per_t / total

    return heatmap


def plot_accuracy_heatmap(
    heatmap: np.ndarray,
    *,
    sigma: float = 1,
    contour_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    title: str = "Mean Correct Fraction Heatmap",
    t_start: int = 0,
    xlabel: str = "Time after odor onset (ms)",
    ylabel: str = "Number of Odors Present",
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Draw the smoothed heat-map with black isolines and
    a yellow 0.5 contour, matching the previous style.

    Returns the matplotlib Figure so the caller can
    `.savefig()` or further customise if needed.
    """
    import matplotlib as mpl

    n_high_levels, n_timepoints = heatmap.shape
    time_axis = np.arange(t_start, t_start + n_timepoints)
    odor_axis = np.arange(1, n_high_levels + 1)

    X, Y = np.meshgrid(time_axis, odor_axis)

    smoothed = gaussian_filter(heatmap, sigma=sigma)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(X, Y, heatmap, shading="auto", cmap=cmap)
    cbar = fig.colorbar(mesh, ax=ax, label="Proportion Correct")
    cbar.ax.yaxis.label.set_size(26)
    cbar.ax.yaxis.label.set_weight("bold")
    cbar.ax.tick_params(labelsize=20)

    other_levels = [lev for lev in contour_levels if lev != 0.5]
    contours = ax.contour(X, Y, smoothed, levels=other_levels, colors="k")
    ax.clabel(contours, inline=True, fontsize=20)
    contour_yellow = ax.contour(X, Y, smoothed, levels=[0.5], colors="yellow")
    ax.clabel(contour_yellow, inline=True, fontsize=20, fmt="%1.1f")

    ax.set_xticks([0, 10, 20])
    ax.set_xticklabels(["0", "250", "500"], fontsize=26)
    # ax.set_yticks(odor_axis[:: max(len(odor_axis) // 10, 1)])
    ax.set_yticks([0,25, 50])
    ax.tick_params(axis="y", labelsize=26)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.set_xlabel(xlabel, fontsize=26, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=26, fontweight="bold")
    ax.set_title(title, fontsize=26, fontweight="bold")

    fig.tight_layout()
    return fig
