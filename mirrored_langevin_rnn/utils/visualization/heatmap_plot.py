"""
Heatmap plotting utilities for present sweep analysis.

This module provides clean, reusable functions for plotting heatmaps
with consistent styling and advanced features like vertical slices.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List, Union
from pathlib import Path

from .styling import PlotStyle, apply_style, add_vertical_markers, COLORS

__all__ = [
    "plot_heatmap_with_scale",
    "plot_vertical_slices", 
    "plot_individual_slices",
    "plot_comparative_heatmaps",
]


def plot_heatmap_with_scale(
    heatmap: np.ndarray,
    *,
    title: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vertical_markers: Optional[List[int]] = None,
    cmap: str = "Blues",
    sigma: float = 3,
    contour_levels: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
    style: Optional[PlotStyle] = None,
    figsize: Tuple[float, float] = (6, 6),
) -> Tuple[plt.Figure, Axes]:
    """
    Plot a heatmap with consistent styling and optional color scale.
    
    Parameters
    ----------
    heatmap : np.ndarray
        Heatmap data (n_high, n_timepoints)
    title : str
        Plot title
    vmin, vmax : float, optional
        Color scale limits. If None, uses data min/max
    vertical_markers : List[int], optional
        Time points to mark with vertical lines
    cmap : str, default="Blues"
        Colormap name
    sigma : float, default=3
        Gaussian smoothing parameter
    contour_levels : Tuple[float, ...] 
        Contour levels to draw
    style : PlotStyle, optional
        Styling configuration
    figsize : Tuple[float, float], default=(6, 6)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if style is None:
        style = PlotStyle()
    
    # Prepare data
    n_high_levels, n_timepoints = heatmap.shape
    time_axis = np.arange(0, n_timepoints)
    odor_axis = np.arange(1, n_high_levels + 1)
    
    X, Y = np.meshgrid(time_axis, odor_axis)
    smoothed = gaussian_filter(heatmap, sigma=sigma)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Main heatmap
    mesh = ax.pcolormesh(X, Y, heatmap, shading='nearest', cmap=cmap, 
                        vmin=vmin, vmax=vmax)
    ax.set_xlim(time_axis[0], time_axis[-1])
    
    # Contour lines
    other_levels = [lev for lev in contour_levels if lev != 0.5]
    if other_levels:
        ax.contour(X, Y, smoothed, levels=other_levels, colors="k", linewidths=3)
    ax.contour(X, Y, smoothed, levels=[0.5], colors="yellow", linewidths=2)
    
    # Apply consistent styling
    ax.set_ylim([0, 50])
    ax.set_xticks([0, 182.5, 375, 552.5, 750])
    ax.set_xticklabels(["0", "", "375", "", "750"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "", "50", "", "100"])
    
    # Set labels and title
    ax.set_xlabel("Time after odour onset (ms)", fontsize=style.axes_labelsize, fontweight="bold")
    ax.set_ylabel("Number of Odours Present", fontsize=style.axes_labelsize, fontweight="bold")
    ax.set_title(title, fontsize=style.axes_titlesize, fontweight="bold")
    
    # Apply general styling
    apply_style(ax, style)
    
    # Add vertical markers if specified
    if vertical_markers:
        add_vertical_markers(ax, vertical_markers)
    
    fig.tight_layout()
    return fig, ax


def plot_vertical_slices(
    heatmap1: np.ndarray, 
    heatmap2: Optional[np.ndarray],
    t1: int, 
    t2: int,
    labels: List[str],
    colors: List[str],
    *,
    title_suffix: str = "",
    style: Optional[PlotStyle] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[plt.Figure, Tuple[Axes, Axes]]:
    """
    Plot vertical slices of heatmaps at specified time points.
    
    Parameters
    ----------
    heatmap1 : np.ndarray
        First heatmap data (n_high, n_timepoints)
    heatmap2 : np.ndarray, optional
        Second heatmap data for comparison
    t1, t2 : int
        Time indices for vertical slices
    labels : List[str]
        Labels for the heatmaps
    colors : List[str]
        Colors for the plots
    title_suffix : str, default=""
        Additional text for plot titles
    style : PlotStyle, optional
        Styling configuration
    figsize : Tuple[float, float], default=(12, 5)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
    """
    if style is None:
        style = PlotStyle()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Number of odors present (y-axis values)
    n_odors = np.arange(1, heatmap1.shape[0] + 1)
    
    # Plot at first time point
    ax1.plot(n_odors, heatmap1[:, t1], 'o-', color=colors[0], 
             linewidth=style.line_width, markersize=style.marker_size, label=labels[0])
    if heatmap2 is not None:
        ax1.plot(n_odors, heatmap2[:, t1], 's-', color=colors[1], 
                 linewidth=style.line_width, markersize=style.marker_size, label=labels[1])
    
    ax1.set_xlabel('Number of Odours Present', fontsize=style.axes_labelsize)
    ax1.set_ylabel('Performance', fontsize=style.axes_labelsize)
    ax1.set_title(f'Performance at Early Time Point (t={t1}){title_suffix}', 
                  fontsize=style.axes_titlesize)
    ax1.grid(True, alpha=style.grid_alpha)
    ax1.legend(fontsize=style.legend_fontsize, frameon=style.legend_frame)
    ax1.set_ylim([0, 1])
    
    # Plot at second time point
    ax2.plot(n_odors, heatmap1[:, t2], 'o-', color=colors[0], 
             linewidth=style.line_width, markersize=style.marker_size, label=labels[0])
    if heatmap2 is not None:
        ax2.plot(n_odors, heatmap2[:, t2], 's-', color=colors[1], 
                 linewidth=style.line_width, markersize=style.marker_size, label=labels[1])
    
    ax2.set_xlabel('Number of Odours Present', fontsize=style.axes_labelsize)
    ax2.set_ylabel('Performance', fontsize=style.axes_labelsize)
    ax2.set_title(f'Performance at Late Time Point (t={t2}){title_suffix}', 
                  fontsize=style.axes_titlesize)
    ax2.grid(True, alpha=style.grid_alpha)
    ax2.legend(fontsize=style.legend_fontsize, frameon=style.legend_frame)
    ax2.set_ylim([0, 1])
    
    # Apply styling to both axes
    apply_style(ax1, style)
    apply_style(ax2, style)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_individual_slices(
    heatmap1: np.ndarray,
    heatmap2: Optional[np.ndarray],
    t1: int,
    t2: int,
    labels: List[str],
    colors: List[str],
    *,
    style: Optional[PlotStyle] = None,
    figsize: Tuple[float, float] = (12, 10),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot individual slices for each model separately in a 2x2 grid.
    
    Parameters
    ----------
    heatmap1 : np.ndarray
        First heatmap data (n_high, n_timepoints)
    heatmap2 : np.ndarray, optional
        Second heatmap data
    t1, t2 : int
        Time indices for slices
    labels : List[str]
        Labels for the heatmaps
    colors : List[str]
        Colors for the plots
    style : PlotStyle, optional
        Styling configuration
    figsize : Tuple[float, float], default=(12, 10)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
        2x2 array of axes
    """
    if style is None:
        style = PlotStyle()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    n_odors = np.arange(1, heatmap1.shape[0] + 1)
    
    # First heatmap slices
    axes[0,0].plot(n_odors, heatmap1[:, t1], 'o-', 
                   color=colors[0], linewidth=style.line_width, 
                   markersize=style.marker_size)
    axes[0,0].set_title(f'{labels[0]} - Early (t={t1})', fontsize=style.axes_titlesize)
    axes[0,0].set_ylabel(labels[0], fontsize=style.axes_labelsize)
    axes[0,0].grid(True, alpha=style.grid_alpha)
    axes[0,0].set_ylim([0, 1])
    
    axes[0,1].plot(n_odors, heatmap1[:, t2], 'o-', 
                   color=colors[0], linewidth=style.line_width, 
                   markersize=style.marker_size)
    axes[0,1].set_title(f'{labels[0]} - Late (t={t2})', fontsize=style.axes_titlesize)
    axes[0,1].set_ylabel(labels[0], fontsize=style.axes_labelsize)
    axes[0,1].grid(True, alpha=style.grid_alpha)
    axes[0,1].set_ylim([0, 1])
    
    if heatmap2 is not None:
        # Second heatmap slices
        axes[1,0].plot(n_odors, heatmap2[:, t1], 's-', 
                       color=colors[1], linewidth=style.line_width, 
                       markersize=style.marker_size)
        axes[1,0].set_title(f'{labels[1]} - Early (t={t1})', fontsize=style.axes_titlesize)
        axes[1,0].set_ylabel(labels[1], fontsize=style.axes_labelsize)
        axes[1,0].set_xlabel('Number of Odours Present', fontsize=style.axes_labelsize)
        axes[1,0].grid(True, alpha=style.grid_alpha)
        axes[1,0].set_ylim([0, 1])
        
        axes[1,1].plot(n_odors, heatmap2[:, t2], 's-', 
                       color=colors[1], linewidth=style.line_width, 
                       markersize=style.marker_size)
        axes[1,1].set_title(f'{labels[1]} - Late (t={t2})', fontsize=style.axes_titlesize)
        axes[1,1].set_ylabel(labels[1], fontsize=style.axes_labelsize)
        axes[1,1].set_xlabel('Number of Odours Present', fontsize=style.axes_labelsize)
        axes[1,1].grid(True, alpha=style.grid_alpha)
        axes[1,1].set_ylim([0, 1])
    else:
        # Hide bottom row if no second heatmap
        axes[1,0].set_visible(False)
        axes[1,1].set_visible(False)
    
    # Apply styling to all visible axes
    for ax in axes.flat:
        if ax.get_visible():
            apply_style(ax, style)
    
    plt.tight_layout()
    return fig, axes


def plot_comparative_heatmaps(
    heatmap1: np.ndarray,
    heatmap2: Optional[np.ndarray],
    titles: List[str],
    vertical_markers: Optional[List[int]] = None,
    *,
    style: Optional[PlotStyle] = None,
    figsize: Tuple[float, float] = (12, 5),
    **kwargs
) -> Tuple[plt.Figure, List[Axes]]:
    """
    Plot two heatmaps side by side with consistent color scale.
    
    Parameters
    ----------
    heatmap1 : np.ndarray
        First heatmap data
    heatmap2 : np.ndarray, optional
        Second heatmap data
    titles : List[str]
        Titles for each heatmap
    vertical_markers : List[int], optional
        Time points to mark with vertical lines
    style : PlotStyle, optional
        Styling configuration
    figsize : Tuple[float, float], default=(12, 5)
        Figure size
    **kwargs
        Additional arguments passed to plot_heatmap_with_scale
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : List[matplotlib.axes.Axes]
    """
    if style is None:
        style = PlotStyle()
    
    # Determine consistent color scale
    if heatmap2 is not None:
        vmin = min(np.nanmin(heatmap1), np.nanmin(heatmap2))
        vmax = max(np.nanmax(heatmap1), np.nanmax(heatmap2))
    else:
        vmin, vmax = np.nanmin(heatmap1), np.nanmax(heatmap1)
    
    print(f"Using color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")
    
    # Create subplots
    n_plots = 2 if heatmap2 is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    # Plot first heatmap
    _plot_single_heatmap(axes[0], heatmap1, titles[0], vmin, vmax, 
                        vertical_markers, style, **kwargs)
    
    # Plot second heatmap if available
    if heatmap2 is not None and len(titles) > 1:
        _plot_single_heatmap(axes[1], heatmap2, titles[1], vmin, vmax, 
                            vertical_markers, style, **kwargs)
    
    plt.tight_layout()
    return fig, axes


def _plot_single_heatmap(
    ax: Axes, 
    heatmap: np.ndarray, 
    title: str,
    vmin: float, 
    vmax: float,
    vertical_markers: Optional[List[int]],
    style: PlotStyle,
    **kwargs
) -> None:
    """Helper function to plot a single heatmap on given axes."""
    # Default parameters
    cmap = kwargs.get('cmap', 'Blues')
    sigma = kwargs.get('sigma', 3)
    contour_levels = kwargs.get('contour_levels', (0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    
    # Prepare data
    n_high_levels, n_timepoints = heatmap.shape
    time_axis = np.arange(0, n_timepoints)
    odor_axis = np.arange(1, n_high_levels + 1)
    
    X, Y = np.meshgrid(time_axis, odor_axis)
    smoothed = gaussian_filter(heatmap, sigma=sigma)
    
    # Main heatmap
    ax.pcolormesh(X, Y, heatmap, shading='nearest', cmap=cmap, 
                 vmin=vmin, vmax=vmax)
    ax.set_xlim(time_axis[0], time_axis[-1])
    
    # Contour lines
    other_levels = [lev for lev in contour_levels if lev != 0.5]
    if other_levels:
        ax.contour(X, Y, smoothed, levels=other_levels, colors="k", linewidths=3)
    ax.contour(X, Y, smoothed, levels=[0.5], colors="yellow", linewidths=2)
    
    # Styling
    ax.set_ylim([0, 50])
    ax.set_xticks([0, 182.5, 375, 552.5, 750])
    ax.set_xticklabels(["0", "", "375", "", "750"])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "", "50", "", "100"])
    
    ax.set_xlabel("Time after odour onset (ms)", fontsize=style.axes_labelsize, fontweight="bold")
    ax.set_ylabel("Number of Odours Present", fontsize=style.axes_labelsize, fontweight="bold")
    ax.set_title(title, fontsize=style.axes_titlesize, fontweight="bold")
    
    apply_style(ax, style)
    
    # Add vertical markers if specified
    if vertical_markers:
        add_vertical_markers(ax, vertical_markers)
