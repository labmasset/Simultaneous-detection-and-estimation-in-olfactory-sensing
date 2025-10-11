"""
General-purpose matplotlib styling utilities.

This module provides clean, reusable styling functions that can be used
across different projects to create consistent, publication-ready plots.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import Tuple, Optional, List
from dataclasses import dataclass, field

__all__ = [
    "PlotStyle",
    "apply_style",
    "add_vertical_markers",
]

# General color palette
COLORS = {
    'dark_orange': '#D95319',
    'gray': (0.7, 0.7, 0.7), 
    'dark_gray': (0.3, 0.3, 0.3),
    'blue': '#0072BD',
    'green': '#77AC30',
    'red': '#A2142F',
    'purple': '#7E2F8E',
    'yellow': '#EDB120',
    'cyan': '#4DBEEE',
}

@dataclass(slots=True)
class PlotStyle:
    """Clean, general-purpose plot styling configuration."""
    
    # Global rcParams settings
    figsize: Tuple[float, float] = field(init=False)
    font_size: int = 14
    label_size: int = 14  # Added attribute for label size

    fig_width: float = 6
    fig_height: float = 6
    fig_dpi: int = 300
    dpi: int = 300

    line_width: float = 2
    marker_size: float = 8
    cap_size: float = 3
    legend_frame: bool = False
    legend_fontsize: int = 12
    axes_linewidth: float = 2
    axes_titlesize: int = 16
    axes_labelsize: int = 14
    xtick_labelsize: int = 12
    ytick_labelsize: int = 12
    xtick_width: float = 1.5
    ytick_width: float = 1.5
    xtick_size: float = 5
    ytick_size: float = 5
    grid_linewidth: float = 0.5
    grid_alpha: float = 0.3
    max_ticks: int = 4
    # Per-axis settings (if ax is provided)
    hide_top_right: bool = True
    spine_width: float = 2.0
    tick_length: float = 10
    tick_width: float = 2
    tick_direction: str = "in"
    tick_labelsize: int = 16
    tick_pad: float = 10
    
    save_dir: str = "./figures"
    
    def __post_init__(self):
        self.figsize = (self.fig_width, self.fig_height)


def apply_style(ax: Optional[Axes] = None, style: Optional[PlotStyle] = None) -> None:
    """Apply clean matplotlib styling.
    
    Args:
        ax: Axes to style. If None, applies global rcParams only.
        style: PlotStyle instance. If None, uses default.
    """
    if style is None:
        style = PlotStyle()
    
    # Set global rcParams
    plt.rcParams.update({
        # fonts & text
        "font.size": style.font_size,

        # figure defaults
        "figure.figsize": (style.fig_width, style.fig_height),
        "figure.dpi": style.fig_dpi,
        "savefig.dpi": style.dpi,

        # lines, markers, errorbars
        "lines.linewidth": style.line_width,
        "lines.markersize": style.marker_size,
        "errorbar.capsize": style.cap_size,

        # legend
        "legend.frameon": style.legend_frame,
        "legend.fontsize": style.legend_fontsize,

        # axes
        "axes.linewidth": style.axes_linewidth,
        "axes.titlesize": style.axes_titlesize,
        "axes.labelsize": style.axes_labelsize,

        # ticks
        "xtick.labelsize": style.xtick_labelsize,
        "ytick.labelsize": style.ytick_labelsize,
        "xtick.major.width": style.xtick_width,
        "ytick.major.width": style.ytick_width,
        "xtick.major.size": style.xtick_size,
        "ytick.major.size": style.ytick_size,

        # grid
        "grid.linewidth": style.grid_linewidth,
        "grid.alpha": style.grid_alpha,
    })
    
    # 2D axis styling
    if ax is not None:
        if style.hide_top_right:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax.spines["bottom"].set_linewidth(style.spine_width)
        ax.spines["left"].set_linewidth(style.spine_width)

        ax.tick_params(
            which="both",
            direction=style.tick_direction,
            length=style.tick_length,
            width=style.tick_width,
            labelsize=style.tick_labelsize,
            pad=style.tick_pad,
            zorder=50
        )
        ax.xaxis.set_major_locator(plt.MaxNLocator(style.max_ticks))
        ax.yaxis.set_major_locator(plt.MaxNLocator(style.max_ticks))
        
        ax.margins(x=0.0, y=0.0)  # control scatter plot margins

def hide_labels(ax: Axes) -> None:
    """
    Hide all labels and ticks for a given Axes. 
    Enable it when assembling figure panels for the paper.
    """
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if ax.get_legend() is not None:
        ax.get_legend().set_visible(False)

    # Hide spines
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

def add_vertical_markers(ax: Axes, time_points: List[int], **kwargs) -> None:
    """
    Add vertical lines at specified time points.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to add markers to
    time_points : List[int]
        Time indices where to add vertical lines
    **kwargs
        Additional arguments passed to axvline
    """
    default_kwargs = {
        'color': 'greenyellow',
        'linestyle': 'solid',
        'alpha': 0.7,
        'linewidth': 2,
        'zorder': 1,
    }
    default_kwargs.update(kwargs)
    
    for t in time_points:
        ax.axvline(x=t, **default_kwargs)
