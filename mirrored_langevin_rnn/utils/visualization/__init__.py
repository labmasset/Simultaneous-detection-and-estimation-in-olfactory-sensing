"""
Visualization module for mirrored Langevin RNN olfaction project.

This module provides clean, reusable styling and plotting utilities
for creating consistent plots across the project.
"""

from .styling import (
    PlotStyle,
    COLORS,
    apply_style,
    add_vertical_markers,
)
from .dynamics_plot import DynamicsPlotter
from .heatmap_plot import (
    plot_heatmap_with_scale,
    plot_vertical_slices,
    plot_individual_slices,
    plot_comparative_heatmaps,
)

__all__ = [
    # Core styling
    'PlotStyle',
    'COLORS',
    'apply_style',
    'add_vertical_markers',
    
    # Plotters
    'DynamicsPlotter',
    
    # Heatmap utilities
    'plot_heatmap_with_scale',
    'plot_vertical_slices', 
    'plot_individual_slices',
    'plot_comparative_heatmaps',
]
