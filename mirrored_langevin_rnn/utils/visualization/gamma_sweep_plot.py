"""
Utility helfrom pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import logging
from .styling import PlotStyle, create_figure

logger = logging.getLogger(__name__)nalysing SLAM sweep experiments.

▪  _compute_gamma_L1_curve  - load one “gamma_sweep” .mat file and return
   gamma values, mean L1 error, and std across repeats.

▪  plot_gamma_L1_curves     - convenience wrapper to plot one or more
   gamma-sweep result files on the same figure.

Both functions parallel the API you already have for the nHigh sweep so the
calling code in notebooks feels identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from scipy import io as sio
import matplotlib.pyplot as plt
import logging
from .styling import PlotStyle, create_figure

logger = logging.getLogger(__name__)


def _compute_gamma_L1_curve(
    mat_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return gamma values, mean L1 error, and std-dev L1 error for one .mat file.

    The .mat file is expected to come from `GammaSteepnessSweep._save`, i.e.
    it must contain:

        •  L1_all      - shape (nGamma, nRep)
        •  gamma_vals  - shape (nGamma,)
    """
    data = sio.loadmat(mat_path, simplify_cells=True)

    L1_all = np.asarray(data["L1_all"])          # (nGamma, nRep)
    # Replace unreasonably large values with NaN
    L1_all = np.where(L1_all > 1e6, np.nan, L1_all)

    gamma  = np.asarray(data["gamma_vals"]).ravel()

    # stats across the repeat axis
    mu  = L1_all.mean(axis=1)                    # (nGamma,)
    sig = L1_all.std(axis=1, ddof=1)             # (nGamma,)

    return gamma, mu, sig


def plot_gamma_L1_curves(
    mat_paths: Sequence[str | Path],
    labels:    Sequence[str],
    *,
    show_std: bool = False,
    colors:   list[str] | None = None,
    title:    str | None = None,
    style = None,
) -> None:
    if style is None:
        from .styling import PlotStyle
        style = PlotStyle()
    
    if colors is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if title is None:
        title = "L1 Error vs. Sigmoid Steepness gamma"

    fig, ax = create_figure(style)

    for i, (path, label) in enumerate(zip(mat_paths, labels)):
        color = colors[i % len(colors)]
        gamma, mu, sig = _compute_gamma_L1_curve(path)
        logger.debug("mu values: %s", mu)
        
        # Plot error band
        if show_std:
            ax.fill_between(gamma, mu - sig, mu + sig,
                           color=color, alpha=0.25, linewidth=0)
        
        # Plot main line
        ax.plot(gamma, mu, marker="o", linewidth=2.5, color=color, label=label)

    ax.set_title(title, fontsize=style.title_size)
    ax.set_xlabel("Sigmoid steepness gamma", fontsize=style.label_size)
    ax.set_ylabel("Average L1 Error", fontsize=style.label_size)
    ax.grid(True, alpha=style.grid_alpha)
    ax.legend(fontsize=style.legend_size, frameon=False)
    plt.tight_layout()
    plt.show()
