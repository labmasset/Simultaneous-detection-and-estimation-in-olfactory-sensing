import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from dataclasses import dataclass
from .styling import PlotStyle, apply_style, COLORS

# Project-specific color scheme
PROJECT_COLORS = {
    'present': COLORS['dark_orange'],
    'absent': COLORS['gray'],
    'threshold': COLORS['dark_orange'],
}


@dataclass
class DynamicsPlotter:
    """
    Simplified dynamics plotter for concentration, presence, and probability plots.
    """
    # Data arrays: shape (nOdor, nTime)
    C: np.ndarray | None = None              # Concentration matrix
    U: np.ndarray | None = None              # Presence matrix
    Theta: np.ndarray | None = None          # Probability matrix

    # Simulation parameters
    nLow: int = 0                            # Number of odors at low concentration
    nHigh: int = 0                           # Number of odors actually present
    cHigh: float | None = None               # Constant true concentration value
    sample_rate: int = 100                   # Steps saved per recorded point
    dt: float = 1e-5                         # Simulation timestep in seconds

    # Plotting configuration
    tMax: float = 1.0                        # Total real time span in seconds
    n_absent_plot: int = 100                 # Number of absent-odor traces to draw
    # True concentration trace (auto-filled)
    cTrue: np.ndarray | None = None
    pTrue: np.ndarray | None = None
    num_xticks: int = 5                      # Number of ticks on the x-axis
    style: PlotStyle | None = None           # Plot style
    
    # Color configuration
    present_color: str = COLORS['dark_orange']    # Color for present odors
    absent_color: str | tuple = COLORS['gray']    # Color for absent odors  
    threshold_color: str = COLORS['dark_orange']  # Color for threshold lines

    def __post_init__(self):
        # Initialize style if not provided
        if self.style is None:
            self.style = PlotStyle()

        # If no true-concentration trace provided, fill with constant cHigh
        if self.cTrue is None and self.C is not None:
            n_time = self.C.shape[1]
            self.cTrue = np.full((n_time, self.nHigh), self.cHigh, dtype=float)

    def _sample_indices(self, n_steps: int, sample_step: int = 1) -> np.ndarray:
        """Compute x-values in seconds for n_steps of recorded data."""
        return np.arange(n_steps) * self.dt * self.sample_rate * sample_step

    def _apply_time_labels(self, ax, n_steps: int):
        """Set x-axis limits and ticks based on real time."""
        # Compute true max time from number of steps
        t_max = (n_steps - 1) * self.dt * self.sample_rate
        ax.set_xlim(0, t_max)
        ax.xaxis.set_major_locator(LinearLocator(self.num_xticks))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.set_xlabel("Time (s)")

    def _plot_bulk(self,
                   data: np.ndarray,
                   ax,
                   sample_step: int = 1,
                   threshold: float | None = None):
        """Core plotting routine using configurable colors."""
        # Subsample the data along time
        mat = data[:, ::sample_step]
        n_steps = mat.shape[1]
        t_idx = self._sample_indices(n_steps, sample_step)

        # Plot absent odors as a LineCollection
        segments = [
            np.column_stack([t_idx, mat[i]])
            for i in range((self.nHigh+self.nLow),
                           min(data.shape[0], self.nHigh + self.nLow + self.n_absent_plot))
        ]
        lc = LineCollection(
            segments,
            colors=[self.absent_color],
            linewidths=1,
            zorder=1,
            alpha=0.50
        )
        ax.add_collection(lc)
        # Plot present odors
        if self.nLow > 0:
            ax.plot(
                t_idx,
                mat[:self.nLow].T,
                # color=(246/255, 202/255, 120/255),
                color="#E45B11F2",
                # color="#c62d1fdd",
                # color="#A90A0AFF",
                # color="#D20F0FFF",
                linewidth=1.5,
                zorder=9,
                alpha=0.99
            )
            ax.plot(
                t_idx,
                mat[self.nLow:(self.nLow+self.nHigh)].T,
                color=(109/255, 51/255, 178/255),
                # color = "#39568CFF",
                # color="#063592ea",
                # color="#094bceea",
                # color = "#2c36a4c7",
                linewidth=1.5,
                zorder=10,
                alpha=0.99
            )
        else:
            ax.plot(
            t_idx,
            mat[:self.nHigh].T,
            color=self.present_color,
            linewidth=1.5,
            zorder=10
            )   

        # Optional threshold line
        if threshold is not None:
            ax.axhline(
                threshold,
                color='black',
                linestyle='--',
                linewidth=3,
                zorder=5
            )

        # Apply time-axis formatting
        self._apply_time_labels(ax, n_steps)

    def plot_concentration(self):
        """Plot concentration dynamics with true concentration overlay."""
        fig, ax = plt.subplots(figsize=self.style.figsize)
        self._plot_bulk(self.C, ax)

        # Overlay the true concentration trace
        if self.cTrue is not None:
            t_true_idx = self._sample_indices(self.cTrue.shape[0])
            if self.nLow > 0:
                ax.plot(
                    t_true_idx,
                    self.cTrue[:, self.nLow-1].T,
                    # color=(153/255, 24/255, 24/255),
                    # color="#94B323FF",
                    # color="#A90A0AFF",
                    # color="#DC1818FF",
                    color="#E45B11F2",
                    linestyle='--',
                    linewidth=2,
                    zorder=5,  # Ensure lines are below the axis
                    alpha=0.99
                )
                ax.plot(
                    t_true_idx,
                    self.cTrue[:, (self.nLow+self.nHigh-1)].T,
                    # color="#063592ea",
                    # color="#a309bfea",
                    color=(109/255, 51/255, 178/255),
                    # color="#453781FF",
                    linestyle='--',
                    linewidth=2,
                    zorder=6,  # Ensure lines are below the axis
                    alpha=0.99
                )
            else:
                t_true_idx = self._sample_indices(self.cTrue.shape[0])
                ax.plot(
                    t_true_idx,
                    self.cTrue[:, :self.nHigh],
                    color=self.present_color,
                    linestyle='--',
                    linewidth=3,
                    zorder=2  # Ensure lines are below the axis
                )

        ax.set_ylabel("Concentration", fontsize=self.style.label_size)

        # Simple legend
        handles = [
            plt.Line2D([], [], color=self.present_color, linewidth=2),
            plt.Line2D([], [], color=self.absent_color, linewidth=0.5),
            plt.Line2D([], [], linestyle='-',
                       color="black", linewidth=3)
        ]
        ax.legend(handles, ['present', 'not present',
                  'true'], loc='best', frameon=False)

        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        return fig, ax
    
    def plot_output(self):
        """Plot concentration dynamics with true concentration overlay."""
        fig, ax = plt.subplots(figsize=self.style.figsize)
        # binarize; keep it boolean so counts are integers
        thr = 0.2
        # min_active_steps = 100
        b_theta = (self.Theta >= thr)

        # # rows strictly after the current high block
        # rows_after = np.arange(self.nHigh + 1, b_theta.shape[0])

        # # count active time steps per row (assumes time is axis=1; change if yours differs)
        # counts_after = np.count_nonzero(b_theta[rows_after], axis=1)

        # # rows to promote (>= min_active_steps) and those to leave
        # active_rows   = rows_after[counts_after >= min_active_steps]
        # inactive_rows = rows_after[counts_after <  min_active_steps]

        # # new row order: keep [0..nHigh] as-is, then promoted, then the rest
        # row_order = np.r_[np.arange(self.nHigh + 1), active_rows, inactive_rows]

        # # apply mask and reorder in one shot
        # C_output = (self.C * b_theta)[row_order, :]

        # # (optional) keep the binarized order aligned if you use it later
        # b_theta = b_theta[row_order, :]
        # self.C     = self.C[row_order, :]
        # self.Theta = self.Theta[row_order, :]

        # # update nHigh: old nHigh plus how many you just promoted
        # self.nHigh += active_rows.size
        C_output = (self.C * b_theta)
        self._plot_bulk(C_output, ax)

        # Overlay the true concentration trace
        if self.cTrue is not None:
            t_true_idx = self._sample_indices(self.cTrue.shape[0])
            if self.nLow > 1:
                ax.plot(
                    t_true_idx,
                    self.cTrue[:, self.nLow-1].T,
                    # color=(153/255, 24/255, 24/255),
                    # color="#94B323FF",
                    # color="#A90A0AFF",
                    # color="#DC1818FF",
                    color="#E45B11F2",
                    linestyle='--',
                    linewidth=2,
                    zorder=5,  # Ensure lines are below the axis
                    alpha=0.99
                )
            ax.plot(
                t_true_idx,
                self.cTrue[:, (self.nLow+self.nHigh-1)].T,
                # color="#063592ea",
                # color="#a309bfea",
                color=(109/255, 51/255, 178/255),
                # color="#453781FF",
                linestyle='--',
                linewidth=2,
                zorder=6,  # Ensure lines are below the axis
                alpha=0.99
            )

        ax.set_ylabel("Concentration", fontsize=self.style.label_size)

        # Simple legend
        handles = [
            plt.Line2D([], [], color=self.present_color, linewidth=2),
            plt.Line2D([], [], color=self.absent_color, linewidth=0.5),
            plt.Line2D([], [], linestyle='-',
                       color="black", linewidth=3)
        ]
        ax.legend(handles, ['present', 'not present',
                  'true'], loc='best', frameon=False)

        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)
        
        return fig, ax

    def plot_presence(self, sample_step: int = 1):
        """Plot presence dynamics."""
        fig, ax = plt.subplots(figsize=self.style.figsize)
        self._plot_bulk(self.U, ax, sample_step=sample_step)
        ax.set_ylabel("Presence (u)", fontsize=self.style.label_size)
        plt.tight_layout()

        # Simple legend
        handles = [
            plt.Line2D([], [], color=self.present_color, linewidth=1.5),
            plt.Line2D([], [], color=self.absent_color, linewidth=0.5)
        ]
        ax.legend(handles, ['present', 'not present'],
                  loc='best', frameon=False)

        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)

        return fig, ax

    def plot_presence(self, sample_step: int = 1):
        """Plot presence dynamics."""
        fig, ax = plt.subplots(figsize=self.style.figsize)
        self._plot_bulk(self.U, ax, sample_step=sample_step)
        ax.set_ylabel("Presence (u)", fontsize=self.style.label_size)
        plt.tight_layout()

        # Simple legend
        handles = [
            plt.Line2D([], [], color=self.present_color, linewidth=1.5),
            plt.Line2D([], [], color=self.absent_color, linewidth=0.5)
        ]
        ax.legend(handles, ['present', 'not present'],
                  loc='best', frameon=False)

        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)

        return fig, ax

    def plot_probability(self,
                         sample_step: int = 1,
                         threshold: float = 0.2):
        """Plot probability dynamics with threshold line."""
        fig, ax = plt.subplots(figsize=self.style.figsize)
        self._plot_bulk(self.Theta, ax, sample_step=sample_step,
                        threshold=threshold)
        ax.set_ylabel("Presence (p)", fontsize=self.style.label_size)
        plt.tight_layout()

        # Simple legend
        handles = [
            plt.Line2D([], [], color=self.present_color, linewidth=1.5),
            plt.Line2D([], [], color=self.absent_color, linewidth=0.5),
            plt.Line2D([], [], linestyle='--',
                       color=self.threshold_color, linewidth=3)
        ]
        ax.legend(handles, ['present', 'not present',
                  'threshold'], loc='best', frameon=False)
        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)
            
        return fig, ax
    
    def plot_true_concentration(self, sample_step: int = 1):
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
                # Overlay the true concentration trace
        if self.cTrue is not None:
            t_true_idx = self._sample_indices(self.cTrue.shape[0])
            if self.nLow > 1:
                ax.plot(
                    t_true_idx,
                    self.cTrue[:, self.nLow-1].T,
                    # color=(153/255, 24/255, 24/255),
                    # color="#94B323FF",
                    # color="#A90A0AFF",
                    # color="#DC1818FF",
                    color="#E45B11F2",
                    linestyle='--',
                    linewidth=2,
                    zorder=5,  # Ensure lines are below the axis
                    alpha=0.99
                )
            ax.plot(
                t_true_idx,
                self.cTrue[:, (self.nLow+self.nHigh-1)].T,
                # color="#063592ea",
                # color="#a309bfea",
                color=(109/255, 51/255, 178/255),
                # color="#453781FF",
                linestyle='--',
                linewidth=2,
                zorder=6,  # Ensure lines are below the axis
                alpha=0.99
            )
            
        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)
            
        return fig, ax
    
    def plot_true_presence(self, sample_step: int = 1):
        fig, ax = plt.subplots(figsize=self.style.figsize)
        
                # Overlay the true concentration trace
        if self.cTrue is not None:
            t_true_idx = self._sample_indices(self.cTrue.shape[0])
            if self.nLow > 1:
                ax.plot(
                    t_true_idx,
                    self.pTrue[:, self.nLow-1].T,
                    # color=(153/255, 24/255, 24/255),
                    # color="#94B323FF",
                    # color="#A90A0AFF",
                    # color="#DC1818FF",
                    color="#E45B11F2",
                    linestyle='-',
                    linewidth=2,
                    zorder=5,  # Ensure lines are below the axis
                    alpha=0.95
                )
            ax.plot(
                t_true_idx,
                self.pTrue[:, (self.nLow+self.nHigh-1)].T,
                # color="#063592ea",
                # color="#a309bfea",
                color=(109/255, 51/255, 178/255),
                # color="#453781FF",
                linestyle='--',
                linewidth=2,
                zorder=6,  # Ensure lines are below the axis
                alpha=0.95
            )
            
        apply_style(ax, style=self.style)
        
        # Set spine z-order to be above data
        for spine in ax.spines.values():
            spine.set_zorder(10)
            
        return fig, ax
