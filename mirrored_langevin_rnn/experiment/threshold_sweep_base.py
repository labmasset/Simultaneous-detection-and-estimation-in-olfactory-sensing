"""Base classes for threshold sweep experiments."""

import logging

import numpy as np
import h5py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pathlib import Path

from .parameters import ThresholdSweepConfig
from .results_data import ThresholdResult
from .experiment_base import SweepExperimentBase





class ThresholdSweepExperimentBase(SweepExperimentBase, ABC):
    """Adds early-stopping logic and heat-map I/O on top of *SweepExperimentBase*."""

    cfg: ThresholdSweepConfig
    _stop_rule = staticmethod(lambda score, thr: score >= thr)  # default = L1
    use_binary_search = (
        True  # Set to True to use binary search instead of linear search
    )
    # Whether to save the search paths (default: False to simplify data analysis)
    save_search_paths = False

    def __init__(
        self, cfg: ThresholdSweepConfig, logger: Optional[logging.Logger] = None
    ):
        super().__init__(cfg)  
        self.log = logger or logging.getLogger(__name__)

    def _simulate_single(
        self,
        num_potential_odors: int,
        num_osn: int,
        *,
        num_low: Optional[int] = None,
        num_high: Optional[int] = None,
        rep: Optional[int] = None,
        gamma_val: Optional[float] = None,
        **kwargs,
    ) -> ThresholdResult:
        """Run one (num_potential_odors, num_osn) point of the sweep and return aggregated stats."""

        cfg = self.cfg
        n_high_vals: List[int] = getattr(
            cfg, "n_high_values", getattr(cfg, "num_high_values")
        )

        # Choose the appropriate search method
        if self.use_binary_search:
            return self._binary_search_thresholds(
                num_potential_odors, num_osn, n_high_vals
            )
        else:
            # Original linear search implementation
            return self._linear_search_thresholds(
                num_potential_odors, num_osn, n_high_vals
            )

    def _linear_search_thresholds(
        self, num_potential_odors: int, num_osn: int, n_high_vals: List[int]
    ) -> ThresholdResult:
        """Original linear search implementation."""
        # Configure logging for worker processes
        import logging
        from ..logging_utils import setup_logging
        if not logging.getLogger().handlers:
            setup_logging("INFO")

        cfg = self.cfg
        thresholds: List[int] = []

        self.log.info(
            f"Using linear search for nOdor={num_potential_odors}, nSens={num_osn}")

        for rep_idx in range(cfg.repeats):
            self.log.info(f"  Starting linear search for rep {rep_idx}")
            thr_for_rep: Optional[int] = None
            for num_high_ in n_high_vals:
                score = self._metric(
                    num_potential_odors=num_potential_odors,
                    num_high=num_high_,
                    num_osn=num_osn,
                    rep=rep_idx,
                )
                self.log.info(f"    num_high={num_high_}, score={score:.3f}")
                if self._stop_rule(score, cfg.error_threshold):
                    thr_for_rep = num_high_
                    break
            thresholds.append(thr_for_rep or max(n_high_vals))

        return ThresholdResult(
            num_potential_odors=num_potential_odors,
            num_osn=num_osn,
            threshold_num_high=float(np.mean(thresholds)),
            std_threshold=float(np.std(thresholds)),
        )

    def _binary_search_thresholds(
        self, num_potential_odors: int, num_osn: int, n_high_vals: List[int]
    ) -> ThresholdResult:
        """Binary search implementation for faster threshold finding."""
        import time

        # Configure logging for worker processes
        import logging
        from ..logging_utils import setup_logging
        if not logging.getLogger().handlers:
            setup_logging("INFO")

        cfg = self.cfg
        thresholds: List[int] = []
        metrics_data = (
            [] if self.save_search_paths else None
        )  # Only store paths if explicitly enabled

        self.log.info(
            f"Using binary search for nOdor={num_potential_odors}, nSens={num_osn}"
        )

        for rep_idx in range(cfg.repeats):
            self.log.info(f"  Starting binary search for rep {rep_idx}")

            # Binary search requires sorted n_high values
            sorted_n_high = sorted(n_high_vals)
            search_path = [] if self.save_search_paths else None

            low, high = 0, len(sorted_n_high) - 1
            thr_for_rep: Optional[int] = None

            while low <= high:
                mid = (low + high) // 2
                n_high = sorted_n_high[mid]

                start_time = time.time()
                score = self._metric(
                    num_potential_odors=num_potential_odors,
                    num_high=n_high,
                    num_osn=num_osn,
                    rep=rep_idx,
                )
                elapsed = time.time() - start_time

                # Only store search path if enabled
                if self.save_search_paths and search_path is not None:
                    search_path.append((n_high, score))

                self.log.info(
                    f"    num_high={n_high}, score={score:.3f}, time={elapsed:.3f}s"
                )

                if self._stop_rule(score, cfg.error_threshold):
                    # Found threshold, but continue binary search to find exact transition point
                    thr_for_rep = n_high
                    high = mid - 1
                else:
                    # Need higher n_high value
                    low = mid + 1

            # If we never found a threshold, use the maximum value
            final_threshold = thr_for_rep or max(sorted_n_high)
            thresholds.append(final_threshold)

            # Only store metrics data if enabled
            if (
                self.save_search_paths
                and metrics_data is not None
                and search_path is not None
            ):
                metrics_data.append(search_path)

            self.log.info(f"  Rep {rep_idx} threshold: {final_threshold}")
            if self.save_search_paths and search_path:
                self.log.info(f"  Binary search path: {search_path}")

        result = ThresholdResult(
            num_potential_odors=num_potential_odors,
            num_osn=num_osn,
            threshold_num_high=float(np.mean(thresholds)),
            std_threshold=float(np.std(thresholds)),
        )

        # Attach search path data only if saving is enabled
        if self.save_search_paths and metrics_data is not None:
            setattr(result, "search_paths", metrics_data)

        return result

    @abstractmethod
    def _metric(
        self, *, num_potential_odors: int, num_high: int, num_osn: int, rep: int
    ) -> float: ...

    # type: ignore[override]
    def _save(self, results: Dict[str, ThresholdResult], batch_index: Optional[int] = None):
        """Save threshold results as a grid in HDF5 format."""
        cfg = self.cfg
        nO_v = list(cfg.n_odor_values)
        nS_v = list(cfg.n_sens_values)

        # Create the threshold grid
        grid = np.full((len(nS_v), len(nO_v)), np.nan)
        for r in results.values():
            grid[nS_v.index(r.num_osn), nO_v.index(r.num_potential_odors)] = (
                r.threshold_num_high
                if r.threshold_num_high is not None
                else max(cfg.n_high_values)
            )

        # Save the grid to file
        out_dir = self._give_save_path()
        fname = self._get_save_filename(batch_index)
        with h5py.File(out_dir / fname, "w") as f:
            f.create_dataset("grid", data=grid)
            f.attrs.update(
                {
                    "num_potential_odors_values": nO_v,
                    "num_osn_values": nS_v,
                    "batch_index": -1 if batch_index is None else batch_index,
                    "use_binary_search": self.use_binary_search,
                }
            )

            # If binary search was used and saving search paths is enabled, save them
            if self.use_binary_search and self.save_search_paths:
                paths_grp = f.create_group("search_paths")
                for key, res in results.items():
                    if hasattr(res, "search_paths"):
                        run_grp = paths_grp.create_group(key)
                        # Convert search paths to datasets
                        for rep_idx, path_data in enumerate(res.search_paths):
                            # Convert to array with columns: num_high, score
                            path_array = np.array(path_data)
                            run_grp.create_dataset(
                                f"rep_{rep_idx}", data=path_array, compression="gzip"
                            )

        self.log.info(f"Saved threshold grid to {out_dir / fname}")

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        """Get filename for saving threshold results. Can be overridden by subclasses."""
        prefix = getattr(self, "basename", self.__class__.__name__)
        if self.use_binary_search:
            prefix = f"{prefix}_binary"

        if batch_idx is None:
            return f"{prefix}_threshold_results.h5"
        return f"{prefix}_threshold_results_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        """Create and return the directory path for saving results. Can be overridden by subclasses."""
        if hasattr(self, "subdir"):
            path = self.cfg.out_dir / self.subdir
        else:
            path = self.cfg.out_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def merge_batch_results(self, pattern: str = None):
        """Merge multiple batch threshold grids by taking the maximum value at each point."""
        prefix = getattr(self, "basename", self.__class__.__name__)
        if self.use_binary_search:
            prefix = f"{prefix}_binary"

        pattern = pattern or f"{prefix}_threshold_results_batch*.h5"

        out_dir = self._give_save_path()
        parts = sorted(out_dir.glob(pattern))
        if not parts:
            self.log.info("No batch files found to merge")
            return

        self.log.info(f"Merging {len(parts)} threshold grid files")

        # Start with None and take maximum of all grids
        merged = None
        for p in parts:
            with h5py.File(p, "r") as f:
                grid_part = f["grid"][:]
                merged = grid_part if merged is None else np.fmax(
                    merged, grid_part)

        # Save merged grid
        merged_file = out_dir / f"{prefix}_threshold_results_merged.h5"
        with h5py.File(merged_file, "w") as f:
            f.create_dataset("grid", data=merged)
            f.attrs.update(
                {
                    "num_potential_odors_values": list(self.cfg.n_odor_values),
                    "num_osn_values": list(self.cfg.n_sens_values),
                    "use_binary_search": self.use_binary_search,
                }
            )

        self.log.info(f"Created merged threshold grid: {merged_file}")


class L1ThresholdBase(ThresholdSweepExperimentBase, ABC):
    """Stop when L1 >= ``cfg.error_threshold``."""

    _stop_rule = staticmethod(lambda score, thr: score > thr)


class AUCThresholdBase(ThresholdSweepExperimentBase, ABC):
    """Stop when AUC <= ``cfg.error_threshold``."""

    _stop_rule = staticmethod(lambda auc, thr: auc < thr)
