"""Base classes for executing sweep experiments."""

from __future__ import annotations
import os

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import h5py


from .parameters import PresentSweepConfig
from .results_data import SimulationResult


class SweepExperimentBase(ABC):
    """
    Base class for parameter sweep experiments with support for different execution modes:
    - SLURM array job distribution (each task processes a batch)
    - Serial execution (process all parameter combinations sequentially)
    - Parallel execution (process combinations using a local process pool)
    """

    _MODES = {"auto", "slurm", "serial", "parallel"}

    def __init__(self, cfg: PresentSweepConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

    def run(self, mode: str = "auto") -> None:
        """
        Run the parameter sweep experiment in the specified mode.

        Args:
            mode: Execution mode - 'auto', 'slurm', 'serial', or 'parallel'.
                 - 'auto': Choose mode based on environment and configuration
                 - 'slurm': Process a specific batch of parameters based on SLURM_ARRAY_TASK_ID
                 - 'serial': Process all parameters sequentially in a single process
                 - 'parallel': Process parameters in parallel using a local process pool
        """
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}")

        # Get all parameter combinations to run
        combos = self.cfg.combos
        total = len(combos)

        # Check if running as a SLURM array job
        slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")

        # Auto-select mode if not explicitly specified
        if mode == "auto":
            mode = "slurm" if slurm_task_id else (
                   "serial" if self.cfg.batch_size == total else "parallel")
        self.logger.info("Running %d combos in '%s' mode", total, mode)

        match mode:
            case "slurm":   self._run_slice(int(slurm_task_id), combos)
            case "serial": self._run_serial(combos)
            case "parallel": self._run_parallel(combos)
            case _: raise AssertionError("bad mode")

    def _run_slice(self, task_id: int, combos: List[Tuple]) -> None:
        """
        Process a specific batch of parameter combinations (for SLURM array jobs).
        Each SLURM array task processes a different batch determined by task_id.

        Args:
            task_id: SLURM array task ID (determines which batch to process)
            combos: All parameter combinations
        """
        # Calculate which subset of parameter combinations to process
        batch_combos = self._slice(task_id, combos)
        self.logger.info(
            f"[SLURM] Task {task_id}: Processing {len(batch_combos)} of {len(combos)} combinations")

        # Execute the batch (using parallel processing within this node)
        results = self._exec(batch_combos, parallel=True)

        # Save results with the batch index for later merging
        self._save(results, batch_index=task_id)
        self.logger.info(f"[SLURM] Task {task_id}: Batch processing complete")

    def _run_serial(self, combos: List[Tuple]):
        """
        Process all parameter combinations sequentially in a single process.

        Args:
            combos: All parameter combinations
        """
        self.logger.info(
            f"[SERIAL] Processing {len(combos)} combinations sequentially")
        results = self._exec(combos, parallel=False)
        self._save(results)
        self._save_config()
        self.logger.info("[SERIAL] All combinations processed")

    def _run_parallel(self, combos: List[Tuple]):
        """
        Process all parameter combinations in parallel using a local process pool.

        Args:
            combos: All parameter combinations
        """
        self.logger.info(
            f"[PARALLEL] Processing {len(combos)} combinations with {self.cfg.workers} workers")
        results = self._exec(combos, parallel=True)
        self._save(results)
        self._save_config()
        self.logger.info("[PARALLEL] All combinations processed")



    def _exec(self, combos: List[Tuple], *, parallel: bool) -> Dict[str, SimulationResult]:
        """
        Execute simulations for the given parameter combinations.

        Args:
            combos: Parameter combinations to process
            parallel: Whether to use parallel processing

        Returns:
            Dictionary mapping result keys to simulation results
        """
        results: Dict[str, SimulationResult] = {}

        # if parallel:
        #     # Process combinations in parallel using ProcessPoolExecutor
        #     with ProcessPoolExecutor(max_workers=self.cfg.workers) as pool:
        #         # Submit all jobs
        #         futures = [pool.submit(self._simulate_single, *combo)
        #                    for combo in combos]

        #         # Process results as they complete
        #         for future in as_completed(futures):
        #             try:
        #                 result = future.result()
        #                 results[result.key] = result
        #             except Exception:
        #                 self.logger.exception("Simulation failed")
        # else:
        # Process combinations sequentially
        failed_count = 0
        for combo in combos:
            try:
                result = self._simulate_single(*combo)
                results[result.key] = result
            except Exception:
                failed_count += 1
                self.logger.exception("Simulation failed for %s", combo)

        if failed_count > 0:
            self.logger.warning(
                f"Failed simulations: {failed_count}/{len(combos)}")
        if len(results) == 0:
            self.logger.error("ALL simulations failed! No results to save.")
        else:
            self.logger.info(
                f"Successful simulations: {len(results)}/{len(combos)}")

        return results

    def _slice(self, task_id: int, combos: List[Tuple]) -> List[Tuple]:
        """
        Get a slice of parameter combinations for a specific SLURM task.

        Args:
            task_id: SLURM array task ID
            combos: All parameter combinations

        Returns:
            Subset of parameter combinations for this task
        """
        batch_size = self.cfg.batch_size
        start_idx = task_id * batch_size
        end_idx = min((task_id + 1) * batch_size, len(combos))
        return combos[start_idx:end_idx]
    
    @abstractmethod
    def _simulate_single(self, **kwargs) -> SimulationResult:
        """
        Run a single simulation with the given parameters.
        Must be implemented by subclasses.
        """
        ...

    def _save_config(self):
        """Save experiment configuration to JSON file for reproducibility."""
        config_path = self._give_save_path() / "config.json"
        # config_path.write_text(self.cfg.to_json())
        self.logger.info(f"Configuration saved to {config_path}")

    def _save(self, results: Dict[str, SimulationResult], batch_index: Optional[int] = None) -> None:
        """
        Save simulation results to an HDF5 file.

        Args:
            results: Dictionary mapping result keys to simulation results
            batch_index: Optional batch index (for SLURM array jobs)
        """
        # Check if we have any results to save
        if len(results) == 0:
            self.logger.error("No results to save! All simulations failed.")
            self.logger.error("Not creating empty result file.")
            return

        filename = self._get_save_filename(batch_index)
        save_path = self._give_save_path() / filename

        self.logger.info(f"Saving {len(results)} results to {save_path}")

        with h5py.File(save_path, "w") as f:
            runs_grp = f.create_group("runs")

            for res in results.values():
                g = runs_grp.create_group(res.key)

                # Save array data with compression
                for field in ("C", "U", "Theta"):
                    data = getattr(res, field, None)
                    self.logger.debug(f"Saving field '{field}' with shape {data.shape if data is not None else 'None'}")
                    if data is not None:
                        g.create_dataset(
                            field,
                            data=data,
                            compression="gzip",
                            compression_opts=4,
                        )

                # Write scalar fields as attributes
                for k, v in asdict(res).items():
                    if k in {"C", "U", "Theta"} or v is None:
                        continue
                    # if hasattr(v, "item") and not isinstance(v, (str, bytes)):
                    #     try:
                    #         v = v.item()
                    #     except Exception:
                    #         pass
                    g.attrs[k] = v
                    self.logger.debug("Saving attribute '%s' with value %s", k, v)

            # Store batch index for potential merging later
            f.attrs["batch_index"] = -1 if batch_index is None else batch_index

        self.logger.info(f"Saved results to {save_path}")

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        """
        Get filename for saving results, with optional batch index.

        Args:
            batch_idx: Optional batch index (for SLURM array jobs)

        Returns:
            Filename for saving results
        """
        base_name = f"{self.__class__.__name__}_results"
        if batch_idx is None:
            return f"{base_name}.h5"
        return f"{base_name}_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        """
        Create and return the directory path for saving results.

        Returns:
            Path to the directory for saving results
        """
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        return self.cfg.out_dir

    def merge_batch_results(self, pattern: str = None) -> None:
        """
        Merge multiple batch result files into a single file.
        This is useful after running a SLURM array job to combine results.

        Args:
            pattern: Optional glob pattern for finding batch files
        """
        pattern = pattern or f"{self.__class__.__name__}_results_batch*.h5"
        parts = sorted(self.cfg.out_dir.glob(pattern))

        if not parts:
            self.logger.info("No batch files found to merge")
            return

        self.logger.info(f"Merging {len(parts)} batch files")

        # Implementation depends on result type - subclasses should override
        pass
