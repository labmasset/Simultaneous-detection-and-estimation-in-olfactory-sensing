"""N-present sweep experiments for different simulators."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Type
from dataclasses import asdict
import logging

import h5py
import numpy as np

from mirrored_langevin_rnn.simulator.mld_rnn_circuit import SLAMSimCircuit

from ..simulator.mld_rnn import SLAMSim
from ..simulator.mld_rnn_distributed import SLAMGeomSim, SLAMDistSim
from ..simulator.mld_rnn_ks import SLAMSimKST
from ..simulator.poisson_rnn import PoissonCircuitSim
from ..simulator.parameters import SLAMParams, SLAMKParams, PoissonParams, SLAMParamsCircuit
from .parameters import PresentSweepConfig
from .experiment_base import SweepExperimentBase
from .results_data import SimulationResult

class GenericNPresentSweep(SweepExperimentBase):
    """Generic N-Present sweep experiment that can be configured for different simulation models."""

    def __init__(
        self,
        cfg: PresentSweepConfig,
        params_cls: Type,
        sim_cls: Type,
        basename: str,
        subdir: str,
    ):
        super().__init__(cfg)
        self.params_cls = params_cls
        self.sim_cls = sim_cls
        self.basename = basename
        self.subdir = subdir
        self.sim_kwargs: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def _simulate_single(
        self,
        nLow: Optional[int] = None,
        nHigh: Optional[int] = None,
        rep: Optional[int] = None,
        **kwargs,
    ) -> SimulationResult:
        """
        Run one configuration of the sweep.
        """
        # Configure logging for worker processes
        import logging
        from ..logging_utils import setup_logging
        if not logging.getLogger().handlers:
            setup_logging("INFO")
            
        params_cls = self.params_cls
        sim_cls = self.sim_cls
        cfg: PresentSweepConfig = self.cfg
        nLow: int = nLow
        nHigh: int = nHigh
        rep: int = rep
        
        if cfg.seed is not None:
            seed_val = hash((cfg.seed, nLow, nHigh, rep)) & 0xFFFFFFFF
            np.random.seed(seed_val)

        cfg_dict = asdict(cfg)
        fields = params_cls.__dataclass_fields__
        # Only include fields that can be initialized (init=True)
        params_kwargs = {k: cfg_dict[k] for k, field_info in fields.items() 
                if k in cfg_dict and field_info.init}
        params_kwargs.update({"num_low": nLow, "num_high": nHigh})

        params = params_cls(**params_kwargs)
        sim = sim_cls(params)

        start = time.time()
        result = sim.simulate()
        elapsed = time.time() - start

        # Unpack result: SLAM returns (C, Theta), Poisson returns C
        if isinstance(result, tuple) and len(result) == 2:
            C, Theta = result
        else:
            C, Theta = result, None

        # Log execution time
        sim_name = sim_cls.__name__.replace("Sim", "")
        self.logger.info("%s (%d,%d,%d) %.2fs", sim_name, nLow, nHigh, rep, elapsed)

        return SimulationResult(
            num_low=nLow,
            num_high=nHigh,
            rep=rep,
            C=C,
            Theta=Theta,
            c_true=cfg.c_high,
        )

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        if batch_idx is None:
            return f"{self.basename}_results.h5"
        return f"{self.basename}_results_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        """Create and return subdirectory path for saving results."""
        path = self.cfg.out_dir / self.subdir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def merge_batch_results(self) -> None:
        """Merge multiple batch result files into a single merged file."""
        pattern = f"{self.basename}_results_batch*.h5"
        parts = sorted(self._give_save_path().glob(pattern))
        if not parts:
            self.logger.info("No batch files found to merge")
            return

        self.logger.info(f"Merging {len(parts)} batch files for {self.basename}")

        # Create a new merged file
        merged_file = self._give_save_path() / f"{self.basename}_results_merged.h5"

        # Collect all run data from batch files
        all_runs = {}
        for part_path in parts:
            with h5py.File(part_path, "r") as f:
                if "runs" not in f:
                    continue

                for key in f["runs"]:
                    if key in all_runs:
                        continue  # Skip duplicates

                    run_data = {}
                    g = f["runs"][key]

                    # Copy datasets
                    for field in ("C", "U", "Theta"):
                        if field in g:
                            run_data[field] = g[field][:]

                    # Copy attributes
                    for attr_name in g.attrs:
                        run_data[attr_name] = g.attrs[attr_name]

                    all_runs[key] = run_data

        # Only create merged file if we have data
        if len(all_runs) == 0:
            self.logger.warning("No valid runs found in batch files - not creating empty merged file")
            return

        # Write merged file
        with h5py.File(merged_file, "w") as f:
            runs_grp = f.create_group("runs")

            for key, run_data in all_runs.items():
                g = runs_grp.create_group(key)

                # Write datasets
                for field in ("C", "U", "Theta"):
                    if field in run_data:
                        g.create_dataset(
                            field,
                            data=run_data[field],
                            compression="gzip",
                            compression_opts=4,
                        )

                # Write attributes
                for attr_name, value in run_data.items():
                    if attr_name not in ("C", "U", "Theta"):
                        g.attrs[attr_name] = value

        self.logger.info(f"Created merged file: {merged_file} with {len(all_runs)} runs")


# Specific sweeps using the generic implementation
class SLAMNaiveNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMSim,
            basename="slam_naive",
            subdir="naive",
        )

class SLAMCircuitNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMParamsCircuit,
            sim_cls=SLAMSimCircuit,
            basename="slam_circuit",
            subdir="circuit",
        )


class SLAMNaiveKNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMKParams,
            sim_cls=SLAMSim,
            basename="slam_naive_k",
            subdir="naive_k",
        )


class SLAMGeomNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMGeomSim,
            basename="slam_geom",
            subdir="geom",
        )


class SLAMDistNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMDistSim,
            basename="slam_dist",
            subdir="dist",
        )


class PoissonNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=PoissonParams,
            sim_cls=PoissonCircuitSim,
            basename="poisson",
            subdir="poisson",
        )


class SLAMKSNPresentSweep(GenericNPresentSweep):
    def __init__(self, cfg: PresentSweepConfig):
        super().__init__(
            cfg,
            params_cls=SLAMKParams,
            sim_cls=SLAMSimKST,
            basename="slam_ks",
            subdir="ks",
        )
