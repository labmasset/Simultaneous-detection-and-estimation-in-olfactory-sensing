from __future__ import annotations

from dataclasses import asdict
from typing import Any, Type
from pathlib import Path

import h5py

from .parameters import GammaSweepConfig

"""
Gamma-steepness sweep experiment:
This module reuses the existing batch-execution infrastructure (``NPresentSweepBase``)
while replacing the *nHigh* sweep with a sweep over the sigmoid steepness
``gamma_val`` in the SLAM model.  
The experiment keeps ``nHigh`` fixed at 20 and
computes an L1 error metric between the ground-truth concentration matrix *C*
and its estimate *c* returned by ``SLAM_sim``.
"""

import logging
import os
import time
from typing import Dict, Optional

import numpy as np

from ..simulator.parameters import SLAMParams
from .results_data import GammaSimulationResult
from .experiment_base import SweepExperimentBase
from ..simulator.mld_rnn import SLAMSim


def _build_params(cfg: GammaSweepConfig, params_cls: Type, **override: Any):
    """Construct ``params_cls`` using fields shared with ``cfg``.

    Parameters from ``cfg`` that match fields of ``params_cls`` are forwarded,
    and any ``override`` values replace them. This ensures simulator
    parameters stay in sync with the experiment configuration.
    """
    cfg_dict = asdict(cfg)
    fields = params_cls.__dataclass_fields__.keys()
    kwargs = {k: cfg_dict[k] for k in fields if k in cfg_dict}
    kwargs.update(override)
    return params_cls(**kwargs)


class GammaSteepnessSweep(SweepExperimentBase):
    """Batch runner for the ``gamma_val`` steepness experiment.

    This class subclasses *without modifying* the original ABC - only the
    private helpers that assumed an ``(nLow, nHigh)`` grid are replaced.  As a
    result, existing code and CLI entry points require no changes.
    """

    cfg: GammaSweepConfig  # type narrow for editors / linters

    # change signature to match combos (gamma_val, rep)
    def _simulate_single(
        self,
        nLow: Optional[int] = None,
        nHigh: Optional[int] = None,
        rep: Optional[int] = None,
        nOdor: Optional[int] = None,
        nSens: Optional[int] = None,
        gamma_val: Optional[float] = None,
    ) -> GammaSimulationResult:
        """Run a single simulation for the given ``gamma_val``."""
        if self.cfg.seed is not None:
            seed_val = hash((self.cfg.seed, gamma_val, rep)) & 0xFFFFFFFF
            np.random.seed(seed_val)

        params = _build_params(
            self.cfg,
            SLAMParams,
            num_high=self.cfg.num_high,
            gamma_val=gamma_val,
        )

        start = time.perf_counter()
        sim = SLAMSim(params_data=params)
        C_hat, Theta_hat = sim.simulate(saveAll=False)  # or True if needed
        C_hat = np.where(C_hat > 100, np.nan, C_hat)
        runtime = time.perf_counter() - start

        l1_err = float(np.mean(np.abs(C_hat - self.cfg.c_high)))

        return GammaSimulationResult(
            num_low=self.cfg.num_low,
            num_high=self.cfg.num_high,
            gamma_val=gamma_val,
            rep=rep,
            c_true=self.cfg.c_high,
            C=C_hat,
            L1=l1_err,
            runtime_sec=runtime,
        )

    def _save(  # type: ignore[override]
        self,
        results: Dict[str, GammaSimulationResult],
        batch_index: Optional[int] = None,
    ) -> None:
        """Save results of the sweep to an HDF5 file."""

        # remove large L1 error from the results
        def _clean(arr: np.ndarray) -> np.ndarray:
            arr = np.where(np.isinf(arr), np.nan, arr)
            return np.where(arr > 100, np.nan, arr)

        filename = self._get_save_filename(batch_index)
        save_path: Path = self._give_save_path() / filename

        with h5py.File(save_path, "w") as f:
            runs_grp = f.create_group("runs")

            for res in results.values():
                g = runs_grp.create_group(res.key)

                # main arrays -----------------------------------------------------
                g.create_dataset(
                    "C_hat",
                    data=_clean(res.C),  # noqa: E501
                    compression="gzip",
                    compression_opts=4,
                )
                # if you keep the raw C (true concentration) add it the same way

                # scalar attrs ----------------------------------------------------
                for k, v in asdict(res).items():
                    if k == "C" or v is None:
                        continue
                    g.attrs[k] = v

            # batch‑level info
            f.attrs["gamma_values"] = np.array(self.cfg.gamma_values, dtype=np.float32)
            f.attrs["repeats"] = self.cfg.repeats
            f.attrs["batch_index"] = -1 if batch_index is None else batch_index

        # provenance
        (self.cfg.out_dir / "config.json").write_text(self.cfg.to_json())
        self.logger.info("Saved %s", save_path)

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        """Return filename for the given batch index."""
        if batch_idx is None:
            return "poisson_results.h5"
        return f"gamma_sweep_results_batch{batch_idx}.h5"

    # Added override for run to use (gamma_val, rep) combos
    def run(self, mode: str = "auto") -> None:
        import os

        # build list of (gamma_val, rep) combos
        combos = [
            (g, r) for g in self.cfg.gamma_values for r in range(self.cfg.repeats)
        ]
        total = len(combos)
        task = os.getenv("SLURM_ARRAY_TASK_ID")

        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}")

        if mode == "auto":
            mode = (
                "slurm"
                if task
                else ("serial" if self.cfg.batch_size == total else "parallel")
            )
        self.logger.info("Running %d combos in '%s' mode", total, mode)

        match mode:
            case "slurm":
                self._run_slice(int(task), combos)
            case "serial":
                self._run_serial(combos)
            case "parallel":
                self._run_parallel(combos)
            case _:
                raise AssertionError("bad mode")


def run_gamma_sweep(
    cfg: Optional[GammaSweepConfig] = None, *, mode: str = "auto"
) -> None:
    """Entry point mirroring the existing runner interface.

    Parameters
    ----------
    cfg
        Instance of :class:`GammaSweepConfig`.  If *None*, the default
        configuration is used.
    mode
        One of ``{"auto", "slurm", "serial", "parallel"}``.  See
        :pymeth:`NPresentSweepBase.run` for semantics.
    """

    cfg = cfg or GammaSweepConfig()
    logger = logging.getLogger("GammaSweep")

    exp = GammaSteepnessSweep(cfg, logger=logger)
    exp.run(mode=mode)
