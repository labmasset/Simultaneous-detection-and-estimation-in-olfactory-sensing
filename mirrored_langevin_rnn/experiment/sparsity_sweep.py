"""Sweep the sensing matrix sparsity for different simulators."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

from .experiment_base import SweepExperimentBase
from .parameters import SparsitySweepConfig
from .results_data import SparsitySimulationResult
from ..simulator.parameters import PoissonParams, SLAMParams, SLAMKParams
from ..simulator.poisson_rnn import PoissonCircuitSim
from ..simulator.mld_rnn import SLAMSim
from ..simulator.mld_rnn_ks import SLAMSimKST
from ..simulator.mld_rnn_distributed import SLAMGeomSim, SLAMDistSim


def _build_params(cfg: SparsitySweepConfig, params_cls: Type, **override: Any):
    """Build simulator parameters from ``cfg`` and ``override`` values."""
    cfg_dict = asdict(cfg)
    fields = params_cls.__dataclass_fields__
    kwargs = {k: cfg_dict[k] for k, f in fields.items() if k in cfg_dict and f.init}
    kwargs.update(override)
    return params_cls(**kwargs)


class GenericSparsitySweep(SweepExperimentBase):
    """Base class implementing a simple sparsity sweep."""

    def __init__(
        self,
        cfg: SparsitySweepConfig,
        params_cls: Type,
        sim_cls: Type,
        basename: str,
        subdir: str,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        from ..logging_utils import setup_logging

        super().__init__(cfg)
        self.params_cls = params_cls
        self.sim_cls = sim_cls
        self.basename = basename
        self.subdir = subdir
        self.logger = logger or logging.getLogger(__name__)
        if not logging.getLogger().handlers:
            setup_logging("INFO")


    def _simulate_single(
        self,
        nLow: Optional[int] = None,
        nHigh: Optional[int] = None,
        sparsity: Optional[float] = None,
        rep: Optional[int] = None,
    ) -> SparsitySimulationResult:
        """Run one simulation for the given configuration."""
        import logging
        from ..logging_utils import setup_logging

        if not logging.getLogger().handlers:
            setup_logging("INFO")

        assert nLow is not None
        assert nHigh is not None
        assert sparsity is not None
        assert rep is not None

        if self.cfg.seed is not None:
            seed_val = hash((self.cfg.seed, nLow, nHigh, sparsity, rep)) & 0xFFFFFFFF
            np.random.seed(seed_val)

        params = _build_params(
            self.cfg,
            self.params_cls,
            num_low=nLow,
            num_high=nHigh,
            sensing_matrix_sparsity=sparsity,
        )
        sim = self.sim_cls(params)

        start = time.time()
        result = sim.simulate()
        elapsed = time.time() - start

        self.logger.info(
            "%s sparsity=%.3f (nLow=%d, nHigh=%d, rep=%d) %.2fs",
            self.sim_cls.__name__.replace("Sim", ""),
            sparsity,
            nLow,
            nHigh,
            rep,
            elapsed,
        )

        if isinstance(result, tuple) and len(result) == 2:
            C, Theta = result
        else:
            C, Theta = result, None

        return SparsitySimulationResult(
            num_low=nLow,
            num_high=nHigh,
            rep=rep,
            C=C,
            Theta=Theta,
            c_true=self.cfg.c_high,
            sparsity=float(sparsity),
        )

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        if batch_idx is None:
            return f"{self.basename}_results.h5"
        return f"{self.basename}_results_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        path = self.cfg.out_dir / self.subdir
        path.mkdir(parents=True, exist_ok=True)
        return path




class PoissonSparsitySweep(GenericSparsitySweep):
    def __init__(self, cfg: SparsitySweepConfig, logger: Optional[logging.Logger] = None):
        super().__init__(
            cfg,
            params_cls=PoissonParams,
            sim_cls=PoissonCircuitSim,
            basename="poisson",
            subdir="poisson",
            logger=logger,
        )


class SLAMSparsitySweep(GenericSparsitySweep):
    def __init__(self, cfg: SparsitySweepConfig, logger: Optional[logging.Logger] = None):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMSim,
            basename="slam",
            subdir="slam",
            logger=logger,
        )


class SLAMKSSparsitySweep(GenericSparsitySweep):
    def __init__(self, cfg: SparsitySweepConfig, logger: Optional[logging.Logger] = None):
        super().__init__(
            cfg,
            params_cls=SLAMKParams,
            sim_cls=SLAMSimKST,
            basename="slam_ks",
            subdir="ks",
            logger=logger,
        )


class SLAMGeomSparsitySweep(GenericSparsitySweep):
    def __init__(self, cfg: SparsitySweepConfig, logger: Optional[logging.Logger] = None):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMGeomSim,
            basename="slam_geom",
            subdir="geom",
            logger=logger,
        )


class SLAMDistSparsitySweep(GenericSparsitySweep):
    def __init__(self, cfg: SparsitySweepConfig, logger: Optional[logging.Logger] = None):
        super().__init__(
            cfg,
            params_cls=SLAMParams,
            sim_cls=SLAMDistSim,
            basename="slam_dist",
            subdir="dist",
            logger=logger,
        )
