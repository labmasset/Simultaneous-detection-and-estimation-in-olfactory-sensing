"""Specific threshold sweep implementations."""

from __future__ import annotations

from time import time
from typing import Any, Optional, Literal, Type
from dataclasses import asdict
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

from .threshold_sweep_base import (
    ThresholdSweepExperimentBase,
    AUCThresholdBase,
    L1ThresholdBase,
)
from .parameters import ThresholdSweepConfig

def _l1(C: np.ndarray, n_high: int, target: float) -> float:
    """Mean-abs-error of the `n_high` most-concentrated odors vs *target*."""
    return float(np.abs(C[:n_high, -1] - target).mean())


def _get_sensing_matrix_suffix(cfg: ThresholdSweepConfig) -> str:
    """Generate suffix based on sensing matrix configuration."""
    matrix_type = cfg.sensing_matrix_type
    sparsity = cfg.sensing_matrix_sparsity
    
    if matrix_type == "dense_gamma":
        return "affinity_dense_gamma"
    elif matrix_type == "sparse_binary":
        return f"affinity_sparse_binary_sparsity_{sparsity:.2f}"
    elif matrix_type == "sparse_gamma":
        return f"affinity_sparse_gamma_sparsity_{sparsity:.2f}"
    else:
        return f"affinity_{matrix_type}"


def _build_params(cfg: ThresholdSweepConfig, params_cls: Type, **override: Any):
    """Construct ``params_cls`` using fields shared with ``cfg``.

    Parameters from ``cfg`` that match fields of ``params_cls`` are forwarded,
    and any ``override`` values replace them.  This ensures simulator
    parameters stay in sync with the experiment configuration.
    """
    cfg_dict = asdict(cfg)
    fields = params_cls.__dataclass_fields__
    # Only include fields that can be initialized (init=True)
    kwargs = {k: cfg_dict[k] for k, field_info in fields.items()
              if k in cfg_dict and field_info.init}
    kwargs.update(override)
    return params_cls(**kwargs)


class PoissonL1ThresholdSweep(L1ThresholdBase):
    """Poisson circuit L1 threshold sweep with GPU acceleration."""

    _stop_rule = staticmethod(lambda score, thr: score > thr)
    basename = "poisson_l1"
    subdir = "poisson/l1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = {}
        self.log.info(
            f"Initialized Poisson L1 threshold sweep with basename '{self.basename}' and subdir '{self.subdir}'")

    def _metric(
        self, *, num_potential_odors: int, num_high: int, num_osn: int, rep: int
    ) -> float:
        """Compute L1 error for Poisson circuit model with optimizations."""
        # Configure logging for worker processes
        import time
        from ..simulator.parameters import PoissonParams
        from ..simulator.poisson_rnn import PoissonCircuitSim

        # Create parameter - device is already set from system config
        params = _build_params(
            self.cfg,
            PoissonParams,
            num_osn=num_osn,
            num_potential_odors=num_potential_odors,
            num_high=num_high,
        )

        C = PoissonCircuitSim(params_data=params).simulate()
        score = _l1(C, num_high, target=self.cfg.c_high)
        return score

    def _give_save_path(self) -> Path:
        """Create and return subdirectory path for saving results."""
        # Include sensing matrix information in the path
        matrix_info = _get_sensing_matrix_suffix(self.cfg)
        path = self.cfg.out_dir / self.subdir / matrix_info
        path.mkdir(parents=True, exist_ok=True)
        return path
    
# class PoissonAUCThresholdSweep(AUCThresholdBase):
class PoissonRankThresholdSweep(AUCThresholdBase):

    basename = "poisson_auc"
    subdir = "poisson/auc"

    def _metric(
        self, *, num_potential_odors: int, num_high: int, num_osn: int, rep: int
    ) -> float:
        from ..simulator.parameters import PoissonParams
        from ..simulator.poisson_rnn import PoissonCircuitSim

        params = _build_params(
            self.cfg,
            PoissonParams,
            num_osn=num_osn,
            num_potential_odors=num_potential_odors,
            num_high=num_high,
        )
        C = PoissonCircuitSim(params).simulate()

        # Compute rank-based accuracy
        last = C[:, -1]
        presence = last > self.cfg.c_high * 0.5  # presence threshold
        presence = presence.astype(float)
        
        labels = np.zeros(num_potential_odors, dtype=bool)
        labels[:num_high] = True
        score = float(roc_auc_score(labels, presence))
        return score

    def _give_save_path(self) -> Path:
        """Create and return subdirectory path for saving results."""
        # Include sensing matrix information in the path
        matrix_info = _get_sensing_matrix_suffix(self.cfg)
        path = self.cfg.out_dir / self.subdir / matrix_info
        path.mkdir(parents=True, exist_ok=True)
        return path

class SLAML1ThresholdSweep(L1ThresholdBase):
    """SLAM model L1 threshold sweep with early stopping when error exceeds threshold."""

    prior_type: Literal["bernoulli", "kumaraswamy"]

    def __init__(self, cfg, logger=None):
        if cfg.model == "slam_ks":
            self.prior_type = "kumaraswamy"
        else:  # default to bernoulli
            self.prior_type = "bernoulli"
        super().__init__(cfg, logger)
        self.basename = f"slam_l1_{self.prior_type}"
        self.subdir = f"slam/l1/{self.prior_type}"

        self.log.info(
            f"Initialized SLAM L1 threshold sweep with {self.prior_type} prior"
        )
        self.log.info(
            f"Using default nOdor values from config: {list(cfg.n_odor_values)}")
        self.log.info(
            f"Using default nSens values from config: {list(cfg.n_sens_values)}")
        self.log.info(
            f"Using default nHigh values from config: {list(cfg.n_high_values)}")
        self.log.info(f"Error threshold: {cfg.error_threshold}")
        self.log.info(f"Repeats: {cfg.repeats}")

    def _metric(
        self, *, num_potential_odors: int, num_high: int, num_osn: int, rep: int
    ) -> float:
        """Compute L1 error for SLAM model."""
        # Configure logging for worker processes
        from ..simulator.parameters import SLAMParams, SLAMKParams
        from ..simulator.mld_rnn import SLAMSim
        from ..simulator.mld_rnn_ks import SLAMSimKST

        params_cls = SLAMParams if self.prior_type == "bernoulli" else SLAMKParams
        params = _build_params(
            self.cfg,
            params_cls,
            num_osn=num_osn,
            num_potential_odors=num_potential_odors,
            num_high=num_high,
        )

        if self.prior_type == "bernoulli":
            C, _ = SLAMSim(params).simulate()
        else:
            C, _ = SLAMSimKST(params).simulate()
        score = _l1(C, num_high, target=self.cfg.c_high)
        return score

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        if batch_idx is None:
            return f"{self.basename}_threshold_results.h5"
        return f"{self.basename}_threshold_results_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        """Create and return subdirectory path for saving results."""
        # Include sensing matrix information in the path
        matrix_info = _get_sensing_matrix_suffix(self.cfg)
        path = self.cfg.out_dir / self.subdir / matrix_info
        path.mkdir(parents=True, exist_ok=True)
        return path


class SLAMAUCThresholdSweep(AUCThresholdBase):
    """SLAM model AUC threshold sweep with early stopping when AUC < 0.85."""

    _stop_rule = staticmethod(lambda auc, thr: auc <= thr)
    prior_type: Literal["bernoulli", "kumaraswamy"]

    def __init__(self, cfg, logger=None):
        if cfg.model == "slam_ks":
            self.prior_type = "kumaraswamy"
        else:  # default to bernoulli
            self.prior_type = "bernoulli"
        
        super().__init__(cfg, logger)
        self.basename = f"slam_auc_{self.prior_type}"
        self.subdir = f"slam/auc/{self.prior_type}"

        self.log.info(
            f"Initialized SLAM AUC threshold sweep with {self.prior_type} prior"
        )

    def _metric(
        self, *, num_potential_odors: int, num_high: int,
        num_osn: int, rep: int
    ) -> float:
        """Compute AUC for SLAM model using theta values."""
        # Configure logging for worker processes
        from ..simulator.parameters import SLAMParams, SLAMKParams
        from ..simulator.mld_rnn import SLAMSim
        from ..simulator.mld_rnn_ks import SLAMSimKST

        params_cls = SLAMParams if self.prior_type == "bernoulli" else SLAMKParams
        params = _build_params(
            self.cfg,
            params_cls,
            num_osn=num_osn,
            num_potential_odors=num_potential_odors,
            num_high=num_high,
        )

        self.log.info(
            f"Running SLAM AUC simulation for num_potential_odors={num_potential_odors}, num_high={num_high}, num_osn={num_osn}, rep={rep}")
        
        if self.prior_type == "bernoulli":
            _, theta = SLAMSim(params).simulate()
        else:
            _, theta = SLAMSimKST(params).simulate()
            
        # Compute AUC using ground truth labels (first num_high are true)
        labels = np.zeros(num_potential_odors, dtype=bool)
        labels[:num_high] = True
        score = float(roc_auc_score(labels, theta[:, -1]))
        return score

    def _get_save_filename(self, batch_idx: Optional[int] = None) -> str:
        if batch_idx is None:
            return f"{self.basename}_threshold_results.h5"
        return f"{self.basename}_threshold_results_batch{batch_idx}.h5"

    def _give_save_path(self) -> Path:
        """Create and return subdirectory path for saving results."""
        # Include sensing matrix information in the path
        matrix_info = _get_sensing_matrix_suffix(self.cfg)
        path = self.cfg.out_dir / self.subdir / matrix_info
        path.mkdir(parents=True, exist_ok=True)
        return path



