from __future__ import annotations

from mirrored_langevin_rnn.experiment.experiment_base import SweepExperimentBase

"""Helpers to construct experiment instances based on config objects."""

from .parameters import (
    ExperimentConfigBase,
    PresentSweepConfig,
    GammaSweepConfig,
    ThresholdSweepConfig,
    SparsitySweepConfig,
)
from .present_sweep import (
    SLAMCircuitNPresentSweep,
    SLAMDistNPresentSweep,
    SLAMGeomNPresentSweep,
    SLAMNaiveNPresentSweep,
    SLAMKSNPresentSweep,
    PoissonNPresentSweep,
)
from .gamma_sweep import GammaSteepnessSweep
from .threshold_sweep import PoissonL1ThresholdSweep, PoissonRankThresholdSweep, SLAMAUCThresholdSweep, SLAML1ThresholdSweep
from .sparsity_sweep import (
    PoissonSparsitySweep,
    SLAMSparsitySweep,
    SLAMKSSparsitySweep,
    SLAMGeomSparsitySweep,
    SLAMDistSparsitySweep,
)


_PRESENT_EXPERIMENT = {
    ("poisson"): PoissonNPresentSweep,
    ("slam"): SLAMNaiveNPresentSweep,
    ("slam_circuit"): SLAMCircuitNPresentSweep,
    ("slam_ks"): SLAMKSNPresentSweep,
    ("slam_geom"): SLAMGeomNPresentSweep,
    ("slam_dist"): SLAMDistNPresentSweep,
}

_THRESHOLD_EXPERIMENT = {
    ("poisson", "l1"): PoissonL1ThresholdSweep,
    ("poisson", "roc"): PoissonRankThresholdSweep,
    ("slam", "l1"): SLAML1ThresholdSweep,
    ("slam", "roc"): SLAMAUCThresholdSweep,
    ("slam_ks", "l1"): SLAML1ThresholdSweep,
    ("slam_ks", "roc"): SLAMAUCThresholdSweep,
}

_SPARSITY_EXPERIMENT = {
    "poisson": PoissonSparsitySweep,
    "slam": SLAMSparsitySweep,
    "slam_ks": SLAMKSSparsitySweep,
    "slam_geom": SLAMGeomSparsitySweep,
    "slam_dist": SLAMDistSparsitySweep,
}

Experiment = (
    SLAMNaiveNPresentSweep
    | SLAMCircuitNPresentSweep
    | SLAMKSNPresentSweep
    | SLAMGeomNPresentSweep
    | SLAMDistNPresentSweep
    | PoissonNPresentSweep
    | GammaSteepnessSweep
    | PoissonL1ThresholdSweep
    | PoissonRankThresholdSweep
    | SLAML1ThresholdSweep
    | SLAMAUCThresholdSweep
    | PoissonSparsitySweep
    | SLAMSparsitySweep
    | SLAMKSSparsitySweep
    | SLAMGeomSparsitySweep
    | SLAMDistSparsitySweep
)


def create_experiment(cfg: ExperimentConfigBase) -> Experiment:
    """
    Instantiate the appropriate experiment for ``cfg``.

    """
    if isinstance(cfg, PresentSweepConfig):
        key = cfg.model.lower()
        return _PRESENT_EXPERIMENT[key](cfg)
    
    if isinstance(cfg, GammaSweepConfig):
        return GammaSteepnessSweep(cfg)
    
    if isinstance(cfg, ThresholdSweepConfig):
        key = (cfg.model.lower(), cfg.metric.lower())
        return _THRESHOLD_EXPERIMENT[key](cfg)

    if isinstance(cfg, SparsitySweepConfig):
        key = cfg.model.lower()
        return _SPARSITY_EXPERIMENT[key](cfg)
    
    raise ValueError(f"Unsupported experiment config: {type(cfg).__name__}")


