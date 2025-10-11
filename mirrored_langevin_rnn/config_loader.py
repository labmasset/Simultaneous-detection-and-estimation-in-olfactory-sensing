"""Utilities for loading YAML configuration files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Any, Dict, Union

import yaml  # type: ignore

# Set MPS fallback before importing torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

from .experiment.parameters import (
    PresentSweepConfig,
    GammaSweepConfig,
    ThresholdSweepConfig,
    SparsitySweepConfig,
)
from .simulator.parameters import (
    SLAMParams,
    SLAMKParams,
    PoissonParams,
    SLAMParamsCircuit,
    SLAMParamsL1
)

ExperimentParams = PresentSweepConfig | GammaSweepConfig | ThresholdSweepConfig | SparsitySweepConfig
SimulationParams = SLAMParams | SLAMKParams | PoissonParams | SLAMParamsL1

@dataclass(slots=True)
class SystemConfig:
    """
    Dataclass tracking the environment level configuration.

    :param use_gpu: Whether to use GPU for computations.
    :param torch_num_threads: Number of threads for PyTorch to use. (intra-op parallelism)
    :param log_level: Logging verbosity (DEBUG, INFO, etc.)
    :param log_file: Optional log file name passed to ``setup_logging``
    """

    use_gpu: bool = True
    torch_num_threads: int = 1
    log_level: str = "INFO"
    log_file: str | None = None

    def apply(self) -> torch.device:
        """
        Apply the system configuration to the environment.
        """
        torch.set_num_threads(self.torch_num_threads)
        
        if self.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        elif self.use_gpu and torch.backends.mps.is_available():
            os.PYTORCH_ENABLE_MPS_FALLBACK=1 
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        return device


def load_config(path: str | Path) -> Tuple[str, SystemConfig, Any]:
    """
    Load a configuration YAML file and return the instantiated objects.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A tuple containing:
        - kind: 'experiment' or 'simulation' indicating the type of configuration.
          The former is for running sweep experiments, the latter for single simulations.
        - system: An instance of SystemConfig with system-level parameters.
        - payload: An instance of the appropriate configuration class based on the kind.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    system = SystemConfig(**raw.get("system", {}))
    kind = raw.get("kind").lower()

    if kind == "experiment":
        exp_cfg = _build_experiment(raw["experiment"])
        return "experiment", system, exp_cfg
    if kind == "simulation":
        sim_params = _build_simulation(raw["simulation"])
        return "simulation", system, sim_params
    raise ValueError(f"Unknown config kind: {kind}")


def _build_experiment(cfg: Dict[str, Any]) -> ExperimentParams:
    """
    Build an experiment configuration object from a configuration dictionary.

    Args:
        cfg: Configuration dictionary containing experiment parameters.
             Must contain a 'type' key specifying the experiment type.
             Optional keys include:
             - 'params': Dictionary of experiment-specific parameters
             - 'model': Model type
             - 'metric': Metric type (optional, used for threshold sweep)

    Returns:
        An experiment configuration object of the appropriate type based on cfg['type'].
        Supported types: PresentSweepConfig, GammaSweepConfig, ThresholdSweepConfig, 
        SparsitySweepConfig.

    Raises:
        KeyError: If cfg['type'] is not one of the supported experiment types.
    """
    exp_type = cfg["type"].lower()
    params = cfg.get("params")
    model = cfg.get("model")
    device = cfg.get("device")
    if isinstance(device, str):
        device = torch.device(device)
    
    if exp_type == "threshold_sweep":
        metric = cfg.get("metric")
    else:
        metric = None
        
    experiment_choice = {
        "present_sweep": PresentSweepConfig,
        "gamma_sweep": GammaSweepConfig,
        "threshold_sweep": ThresholdSweepConfig,
        "sparsity_sweep": SparsitySweepConfig,
    }
    # return an appropriate experiment config dataclass initialized with the parameters given
    if metric is not None:
        return experiment_choice[exp_type](
            model=model,
            metric=metric,
            device=device if device is not None else torch.device("cpu"),
            **params,
        )
    else: 
        return experiment_choice[exp_type](
            model=model,
            device=device if device is not None else torch.device("cpu"),
            **params
        )


def _build_simulation(cfg: Dict[str, Any]) -> SimulationParams:
    """
    Build a simulation configuration object from a configuration dictionary.

    :param cfg: Configuration dictionary containing simulation parameters.
    :return: An instance of SimulatorParams dataclass: SLAMParams, SLAMKParams, or PoissonParams, 
    based on the 'model' key.
    """
    model = cfg["model"].lower()
    params = cfg.get("params")  # dictionary of parameters specific to the model
    device = cfg.get("device")
    if isinstance(device, str):
        device = torch.device(device)
    simulator_choice = {
        "slam": SLAMParams,
        "slam_ks": SLAMKParams,
        "slam_circuit": SLAMParamsCircuit,
        "slam_geom": SLAMParams,
        "slam_dist": SLAMParams,
        "poisson": PoissonParams,
        "slam_l1": SLAMParamsL1,
    }

    # return an appropriate params dataclass initialized with the parameters given
    return simulator_choice[model](
        model=model, device=device if device is not None else torch.device("cpu"), **(params or {})
    )
