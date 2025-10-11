"""Experiment configuration dataclasses."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

import torch

import numpy as np


def _dict_to_range(range_dict: Dict[str, int]) -> tuple[int, ...]:
    """Convert a dict with 'start', 'stepsize', 'end' keys to a range tuple.

    Args:
        range_dict: Dictionary containing 'start', 'stepsize', and 'end' keys

    Returns:
        Tuple of integers representing the range
    """
    start = range_dict.get("start")
    stepsize = range_dict.get("stepsize")
    end = range_dict.get("end")
    return tuple(range(start, end + 1, stepsize))


def _dict_to_float_range(range_dict: Dict[str, float]) -> tuple[float, ...]:
    """Convert a dict with 'start', 'stepsize', 'end' keys to a float range tuple.

    Args:
        range_dict: Dictionary containing 'start', 'stepsize', and 'end' keys

    Returns:
        Tuple of floats representing the range
    """
    start = range_dict.get("start")
    stepsize = range_dict.get("stepsize")
    end = range_dict.get("end")
    return tuple(np.arange(start, end + stepsize/2, stepsize))


def _dict_to_logspace_range(range_dict: Dict[str, Any]) -> tuple[int, ...]:
    """Convert a dict with 'start', 'end', 'num' keys to a log space range tuple.

    Args:
        range_dict: Dictionary containing 'start', 'end', and 'num' keys

    Returns:
        Tuple of integers representing the log space range
    """
    start = range_dict.get("start")
    end = range_dict.get("end")
    num = range_dict.get("num")
    return tuple(sorted({int(round(x)) for x in np.logspace(np.log10(start), np.log10(end), num)}))


Self = TypeVar("Self", bound="BaseSweepConfig")


@dataclass(slots=True, kw_only=True)
class ExperimentConfigBase(ABC):
    """Common configuration fields for all experiments."""

    model: str

    repeats: int
    batch_size: int

    seed: int | None

    c_high: float
    c_low: float

    # time parameters
    dt: float
    t_max: float
    t_on_low: float
    t_on_high: float
    t_off: Optional[float] = None
    iter_num: int = field(init=False)

    sensing_matrix_type: str
    sensing_matrix_sparsity: float

    # torch device on which simulations will run
    device: torch.device
    workers: int = field(
        default_factory=lambda: int(
            os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        )
    )
    out_dir: Path = field(default_factory=lambda: Path.cwd() / "data")

    def __post_init__(self) -> None:
        if self.t_off is None:
            self.t_off = self.t_max
        self.iter_num = round(float(self.t_max) / float(self.dt))
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def to_json(self) -> str:
        d = asdict(self)
        d["device"] = str(self.device)
        d["out_dir"] = str(self.out_dir)
        return json.dumps(d, indent=2)


@dataclass(slots=True)
class BaseSweepConfig(ExperimentConfigBase, ABC, Generic[Self]):
    """Base class for parameter sweep configurations."""

    @property
    def combos(self) -> List[Tuple[Any, ...]]:
        return self._generate_combos()

    @abstractmethod
    def _generate_combos(self) -> List[Tuple[Any, ...]]:
        ...


@dataclass(slots=True)
class GammaSweepConfig(BaseSweepConfig):
    """Parameter grid for the ``gamma_val`` sweep."""

    gamma_values: Sequence[float]
    num_high: int

    def __post_init__(self) -> None:
        ExperimentConfigBase.__post_init__(self)

    def _generate_combos(self) -> List[Tuple[float, int]]:
        return [
            (g, rep) for g in self.gamma_values for rep in range(self.repeats)
        ]


@dataclass(slots=True)
class PresentSweepConfig(BaseSweepConfig):
    """Experiment parameters for the N-present sweep."""

    num_osn: int
    num_potential_odors: int

    num_low_values: Dict[str, int]
    num_high_values: Dict[str, int]
    sample_rate: int = 1000

    # Simulation parameters that will be passed through to the simulator
    # These are made optional with defaults to maintain compatibility
    threshold: float = 0.2
    eps: float = 1e-5
    eps_denom: float = 1e-3
    alpha: float = 5.0
    beta: float = 0.1
    tau_c: float = 0.02
    tau_p: float = 0.02
    tau_h: float = 0.02
    w: float = 0.01
    gamma_val: float = 5.0

    def __post_init__(self) -> None:
        # Explicitly call ExperimentConfigBase's __post_init__
        ExperimentConfigBase.__post_init__(self)
        self.num_low_values = _dict_to_range(self.num_low_values)
        self.num_high_values = _dict_to_range(self.num_high_values)

    def _generate_combos(self) -> List[Tuple[int, int, int]]:
        return [
            (nL, nH, rep)
            for nL in self.num_low_values
            for nH in self.num_high_values
            for rep in range(self.repeats)
        ]


@dataclass(slots=True, kw_only=True)
class ThresholdSweepConfig(BaseSweepConfig):
    """Parameters for the threshold sweep experiments."""

    out_dir: Path = field(
        default_factory=lambda: Path.cwd() / "data" / "threshold_sweep")

    # num_potential_odors_values: Sequence[int] = tuple(
    #     sorted({int(round(x)) for x in np.logspace(np.log10(1000), np.log10(16000), 16)})
    # )
    # num_osn_values: Sequence[int] = tuple(range(100, 801, 50))

    metric: str
    
    num_potential_odors_values: Dict[str, Any]  # Changed to Any to support both linear and log space configs
    num_osn_values: Dict[str, int]

    num_high_values: Dict[str, int]

    error_threshold: float

    sensing_matrix_type: str
    sensing_matrix_sparsity: float

    def __post_init__(self) -> None:
        BaseSweepConfig.__post_init__(self)
        
        # Convert dict configurations to ranges
        # Check if num_potential_odors_values uses log space (has 'num' key) or linear space (has 'stepsize' key)
        if 'num' in self.num_potential_odors_values:
            self.num_potential_odors_values = _dict_to_logspace_range(self.num_potential_odors_values)
        else:
            self.num_potential_odors_values = _dict_to_range(self.num_potential_odors_values)
        
        self.num_osn_values = _dict_to_range(self.num_osn_values)
        self.num_high_values = _dict_to_range(self.num_high_values)

    def _generate_combos(self) -> List[Tuple[int, int]]:
        return [
            (o, s) for o in self.num_potential_odors_values for s in self.num_osn_values
        ]

    @property
    def n_odor_values(self) -> Sequence[int]:
        return self.num_potential_odors_values

    @property
    def n_sens_values(self) -> Sequence[int]:
        return self.num_osn_values

    @property
    def n_high_values(self) -> Sequence[int]:
        return self.num_high_values


@dataclass(slots=True, kw_only=True)
class SparsitySweepConfig(BaseSweepConfig):
    """Sweep over sensing matrix sparsity while keeping OSN and odor counts fixed."""
    num_osn: int
    num_potential_odors: int
    
    num_low_values: Dict[str, int]
    num_high_values: Dict[str, int]
    sparsity_values: Dict[str, float]

    # Threshold search options
    error_threshold: float = 5.0
    use_binary_search: bool = False

    sensing_matrix_type: str = "sparse_binary"
    # placeholder for avoiding missing args for __init__
    sensing_matrix_sparsity: float = 0.1
    out_dir: Path = field(
        default_factory=lambda: Path.cwd() / "data" / "sparsity_sweep")

    def __post_init__(self) -> None:
        BaseSweepConfig.__post_init__(self)

        # Convert dict configurations to ranges
        self.num_low_values = _dict_to_range(self.num_low_values)
        self.num_high_values = _dict_to_range(self.num_high_values)
        self.sparsity_values = _dict_to_float_range(self.sparsity_values)

    def _generate_combos(self) -> List[Tuple[int, Optional[int], float, int]]:
        if self.use_binary_search:
            return [
                (nL, None, s, rep)
                for s in self.sparsity_values
                for nL in self.num_low_values
                for rep in range(self.repeats)
            ]
        else:
            return [
                (nL, nH, s, rep)
                for s in self.sparsity_values
                for nL in self.num_low_values
                for nH in self.num_high_values
                for rep in range(self.repeats)
            ]
