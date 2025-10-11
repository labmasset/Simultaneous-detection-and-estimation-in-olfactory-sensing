"""Data objects produced by experiment runs."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence


@dataclass(slots=True)
class SimulationResult:
    num_low: int
    num_high: int
    rep: int
    C: np.ndarray
    c_true: float

    U: Optional[np.ndarray] = None
    Theta: Optional[np.ndarray] = None

    @property
    def key(self) -> str:
        return f"num_low={self.num_low}_num_high={self.num_high}_rep={self.rep}"


@dataclass(slots=True)
class GammaSimulationResult(SimulationResult):
    gamma_val: float = 0.0
    L1: float = 0.0
    runtime_sec: float = 0.0

    @property
    def key(self) -> str:
        return f"num_low={self.num_low}_num_high={self.num_high}_rep={self.rep}_gamma={self.gamma_val}"


@dataclass(slots=True)
class ThresholdResult:
    num_potential_odors: int
    num_osn: int
    threshold_num_high: Optional[float]
    std_threshold: Optional[float]

    @property
    def key(self) -> str:
        return f"nOdor={self.num_potential_odors}_nSens={self.num_osn}"


@dataclass(slots=True)
class SparsitySimulationResult(SimulationResult):
    sparsity: float = 0.0

    @property
    def key(self) -> str:
        return (
            f"num_low={self.num_low}_num_high={self.num_high}_"
            f"sparsity={self.sparsity}_rep={self.rep}"
        )
