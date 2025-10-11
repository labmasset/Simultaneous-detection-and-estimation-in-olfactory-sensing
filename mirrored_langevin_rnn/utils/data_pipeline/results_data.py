from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Optional, Sequence

@dataclass(slots=True)
class SimulationResult:
    nLow: int
    nHigh: int
    rep:  int
    C: np.ndarray
    c_true: float

    U: Optional[np.ndarray] = None
    Theta: Optional[np.ndarray] = None

    @property
    def key(self) -> str:
        return f"nLow={self.nLow}_nHigh={self.nHigh}_rep={self.rep}"
    
@dataclass(slots=True)
class GammaSimulationResult(SimulationResult):
    gamma_val: float = 0.0
    L1: float = 0.0
    runtime_sec: float = 0.0

    @property
    def key(self) -> str:
        return (f"nLow={self.nLow}_nHigh={self.nHigh}_rep={self.rep}"
                f"_gamma={self.gamma_val}")

@dataclass(slots=True)
class ThresholdResult:
    nOdor: int
    nSens: int
    threshold_nHigh: Optional[float]
    std_threshold: Optional[float]

    @property
    def key(self) -> str:
        return f"nOdor={self.nOdor}_nSens={self.nSens}"