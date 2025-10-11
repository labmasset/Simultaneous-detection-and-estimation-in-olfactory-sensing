"""Factory utilities for choosing the correct simulator."""

from __future__ import annotations

from .parameters import SLAMParams, SLAMKParams, PoissonParams, SimulationParamsBase, SLAMParamsCircuit, SLAMParamsL1
from .mld_rnn import SLAMSim
from .mld_rnn_ks import SLAMSimKST
from .mld_rnn_circuit import SLAMSimCircuit
from .mld_rnn_distributed import SLAMGeomSim, SLAMDistSim
from .poisson_rnn import PoissonCircuitSim
from .mld_rnn_L1 import SLAMSimL1
Simulator = (
    SLAMSim | SLAMSimKST | SLAMGeomSim | SLAMDistSim | PoissonCircuitSim | SLAMSimCircuit | SLAMSimL1
)

SimulatorParams = (
    SLAMParams | SLAMKParams | PoissonParams | SimulationParamsBase | SLAMParamsCircuit | SLAMParamsL1
)


def create_simulator(params: SimulatorParams) -> Simulator:
    """
    Return a new simulator using the provided parameter dataclass.

    Parameters
    ----------
    params:
        Dataclass instance containing simulation parameters.
    
    Returns
    -------
    object
        Concrete simulator ready for :py:meth:`simulate`.
    """

    model = getattr(params, "model").lower()

    match model:
        case "slam":
            return SLAMSim(params)
        case "slam_ks":
            return SLAMSimKST(params)
        case "slam_circuit":
            return SLAMSimCircuit(params)
        case "slam_geom":
            return SLAMGeomSim(params)
        case "slam_dist":
            return SLAMDistSim(params)
        case "poisson":
            return PoissonCircuitSim(params)
        case "slam_l1":
            return SLAMSimL1(params)
        case _:
            raise ValueError(f"Unknown model name: {model}")
