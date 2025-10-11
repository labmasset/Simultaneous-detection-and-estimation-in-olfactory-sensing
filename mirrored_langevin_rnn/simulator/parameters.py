"""
Parameter classes for mirrored Langevin RNN olfaction simulations.

This module defines parameter dataclasses for different simulation models used in 
the odors sampling simulations.

Classes:
    SimulationParamsBase: Base class with common simulation parameters
    SLAMParams: Parameters specific to the SLAM model with Bernoulli priors
    SLAMKParams: Parameters for SLAM model with Kumaraswamy distribution priors
    PoissonParams: Parameters specific to the Poisson baseline model
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
import torch


@dataclass(slots=True)
class SimulationParamsBase:
    """
    Base class containing common parameters for all olfactory simulation models.

    This class defines shared parameters used by both the MLD/SLAM and Poisson baseline 
    models for odors sampling simulations. It handles basic
    simulation setup including network dimensions, sensing matrix configuration, data handling 
    and computational device selection.

    Attributes:
        num_osn (int): Number of olfactory sensory neurons (OSNs) in the simulation.
            Default: 300
        num_potential_odors (int): Total number of potential odor types that can be 
            represented. Default: 500
        sensing_matrix_type (Literal): Type of sensing matrix connecting odors to OSNs.
            Options: "dense_gamma", "sparse_binary", "sparse_gamma". Default: "sparse_binary"
        sensing_matrix_sparsity (float): Sparsity level of the sensing matrix (fraction of
            non-zero connections). Default: 0.1
        num_low (int): Number of odors present at low concentration. Default: 0
        num_high (int): Number of odors present at high concentration. Default: 15
        c_low (float): Low concentration value. Default: 10.0
        c_high (float): High concentration value. Default: 40.0
        r0 (float): Baseline firing rate parameter for neurons. Default: 1.0
        dt (float): Time step size for numerical integration. Default: 1e-5
        t_max (float): Maximum simulation time. Default: 0.1
        t_on_low (float): Time when low concentration odors are turned on. Default: 0.0
        t_on_high (float): Time when high concentration odors are turned on. Default: 0.0
        t_off (Optional[float]): Time when odors are turned off. If None, defaults to t_max.
        iter_num (int): Number of simulation iterations (computed from t_max/dt).
        sample_rate (int): Rate at which latent variables are recorded during simulation. Default: 2000
        eps (float): Small numerical epsilon for numerical stability. Default: 1e-5
        model (str): Type of model to use ("poisson" or "slam"). Default: "poisson"
        use_gpu (bool): Whether to attempt GPU acceleration. Default: False
        device (torch.device): Device on which tensors will be allocated
    """
    # simulator selection
    model: str
    
    device: torch.device

    # dimensionality of the simulation
    num_osn: int = 300
    num_potential_odors: int = 500

    # sensing matrix parameters
    sensing_matrix_type: Literal["dense_gamma",
                                 "sparse_binary", "sparse_gamma"] = "sparse_binary"
    sensing_matrix_sparsity: float = 0.1

    # simulated conditions
    num_low: int = 0
    num_high: int = 15
    c_low: float = 10.0
    c_med: float = 0
    c_high: float = 40.0

    # neuron parameters
    r0: float = 1.0

    # time parameters
    dt: float = 1e-5
    t_max: float = 0.1
    t_on_low: float = 0.0
    t_on_med: float = 0.0
    t_on_high: float = 0.0
    t_off: Optional[float] = None
    iter_num: int = field(init=False)

    # data recording rate
    sample_rate: int = 2000

    eps: float = 1e-5

    # device settings
    use_gpu: bool = False
     

    def __post_init__(self):
        """
        Post-initialization method to compute derived parameters.

        Sets t_off to t_max if not specified and computes the number of iterations
        needed for the simulation based on the time step and maximum time.
        """

        if self.t_off is None:
            self.t_off = self.t_max
        elif isinstance(self.t_off, str):
            self.t_off = float(self.t_off)

        self.iter_num = round(float(self.t_max) / float(self.dt))
        if isinstance(self.device, str):
            self.device = torch.device(self.device)


@dataclass(slots=True)
class SLAMParams(SimulationParamsBase):
    """
    Parameters specific to the MLD/SLAM with Bernoulli priors on presence.

    Attributes:
        gamma_val (float): Sigmoid steepness parameter controlling the sharpness of 
            the activation function. Default: 5
        w (float): Prior probability parameter for Bernoulli distribution. Default: 0.01
        alpha (float): Default: 5.0
        beta (float): Default: 0.1
        tau_c (float): Time constant for concentration dynamics. Default: 0.02
        tau_p (float): Time constant for presence dynamics. Default: 0.02
        threshold (float): Threshold value for Bernoulli prior decisions. Default: 0.2
        eps_denom (float): Small epsilon for numerical stability in denominators. 
            Default: 1e-3
        u_prior (float): Precomputed prior value for u variable, derived from w and 
            gamma_val during initialization.
    """

    # SLAM model's static parameters
    saveAll: bool = False  # whether to save all latent variables
    gamma_val: float = 5
    w: float = 0.01
    alpha: float = 5.0
    beta: float = 0.1
    tau_c: float = 0.02
    tau_p: float = 0.02

    threshold: float = 0.2  # threshold for bernoulli prior\

    # for distributed coding
    eps_denom: float = 1e-3

    # precompute prior for u
    # solve 1/(1 + exp(-gamma_val * x)) = w  =>  x = -ln((1-w)/w) / gamma_val
    u_prior: float = field(init=False)

    def __post_init__(self):
        """
        Post-initialization method for SLAM parameters.

        Calls parent post_init and computes the u_prior value by solving the equation:
        1/(1 + exp(-gamma_val * x)) = w  =>  x = -ln((1-w)/w) / gamma_val

        This precomputation optimizes the sigmoid transformation used in the model.
        """
        SimulationParamsBase.__post_init__(self)
        self.u_prior = -np.log((1 - self.w) / self.w) / self.gamma_val

@dataclass(slots=True)
class SLAMParamsCircuit(SLAMParams):
    """
    SLAM parameters for circuit-based implementation.
    """
    
    tau_h: float = 0.02
    
    def __post_init__(self):
        SLAMParams.__post_init__(self)

@dataclass(slots=True)
class SLAMKParams(SLAMParams):
    """
    Parameters for SLAM model with Kumaraswamy distribution priors.

    This class extends SLAMParams to use Kumaraswamy distribution instead of 
    Bernoulli priors. The Kumaraswamy distribution provides more flexible 
    modeling of sparse coding with continuous probability distributions.

    Attributes:
        aa (float): First shape parameter of the Kumaraswamy distribution. Default: 0.055
        r (float): Rate parameter used in computing the second shape parameter. 
            Default: 0.75
        bb (float): Second shape parameter of the Kumaraswamy distribution, computed
            from r and aa using the formula: log2(1-r) / log2(1-0.5^aa)
        eps_trunc (float): Truncation epsilon for numerical computations. 
            Default: 0.00005
        threshold (float): Threshold value for Kumaraswamy prior decisions. 
            Default: 0.5 (overrides the Bernoulli threshold from parent class)
    """
    # K-dist parameters
    aa: float = 0.055
    r = 0.75
    bb = np.log2(1 - r) / np.log2(1 - 0.5 ** aa)         # same as your code
    eps_trunc: float = 0.00005
    # bb: float = 0.5

    threshold: float = 0.5  # threshold for kumaraswamy prior

    def __post_init__(self):
        """
        Post-initialization method for SLAM with Kumaraswamy parameters.

        Calls parent post_init method. Note that u_prior computation is commented out
        as it may not be applicable for the Kumaraswamy distribution variant.
        """
        SLAMParams.__post_init__(self)
        # self.u_prior = -np.log((1 - self.w) / self.w) / self.gamma_val


@dataclass(slots=True)
class PoissonParams(SimulationParamsBase):
    """
    Parameters specific to the Poisson baseline model.

    Attributes:
        tauE (float): Time constant for excitatory neuron dynamics. Default: 0.020
        tauI (float): Time constant for inhibitory neuron dynamics. Default: 0.030
        lambda_val (float): Default: 1.0
        normFlag (Literal): Normalization method for the model outputs.
            Options: "max" (normalize by maximum) or "sum" (normalize by sum).
            Default: "max"
    """

    # poisson model's neuron parameters
    tauE: float = 0.020
    tauI: float = 0.030

    # poisson model's static parameters
    lambda_val: float = 1.0
    normFlag: Literal["max", "sum"] = "max"

    def __post_init__(self):
        """
        Post-initialization method for Poisson parameters.

        Calls the parent class post_init method to set up common simulation parameters.
        No additional computations are needed for the Poisson model parameters.
        """
        SimulationParamsBase.__post_init__(self)

@dataclass(slots=True)
class SLAMParamsL1(SLAMParams):
    """
    SLAM parameters with L1 prior.
    """
    
    lambda_para: float = 5.0
    
    def __post_init__(self):
        SLAMParams.__post_init__(self)