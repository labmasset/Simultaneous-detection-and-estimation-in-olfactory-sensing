import torch
import logging

from mirrored_langevin_rnn.simulator.sensory_scene.osn_response import (
    generate_osn_response_matrix,
    gen_true_time_series,
)
from .sensing_matrix import gen_sensing_matrix
from ..parameters import SimulationParamsBase

logger = logging.getLogger(__name__)

class SensoryScene:
    """
    A class for generating and managing sensory scenes for olfactory neural network simulations.
    
    This class creates sensory data including OSN (Olfactory Sensory Neuron) responses, 
    sensing matrices, and optionally ground truth concentration and presence matrices.
    """

    def __init__(self, params: SimulationParamsBase, *, include_ground_truth: bool = False):
        """
        Initialize the sensory scene using a dataclass of parameters.
        
        :param params: Parameters of the sampling simulation
        :param include_ground_truth: Whether to include ground truth concentration and presence matrices
        """
        self.params = params
        self.device = params.device
        self.include_ground_truth = include_ground_truth
        self._initialize_scene()

    def __getattr__(self, name):
        """
        Delegate attribute access to the params object for convenient parameter access.
        
        :param name: Name of the attribute to access
        :return: Value of the attribute from params
        :raises AttributeError: If attribute is not found in params
        """
        if hasattr(self.params, name):
            return getattr(self.params, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")
        
    def _initialize_scene(self):
        """
        Initialize the sensory scene: sensing matrix, 
        osn response and optionally ground truth time series.

        Creates the sensing matrix from simulation parameters and generates the complete
        sensory scene including OSN responses and optionally ground truth data.
        """
        np_sensing_matrix = gen_sensing_matrix(
            num_osn=self.num_osn,
            num_potential_odors=self.num_potential_odors,
            matrix_type=self.sensing_matrix_type,
            sparsity=self.sensing_matrix_sparsity,
        )
        self.sensing_matrix = torch.from_numpy(np_sensing_matrix).to(
            device=self.device, dtype=torch.float32
        )
        self.osn_response_mat, self.c_mat_true, self.p_mat_true = self.gen_sensory_scene()
        
    def gen_sensory_scene(self):
        """
        Generate sensory data and true concentration/presence time series using PyTorch.
        
        Creates OSN response matrices based on simulation parameters and optionally
        generates ground truth concentration and presence matrices.
        
        :return: Tuple of (OSN response matrix, true concentration matrix or None, true presence matrix or None)
        :raises ValueError: If timing parameters are invalid
        """
        
        # Validate timing parameters
        if not (self.t_on_low <= self.t_on_high <= self.t_off <= self.t_max):
            raise ValueError('Invalid onset and offset times')
        
        # Time vector spans the indices from 0 to t_max with step dt
        time_vector = torch.arange(0, self.t_max, self.dt, dtype=torch.float32, device=self.device)

        # Generate OSN response matrix
        osn_response_mat, c_mat, p_mat = generate_osn_response_matrix(
            num_potential_odors=self.num_potential_odors,
            num_low=self.num_low,
            num_high=self.num_high,
            c_low=self.c_low,
            c_med=self.c_med,
            c_high=self.c_high,
            time_vector=time_vector,
            t_on_low=self.t_on_low,
            t_on_med=self.t_on_med,
            t_on_high=self.t_on_high,
            t_off=self.t_off,
            sensing_matrix=self.sensing_matrix,
            device=self.device
        )
        # Generate true time series
        if self.include_ground_truth:
            c_mat_true, p_mat_true = gen_true_time_series(
                time_vector=time_vector,
                t_on_low=self.t_on_low,
                t_on_med=self.t_on_med,
                t_on_high=self.t_on_high,
                t_off=self.t_off,
                num_low=self.num_low,
                num_high=self.num_high,
                num_potential_odors=self.num_potential_odors,
                c_low=self.c_low,
                c_med=self.c_med,
                c_high=self.c_high,
                device=self.device,
                c_matrix=c_mat,
                p_matrix=p_mat,
            )

            # Subsample ground truth to match simulator output sampling
            sample_rate = int(self.sample_rate)
            indices = torch.arange(0, self.iter_num, sample_rate, device=self.device)
            # final_idx = self.iter_num - 1
            # if indices[-1] != final_idx:
                # indices = torch.cat([indices, torch.tensor([final_idx], device=self.device)])
            c_mat_true = c_mat_true[::sample_rate, :]
            p_mat_true = p_mat_true[::sample_rate, :]

            return osn_response_mat, c_mat_true, p_mat_true
        else:
            return osn_response_mat, None, None
    
    def get_sensory_scene_components(self):
        """
        Get the generated sensory scene components.
        
        :return: Tuple of (OSN response matrix, sensing matrix, true concentration matrix, true presence matrix)
                 Note: concentration and presence matrices are None if include_ground_truth is False
        """
        if self.include_ground_truth:
            return self.osn_response_mat, self.sensing_matrix, self.c_mat_true, self.p_mat_true
        else:
            return self.osn_response_mat, self.sensing_matrix, None, None
        
    def get_affinity_matrix(self):
        """
        Get the sensing matrix (affinity matrix) used in the sensory scene.
        
        :return: The sensing matrix as a PyTorch tensor
        """
        return self.sensing_matrix