from matplotlib import pyplot as plt
from mirrored_langevin_rnn.simulator.mld_rnn import SLAMSim
from .rnn_integrator.sde_mld_rnn_circuit import compute_sde_loop
from .parameters import SLAMParamsCircuit
import numpy as np
import os
import logging

# setup logger
logger = logging.getLogger(__name__)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch

torch.set_num_threads(1)


class SLAMSimCircuit(SLAMSim):
    """
    SLAM simulator with Bernoulli priors and circuit implementation.
    
    :param params_data: Configuration parameters for SLAM simulation. If None, uses default SLAMParams.
    :type params_data: SLAMParams or None
    """

    def __init__(self, params_data: SLAMParamsCircuit | None = None):
        self.params = params_data if params_data else SLAMParamsCircuit()
        logger.info(f"Simulation completed with u_prior: {self.u_prior}")

    def __getattr__(self, name):
        if hasattr(self.params, name):
            return getattr(self.params, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    def sde_integrate(self, saveU=False):
        """
        Performs Euler-Maruyama integration of the mirrored Langevin dynamics
        to sample from the posterior distribution over odor concentrations and
        presence variables given the OSNs responses.
        
        :param saveU: Whether to save and return the presence variable trajectories
        :type saveU: bool
        :return: Tuple of (C_store, P_store) or (C_store, P_store, U_store) if saveU=True
        :rtype: tuple of numpy.ndarray
        """
        device = self.device
        # Use self.sensing_matrix directly as it's already a tensor
        sensing_matrix = self.sensing_matrix

        # For both prior types, initialize concentration with gamma distribution
        c = torch.tensor(np.random.gamma(6, 4, self.num_potential_odors),
                         device=device, dtype=torch.float32)

        # Set first four odors to specific values if we have enough odors
        # if self.num_potential_odors >= 4:
        #     c[:4] = torch.tensor([60.0, 1.0, 10.0, 20.0], device=device)

        # Initialize latent variable u with the prior value
        u = torch.full_like(c, self.u_prior, dtype=torch.float32)

        # Pre-compute noise scale factors
        sdt_c = torch.sqrt(torch.tensor(
            2.0 * self.dt / self.tau_c, device=device))
        sdt_p = torch.sqrt(torch.tensor(
            2.0 * self.dt / self.tau_p, device=device))

        # Only needed for bernoulli prior
        logit_w = float(np.log(self.w / (1.0 - self.w)))

        logging.debug(f"iter_num: {self.iter_num}, sample_rate: {self.sample_rate}, "
                      f"num_odors: {self.num_potential_odors}, num_osn: {self.num_osn}, "
                      f"r0: {self.r0}, alpha: {self.alpha}, beta: {self.beta}, "
                      f"dt: {self.dt}, threshold: {self.threshold}, eps: {self.eps}, "
                      f"logit_w: {logit_w}, saveU: {saveU}")

        # Call the compiled function
        C_store, P_store, U_store, idx = compute_sde_loop(
            sensing_matrix=sensing_matrix,
            osn_response=self.osn_response,
            c_init=c,
            u_init=u,
            params=self.params,
            sdt_c=sdt_c,
            sdt_p=sdt_p,
        )

        torch.cuda.empty_cache()

        # Convert tensors to NumPy arrays before returning
        if saveU:
            return (C_store.cpu().numpy(),
                    P_store.cpu().numpy(),
                    U_store.cpu().numpy())
        return (C_store.cpu().numpy(),
                P_store.cpu().numpy())


if __name__ == "__main__":
    from mirrored_langevin_rnn.utils.visualization.dynamics_plot import DynamicsPlotter

    from ..logging_utils import setup_logging

    setup_logging("DEBUG")
    logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
    
    torch.device("cpu")

    # Example with standard bernoulli prior
    logger.info("Running simulation with bernoulli prior...")
    slam_sim = SLAMSimCircuit(params_data=SLAMParamsCircuit(num_potential_odors=500,
                                              model="slam", saveAll=True, device=torch.device("cpu"),
                                              num_osn=300, t_max=1, t_off=0.8, t_on_high=0.2,
                                              sample_rate=100, eps=1e-3,
                                              sensing_matrix_type="sparse_binary"))
    results = slam_sim.simulate(save_path="data/simulations/simulation_slam.npz",
                                include_ground_truth=True)

    # Unpack outputs
    C = results[0]
    P = results[1]
    U = results[2]

    # Initialize the plotter
    dp = DynamicsPlotter(C=C, U=U, Theta=P, cHigh=slam_sim.c_high,
                         nHigh=slam_sim.num_high,
                         dt=slam_sim.dt,
                         sample_rate=slam_sim.sample_rate)

    fig, ax = dp.plot_concentration()
    fig1, ax1 = dp.plot_presence()
    fig2, ax2 = dp.plot_probability()
    plt.show()