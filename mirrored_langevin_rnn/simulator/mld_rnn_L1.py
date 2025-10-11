from matplotlib.pylab import f
from .mld_rnn import SLAMSim
from .sensory_scene.scene import SensoryScene
from .rnn_integrator.sde_mld_rnn_L1_prior import compute_sde_loop
from .parameters import SLAMParamsL1
import torch
import numpy as np
import time
import os
from typing import Literal
import logging


# setup logger
logger = logging.getLogger(__name__)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


torch.set_num_threads(1)


class SLAMSimL1(SLAMSim):
    """
    SLAM simulator with exponential distribution priors on concentration variables.

    :param params_data: Configuration parameters with exponential-specific settings. If None, uses default SLAMParamsL1.
    :type params_data: SLAMParamsL1 or None
    """

    def __init__(self, params_data: SLAMParamsL1 | None = None):
        self.params = params_data if params_data else SLAMParamsL1()

    def sde_integrate(self, saveU = False):
        """
        Run MAP estimation on sensor data osn_response (nT x nSens).
        Uses either standard bernoulli prior or Kumaraswamy prior based on self.prior_type.
        """
        device = self.device

        # For both prior types, initialize concentration with gamma distribution
        # c = torch.tensor(np.random.gamma(6, 4, self.num_potential_odors),
        #                  device=device, dtype=torch.float32)
        c = torch.full((self.num_potential_odors,), 0.1, device=device, dtype=torch.float32)

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
        logging.debug(f"iter_num: {self.iter_num}, sample_rate: {self.sample_rate}, "
                      f"num_odors: {self.num_potential_odors}, num_osn: {self.num_osn}, "
                      f"r0: {self.r0}, alpha: {self.alpha}, beta: {self.beta}, "
                      f"dt: {self.dt}, threshold: {self.threshold}, eps: {self.eps}, "
                      f"saveU: {self.saveAll}")

        C_store, P_store, U_store, _ = compute_sde_loop(
            sensing_matrix=self.sensing_matrix,
            osn_response=self.osn_response,
            c_init=c,
            u_init=u,
            params=self.params,
            sdt_c=sdt_c,
            sdt_p=sdt_p,
        )

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

    # Example with standard bernoulli prior
    logger.info("Running simulation with bernoulli prior...")
    slam_sim = SLAMSimL1(params_data=SLAMParamsL1(num_potential_odors=1000,
                                                  num_osn=300, t_max=0.2,
                                                  sample_rate=100,
                                                  sensing_matrix_type="sparse_binary",
                                                  device="cpu",
                                                  model="mld_rnn_L1",
                                                  saveAll=True))
    results = slam_sim.simulate()

    # Unpack outputs
    C = results[0]
    P = results[1]
    U = results[2]

    # Initialize the plotter
    dp = DynamicsPlotter(C=C, U=U, Theta=P, cHigh=slam_sim.c_high,
                         nHigh=slam_sim.num_high,
                         dt=slam_sim.dt,
                         sample_rate=slam_sim.sample_rate)

    # Plot and/or save
    import matplotlib.pyplot as plt
    fig1, ax1 = dp.plot_concentration()
    fig2, ax2 = dp.plot_presence()
    fig3, ax3 = dp.plot_probability()
    plt.show()
    # dp.plot_presence(
    #     save=True, filename="SLAM_dyn_naive_kumaraswamy_presence.png")
    # dp.plot_probability(
    #     save=True, filename="SLAM_dyn_naive_kumaraswamy_probability.png")
