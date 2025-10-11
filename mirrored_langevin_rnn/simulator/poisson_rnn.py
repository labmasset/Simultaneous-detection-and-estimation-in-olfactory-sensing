from mirrored_langevin_rnn.simulator.sensory_scene.scene import SensoryScene
from .rnn_integrator.sde_poisson_rnn import compute_sde_loop
from .parameters import PoissonParams
from mirrored_langevin_rnn.utils.visualization.dynamics_plot import DynamicsPlotter
import torch
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)



class PoissonCircuitSim:
    """
    Poisson circuit simulator implementing the baseline model from NeurIPS 2023.
    
    The simulator generates sensory scenes, computes projection matrices, and integrates
    the neural dynamics SDEs using Euler-Maruyama method.
    
    :param params_data: Configuration parameters for the simulation. If None, uses default PoissonParams.
    :type params_data: PoissonParams or None
    """

    def __init__(self, params_data: PoissonParams | None = None):
        self.params = params_data if params_data else PoissonParams()

    def __getattr__(self, name):
        if hasattr(self.params, name):
            return getattr(self.params, name)
        raise AttributeError(f"{type(self).__name__} object has no attribute {name}")

    def compute_projections(self, A):
        """
        Builds 1-to-1 projection matrix and normalizes according to normFlag.
        Returns AGnaive for excitatory/inhibitory integration.
        In 1-to-1 projection, each odor is mapped to a single granule cell.
        
        :param A: Sensing matrix (nSens x nOdor)
        
        :return: AGnaive, Gnaive
        - AGnaive: Projected sensing matrix (nSens x nGranule)
        - Gnaive: Projection matrix (nGranule x nOdor)
        """
        # Naive projection: identity
        Gnaive = torch.eye(self.num_potential_odors,
                           dtype=torch.float32, device=self.device)
        AGnaive = torch.matmul(A, Gnaive)

        # Choose normalization constant
        if self.normFlag == 'max':
            normConst = torch.max(torch.abs(AGnaive))
        elif self.normFlag == 'std':
            normConst = torch.std(AGnaive)
        elif self.normFlag == 'abs':
            normConst = torch.mean(torch.abs(AGnaive))
        else:
            raise ValueError(f"Unknown normFlag: {self.normFlag}")

        # Apply normalization
        AGnaive /= normConst
        Gnaive /= normConst
        return AGnaive, Gnaive

    def integrate(self, AG, G):
        """
        Runs Euler-Maruyama integration of poisson circuit dynamics in Neurips 2023 paper ().
        
        :param AG: Projected sensing matrix (nSens x nGranule)
        :param G: Projection matrix (nGranule x nOdor)
        :return: C_store: Concentration matrix (nGranule x nSamples)
        """
        nT = self.osn_response.shape[0]
        n_samples = (nT + self.sample_rate - 1) // self.sample_rate + 1  # Ceiling division + 1 for safety
        self.nGranule = self.num_potential_odors

        # Pre-allocate and initialize all tensors on device
        rExc = torch.ones(self.num_osn, dtype=torch.float32,
                          device=self.device) / self.r0
        rInh = torch.zeros(
            self.nGranule, dtype=torch.float32, device=self.device)
        C_store = torch.empty((self.nGranule, n_samples),
                              dtype=torch.float32, device=self.device)

        # Pre-compute constants once
        noise_scale = torch.sqrt(torch.tensor(
            2.0 * self.dt / self.tauI, device=self.device))

        # Pre-generate all random noise for better memory efficiency
        noise = torch.randn(nT, self.nGranule,
                            device=self.device) * noise_scale

        compute_sde_loop(
            osn_response=self.osn_response,
            AG=AG,
            G=G,
            rExc=rExc,
            rInh=rInh,
            C_store=C_store,
            noise=noise,
            params=self.params,
        )

        # Noise tensor no longer needed
        del noise
        torch.cuda.empty_cache()

        return C_store.cpu().numpy()


    def simulate(self, include_ground_truth=False, save_path=None):
        """
        Wraps the full simulation procedure for convenient execution.
        Generates the sensory scene, computes projections, and integrates the dynamics.
        
        :param include_ground_truth: Whether to output ground truth time series of concentration and presence.
        :return c_result: estimated concentration matrix (nGranule x nSamples).
        """
        start = time.time()
        sensory_scene = SensoryScene(
            self.params,
            include_ground_truth=include_ground_truth or save_path is not None,
        )
        self.osn_response, self.sensing_matrix, self.c_true, self.p_true = sensory_scene.get_sensory_scene_components()
        AG, G = self.compute_projections(self.sensing_matrix)
        c_result = self.integrate(AG, G)
        logger.info("Simulation finished in %.3fs", time.time() - start)

        if save_path is not None:
            out = {
                "C": c_result,
                "c_true": self.c_true.cpu().numpy() if self.c_true is not None else None,
                "p_true": self.p_true.cpu().numpy() if self.p_true is not None else None,
                "dt": self.dt,
                "sample_rate": self.sample_rate,
                "num_high": self.num_high,
                "num_low": self.num_low,
            }
            np.savez(save_path, **out)
            logger.info("Saved results to %s", save_path)

        # Free intermediate data to reduce memory footprint
        self.osn_response = None
        self.sensing_matrix = None
        self.c_true = None
        self.p_true = None
        del sensory_scene, AG, G
        torch.cuda.empty_cache()

        return c_result


if __name__ == "__main__":
    # Initialize and run
    params = PoissonParams(num_osn=500, num_potential_odors=1000, t_max=0.2)
    sim = PoissonCircuitSim(params_data=params)
    C = sim.simulate()

    # Plot dynamics
    dp = DynamicsPlotter(C=C, U=None, Theta=None,
                         cHigh=sim.cHigh, nHigh=sim.nHigh,
                         dt=sim.dt, sample_rate=1)
    dp.plot_concentration(save=True, filename="Poisson_dyn_concentration.png")
