import time

from mirrored_langevin_rnn.simulator.sensory_scene.scene import SensoryScene
from .rnn_integrator.sde_mld_rnn_distributed import compute_sde_loop
from .mld_rnn import SLAMSim
from .parameters import SLAMParams
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)



class SLAMGeomSim(SLAMSim):
    """
    Geometry-aware SLAM simulator with geometry-aware coding.
    
    This simulator extends the base SLAM model by incorporating distributed coding
    with geometry-aware projection matrices. 
    
    :param params_data: Configuration parameters for SLAM simulation. If None, uses default SLAMParams.
    :type params_data: SLAMParams or None
    """

    def generate_distributing_matrix(self):
        """
        Generate the affinity matrix and the distributed projection matrix
        specifically for the geometry-aware SLAM model.
        """
        nOdor = self.num_potential_odors
        self.nGranule = 5 * nOdor
        sensing_matrix = self.sensing_matrix
    

        temp = torch.randn(self.nGranule, nOdor,
                           device=self.device, dtype=torch.float32)
        Q_full, _ = torch.linalg.qr(temp)     # Q_full: (nGranule, nOdor)
        Q = Q_full.T                          # (nOdor, nGranule)

        # 4) C = A^T A, normalized so trace(C)=nOdor
        C = sensing_matrix.T @ sensing_matrix               # (nOdor, nOdor)
        trC = torch.trace(C)
        eps = torch.finfo(C.dtype).eps
        if torch.abs(trC) > eps:
            C = C * (nOdor / trC)
        else:
            logger.warning(
                "Trace of C is close to zero. Skipping normalization.")

        # 5) spectral transform B = U diag(1/sqrt(λ+a)) U^T
        # C = eigvecs @ diag(eigvals) @ eigvecs.T
        eigvals, eigvecs = torch.linalg.eigh(C)
        inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + float(self.a)))
        B = eigvecs @ inv_sqrt @ eigvecs.T         # (nOdor, nOdor)

        # 6) geometry matrix Ggeom = B Q
        Ggeom = B @ Q                              # (nOdor, nGranule)

        # 7) normalize so that max|A Ggeom| = 1
        AG = sensing_matrix @ Ggeom                       # (nSens, nGranule)
        norm_const = AG.abs().max()
        Gmat_torch = Ggeom / norm_const
        self.Gmat = Gmat_torch
        return 
    

    def sde_integrate(self,saveAll=False):
        
        device = self.device

        A = self.sensing_matrix
        G = self.Gmat
        
        logit_w = float(np.log(self.w / (1.0 - self.w)))

        g = torch.zeros(G.shape[1], device=device)             # (nGranule,)
        u = torch.full((self.nOdor,), self.u_prior,
                       device=device, dtype=torch.float32)

        # initialize latent variables
        c0 = torch.tensor(np.random.gamma(6, 4, self.nOdor),
                          device=device, dtype=torch.float32)
        if self.nOdor >= 4:
            c0[:4] = torch.tensor([60., 1., 10., 20.],
                                  device=device, dtype=torch.float32)
            
        pinvG = torch.linalg.pinv(G)
        g.copy_(torch.matmul(pinvG, c0)) # assign value to g

        # prelocate
        n_samples = (self.iter_num + self.sample_rate - 1) // self.sample_rate
        C_store = torch.empty(self.nOdor, n_samples + 1, device=device)
        Theta_store = torch.empty_like(C_store)
        U_store = torch.empty_like(C_store) if saveAll else None
        G_store = torch.empty(
            G.shape[1], n_samples + 1, device=device) if saveAll else None

        sdt_c = torch.sqrt(torch.tensor(
            2.0 * self.dt / self.tau_c, device=device))
        sdt_p = torch.sqrt(torch.tensor(
            2.0 * self.dt / self.tau_p, device=device))

        compute_sde_loop(
            params=self.params,
            saveAll=saveAll,
            A=A,
            G=G,
            osn_response=self.osn_response,
            logit_w=logit_w,
            g=g,
            u=u,
            c=c0,
            C_store=C_store,
            Theta_store=Theta_store,
            U_store=U_store,
            G_store=G_store,
            sdt_c=sdt_c,
            sdt_p=sdt_p,
        )

        if saveAll:
            return (C_store.cpu().numpy(),
                    Theta_store.cpu().numpy(),
                    U_store.cpu().numpy(),
                    G_store.cpu().numpy())
        return (C_store.cpu().numpy(),
                Theta_store.cpu().numpy())

    

    def simulate(self, include_ground_truth=False, save_path=None):
        start = time.time()
        sensory_scene = SensoryScene(
            self.params,
            include_ground_truth=include_ground_truth or save_path is not None,
        )
        self.osn_response, self.sensing_matrix, self.c_true, self.p_true = sensory_scene.get_sensory_scene_components()
        self.generate_distributing_matrix()
        results = self.sde_integrate(saveAll=self.saveAll)
        logger.info("Simulation finished in %.3fs", time.time() - start)

        if save_path is not None:
            out = {
                "C": results[0],
                "Theta": results[1],
                "c_true": self.c_true.cpu().numpy() if self.c_true is not None else None,
                "p_true": self.p_true.cpu().numpy() if self.p_true is not None else None,
                "dt": self.dt,
                "sample_rate": self.sample_rate,
                "num_high": self.num_high,
            }
            if self.saveAll and len(results) > 2:
                out["U"] = results[2]
                out["G"] = results[3]
            np.savez(save_path, **out)
            logger.info("Saved results to %s", save_path)

        return results


class SLAMDistSim(SLAMGeomSim):
    """
    SLAM simulator with naive-distributed coding.
    
    :param params_data: Configuration parameters for SLAM simulation. If None, uses default SLAMParams.
    :type params_data: SLAMParams or None
    """

    def generate_distributing_matrix(self):
        """
        Generate the affinity matrix and the distributed projection matrix
        specifically for the geometry-aware SLAM model.
        """
        nOdor = self.num_potential_odors
        self.nGranule = 5 * nOdor
        sensing_matrix = self.sensing_matrix
    

        temp = torch.randn(self.nGranule, nOdor,
                           device=self.device, dtype=torch.float32)
        Q_full, _ = torch.linalg.qr(temp)     # Q_full: (nGranule, nOdor)
        Q = Q_full.T                          # (nOdor, nGranule)

        # 4) C = A^T A, normalized so trace(C)=nOdor
        C = sensing_matrix.T @ sensing_matrix               # (nOdor, nOdor)
        trC = torch.trace(C)
        eps = torch.finfo(C.dtype).eps
        if torch.abs(trC) > eps:
            C = C * (nOdor / trC)
        else:
            logger.warning(
                "Trace of C is close to zero. Skipping normalization.")

        # 5) spectral transform B = U diag(1/sqrt(λ+a)) U^T
        # C = eigvecs @ diag(eigvals) @ eigvecs.T
        eigvals, eigvecs = torch.linalg.eigh(C)
        inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + float(self.a)))
        B = eigvecs @ inv_sqrt @ eigvecs.T         # (nOdor, nOdor)

        Gdist = Q                              # (nOdor, nGranule)

        # 7) normalize so that max|A Ggeom| = 1
        AG = sensing_matrix @ Gdist                       # (nSens, nGranule)
        norm_const = AG.abs().max()
        Gmat_torch = Gdist / norm_const
        self.Gmat = Gmat_torch
        return 


if __name__ == "__main__":
    from mirrored_langevin_rnn.utils.visualization.dynamics_plot import DynamicsPlotter

    testing_param = SLAMParams(num_osn=300, num_potential_odors=1000, t_max=0.1, sample_rate=1, num_high=5,
                               sensing_matrix_type="sparse_binary")
    slam_sim = SLAMGeomSim(params_data=testing_param)
    # slam_sim = SLAMGeomSim(params_data=testing_param)
    results = slam_sim.simulate(saveAll=True)

    C, Theta, U, Gr = results
    dp = DynamicsPlotter(C=C, U=U, Theta=Theta, cHigh=slam_sim.cHigh,
                         nHigh=slam_sim.nHigh, dt=slam_sim.dt,
                         sample_rate=slam_sim.sample_rate)
    dp.plot_concentration(
        save=True, filename="SLAM_geom_dyn_concentration.png")
    dp.plot_presence(save=True, filename="SLAM_geom_dyn_presence.png")
    dp.plot_probability(save=True, filename="SLAM_geom_dyn_probability.png")

    slam_sim_dist = SLAMDistSim(testing_param)
    results = slam_sim_dist.simulate(saveAll=True)

    C, Theta, U, Gr = results
    dp = DynamicsPlotter(C=C, U=U, Theta=Theta, cHigh=slam_sim_dist.cHigh,
                         nHigh=slam_sim_dist.nHigh, dt=slam_sim_dist.dt,
                         sample_rate=slam_sim_dist.sample_rate)
    dp.plot_concentration(
        save=True, filename="SLAM_dyn_dist_concentration.png")
    dp.plot_presence(save=True, filename="SLAM_dyn_dist_presence.png")
    dp.plot_probability(save=True, filename="SLAM_dyn_dist_probability.png")
