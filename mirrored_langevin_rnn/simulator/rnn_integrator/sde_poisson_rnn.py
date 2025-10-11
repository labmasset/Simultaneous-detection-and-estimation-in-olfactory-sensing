import torch
from ..parameters import PoissonParams


def compute_sde_loop(
    *,
    osn_response: torch.Tensor,
    AG: torch.Tensor,
    G: torch.Tensor,
    rExc: torch.Tensor,
    rInh: torch.Tensor,
    C_store: torch.Tensor,
    noise: torch.Tensor,
    params: PoissonParams,
):
    """
    Simulates the Poisson circuit SDE for a given number of time steps.
    """
    nT = osn_response.shape[0]
    n_samples = (nT + params.sample_rate - 1) // params.sample_rate  # Ceiling division
    dt_tau_E = params.dt / params.tauE
    dt_tau_I = params.dt / params.tauI
    idx = 0  # Use explicit index counter like MLD RNN
    for i in range(nT):
        # Use the JIT-compiled poisson_step function for core operations
        drExc, drInh, cEst = sde_step_poisson(
            osn_response[i],
            rExc,
            rInh,
            AG,
            G,
            float(params.r0),
            float(params.lambda_val),
            float(dt_tau_E),
            float(dt_tau_I),
        )

        # Update states with the computed drifts and add noise
        rExc += dt_tau_E * drExc
        rInh += dt_tau_I * drInh + noise[i]

        # Only store at sampling intervals or final step
        if (i % params.sample_rate) == 0 or i == nT - 1:
            # print(f"Iteration {i} completed", flush=True)
            C_store[:, idx] = cEst
            idx += 1


@torch.jit.script                      # comment out while debugging
def sde_step_poisson(
    s_i: torch.Tensor,            # (nSens,) current sensory input
    rExc: torch.Tensor,           # (nSens,) excitatory neuron activities
    rInh: torch.Tensor,           # (nGranule,) inhibitory neuron activities
    AG: torch.Tensor,             # (nSens, nGranule) projection matrix
    G: torch.Tensor,              # (nGranule, nOdor) projection matrix
    r0: float,                    # baseline firing rate
    lambda_val: float,            # regularization parameter
    dt_tau_E: float,              # dt/tau_E for excitatory update
    dt_tau_I: float               # dt/tau_I for inhibitory update
):
    """
    One Euler step of the Poisson RNN with 1-to-1 coding.

    Returns
    -------
    drExc : (nSens,) deterministic drift for excitatory neurons
    drInh : (nGranule,) deterministic drift for inhibitory neurons
    cEst  : (nOdor,) estimated concentration (already clamped to non-negative)
    """
    # Excitatory drive
    drExc = s_i - rExc * (r0 + torch.matmul(rInh, AG.T))

    # Inhibitory drive
    term1 = torch.matmul(rExc - 1.0, AG)
    term2 = lambda_val * torch.matmul(torch.sign(torch.matmul(rInh, G.T)), G)
    drInh = term1 - term2

    # Concentration estimate
    cEst = torch.matmul(rInh, G)
    cEst = torch.clamp(cEst, min=0.0)

    return drExc, drInh, cEst

