import torch


from ..parameters import SLAMParams


def compute_sde_loop(
    *,
    sensing_matrix: torch.Tensor,
    osn_response: torch.Tensor,
    c_init: torch.Tensor,
    u_init: torch.Tensor,
    params: SLAMParams,
    sdt_c: float,
    sdt_p: float,
):
    c = c_init.clone()
    u = u_init.clone()
    iter_num = int(params.iter_num)
    sample_rate = int(params.sample_rate)
    n_samples = (iter_num + sample_rate - 1) // sample_rate
    C_out = torch.empty(c.shape[0], n_samples + 1, device=c.device)
    P_out = torch.empty_like(C_out)
    U_out = torch.empty_like(C_out)

    idx = 0

    for i in range(iter_num):
        s_i = osn_response[i]

        logit_w = float(torch.log(torch.tensor(params.w/(1.0-params.w))))
        dc, du, presence = sde_step_mld(
            sensing_matrix, c, u, s_i,
            r0=float(params.r0),
            alpha=float(params.alpha),
            beta=float(params.beta),
            gamma_val=float(params.gamma_val),
            logit_w=logit_w,
            dt=float(params.dt),
            threshold=float(params.threshold),
            eps=float(params.eps)
        )

        # Update concentration and latent variables with computed drifts plus noise
        c += dc / params.tau_c + sdt_c * torch.randn_like(c)
        u += du / params.tau_p + sdt_p * torch.randn_like(u)
        c.clamp_(min=params.eps)

        # Store results at regular intervals and the final step
        if (i % sample_rate) == 0 or i == iter_num - 1:
            C_out[:, idx] = c
            P_out[:, idx] = presence
            U_out[:, idx] = u
            idx += 1

    return C_out, P_out, U_out, idx


@torch.jit.script                    # comment this out while debugging
def sde_step_mld(A: torch.Tensor,
                 c: torch.Tensor,
                 u: torch.Tensor,
                 s_i: torch.Tensor,
                 r0: float,
                 alpha: float,
                 beta: float,
                 gamma_val: float,
                 logit_w: float,
                 dt: float,
                 threshold: float,
                 eps: float):
    """
    One step simulation of the MLD RNN with distributed coding.

    A        : (nSens, nOdor) affinity matrix
    c, u     : (nOdor,) current state vectors (concentration, latent)
    s_i      : (nSens,) Poisson sample at time i
    Remaining args are scalar hyper-parameters.
    Returns   dc, du, presence   (all (nOdor,) tensors, **no noise added**).
    """
    presence = torch.sigmoid(gamma_val * u)  # presence

    presence_gated = torch.where(presence < threshold,
                                 torch.tensor(
                                     1e-5, dtype=presence.dtype, device=presence.device),
                                 torch.tensor(1.0,  dtype=presence.dtype, device=presence.device))

    m1 = s_i / (r0 + torch.matmul(A, c * presence_gated))
    m2 = s_i / (r0 + torch.matmul(A, c * presence))

    dc = (
        presence_gated
        * (torch.matmul(A.t(), m1 - 1.0)
           + (alpha - 1.0) / (c + eps)
           - beta)
        * dt
    )

    du = (
        gamma_val
        * (
            presence * (1.0 - presence)
            * (c * torch.matmul(A.t(), m2 - 1.0) + logit_w)
            - 2.0 * presence
            + 1.0
        )
        * dt
    )

    return dc, du, presence
