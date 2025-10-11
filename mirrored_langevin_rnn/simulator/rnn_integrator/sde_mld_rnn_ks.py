import torch
from ..parameters import SLAMKParams


def compute_sde_loop(
        *,
        sensing_matrix: torch.Tensor,
        osn_response: torch.Tensor,
        c_init: torch.Tensor,
        u_init: torch.Tensor,
        params: SLAMKParams,
        sdt_c: float,
        sdt_p: float,
):
    """
    Simulates the SDE for a given number of iterations.
    :returns: C_out, P_out, U_out, idx
    """
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

        dc, du, presence = sde_step_mld_ks_transformed(
            A=sensing_matrix,
            c=c,
            u=u,
            s_i=s_i,
            aa=float(params.aa),
            bb=float(params.bb),
            r0=float(params.r0),
            alpha=float(params.alpha),
            beta=float(params.beta),
            gamma_val=float(params.gamma_val),
            dt=float(params.dt),
            threshold=float(params.threshold),
            eps=float(params.eps),
            eps_trunc=float(params.eps_trunc),
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


# @torch.jit.script                       # comment this out while debugging
def sde_step_mld_ks_transformed(
        A: torch.Tensor,
        c: torch.Tensor,
        u: torch.Tensor,
        s_i: torch.Tensor,
        r0: float,
        alpha: float,
        beta: float,
        gamma_val: float,
        aa: float,
        bb: float,
        dt: float,
        threshold: float,
        eps: float = 1e-9,
        eps_trunc: float = 0.00005
):
    """
    One Euler step of the MLD RNN with transformed-KS prior on presence.
    """

    presence_preremap = torch.sigmoid(
        gamma_val * u)                         # (0,1)
    scale = 1.0 - 2.0 * eps_trunc                                # (1–2ε)
    # truncated support
    presence = eps_trunc + scale * presence_preremap

    # From now on the presence is remapped

    presence_gated = torch.where(presence < threshold,
                                 torch.tensor(
                                     1e-5, dtype=presence.dtype, device=presence.device),
                                 torch.tensor(1.0,  dtype=presence.dtype, device=presence.device))

    m1 = s_i / (r0 + torch.matmul(A, c * presence_gated))
    m2 = s_i / (r0 + torch.matmul(A, c * presence))

    # dc is unchanged
    dc = (
        presence_gated
        * (torch.matmul(A.t(), m1 - 1.0)
           + (alpha - 1.0) / (c + eps)
           - beta)
        * dt
    )

    base_grad = (((aa * bb - 1.0) * presence.pow(aa)) - aa + 1.0) / (
        presence * (presence.pow(aa) - 1.0))
    k_dist_term = scale * base_grad           # truncated-prior term

    du = (
        gamma_val
        * (
            presence_preremap * (1.0 - presence_preremap)
            * (c * torch.matmul(A.t(), m2 - 1.0) + k_dist_term)
            - 2.0 * presence_preremap            # same regulariser as before
            + 1)
    ) * dt

    return dc, du, presence
