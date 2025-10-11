import torch
from ..parameters import SLAMParams


# @torch.jit.script
def compute_sde_loop(
        *,
        params: SLAMParams,
        saveAll: bool,
        A: torch.Tensor,
        G: torch.Tensor,
        osn_response: torch.Tensor,
        logit_w: float,
        g: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
        C_store: torch.Tensor,
        Theta_store: torch.Tensor,
        U_store: torch.Tensor,
        G_store: torch.Tensor,
        sdt_c: float,
        sdt_p: float,
):
    """
    Simulates the SDE for a given number of iterations.
    
    :returns: None, but updates C_store, Theta_store, U_store, and G_store tensors.
    """
    iter_num = int(params.iter_num)
    sample_rate = int(params.sample_rate)
    sample_idx = 0
    for i in range(iter_num):
        s_i = osn_response[i]

        dg, du, theta = sde_geom_step(
            A,
            G,
            g,
            u,
            c,
            s_i,
            r0=float(params.r0),
            alpha=float(params.alpha),
            beta=float(params.beta),
            gamma_=float(params.gamma_val),
            logit_w=logit_w,
            dt=float(params.dt),
            threshold=float(params.threshold),
            eps_clamp=float(params.eps_clamp),
            eps_denom=float(params.eps_denom),
        )

        g += dg / params.tau_c + sdt_c * torch.randn_like(g)
        u += du / params.tau_p + sdt_p * torch.randn_like(u)
        c = torch.clamp(torch.matmul(G, g), min=params.eps_clamp)

        if (i % sample_rate) == 0 or i == iter_num - 1:
            C_store[:, sample_idx] = c
            Theta_store[:, sample_idx] = theta
            if saveAll:
                U_store[:, sample_idx] = u
                G_store[:, sample_idx] = g
            sample_idx += 1


@torch.jit.script                      # comment out while debugging
def sde_geom_step(
    A:      torch.Tensor,              # (nSens , nOdor)
    G:      torch.Tensor,              # (nOdor, nGranule)
    g:      torch.Tensor,              # (nGranule,)
    u:      torch.Tensor,              # (nOdor,)
    c:      torch.Tensor,              # (nOdor,) current concentration
    s_i:    torch.Tensor,              # (nSens,)
    r0:     float,
    alpha:  float,
    beta:   float,
    gamma_: float,
    logit_w: float,
    dt:     float,
    threshold: float,
    eps_clamp:    float,
    eps_denom: float,
    device: str = "cpu"
):
    """
    One Euler step of the MLD RNN with distributed coding.
    Returns
    ------
    dg   : (nGranule,)  deterministic drift for g  (noise is added outside)
    du   : (nOdor   ,)  deterministic drift for u
    c    : (nOdor   ,)  concentration  = G @ g    (after current update)
    theta: (nOdor   ,)  presence, sigmoid(gamma * u)
    """

    presence = torch.sigmoid(gamma_ * u.to(device))

    presence_gated = torch.where(
        presence < threshold,
        torch.tensor(1e-5, dtype=presence.dtype, device=presence.device),
        torch.tensor(1.0, dtype=presence.dtype, device=presence.device)
    )

    m1 = s_i / (r0 + torch.matmul(A, c * presence_gated))
    m2 = s_i / (r0 + torch.matmul(A, c * presence))

    rhs1 = presence_gated * (
        torch.matmul(A.t(), m1 - 1.0)
        + (alpha - 1.0) / (c + eps_denom)
        - beta
    )

    dg = torch.matmul(G.t(), rhs1) * dt

    du = gamma_ * (
        presence * (1.0 - presence)
        * (c * torch.matmul(A.t(), m2 - 1.0) + logit_w)
        - 2.0 * presence + 1.0
    ) * dt

    return dg, du, presence
