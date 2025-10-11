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
    
    logit_w = float(torch.log(torch.tensor(params.w/(1.0-params.w))))
    
    # presence = torch.sigmoid(params.gamma_val * u)  # presence
    # presence_gated = torch.where(presence < params.threshold,
    #                              torch.tensor(
    #                                  1e-5, dtype=presence.dtype, device=presence.device),
    #                              torch.tensor(1.0,  dtype=presence.dtype, device=presence.device))
    # s_i = osn_response[0]
    
    # m = s_i / (params.r0 + torch.matmul(sensing_matrix, c * presence))
    # m_gated = s_i / (params.r0 + torch.matmul(sensing_matrix, c * presence_gated))
    m = torch.zeros(params.num_osn, dtype=torch.float32, device=params.device)
    m_gated = torch.zeros_like(m)  

    z =  (params.alpha - 1) / (c + params.eps)

    for i in range(iter_num):
        s_i = osn_response[i]

        dc, du, presence, dm, dm_gated, dz = sde_step_mld(
            A=sensing_matrix, 
            c=c, u=u, s_i=s_i,
            m=m,
            m_gated=m_gated,
            z=z,
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
        m += dm * params.dt / params.tau_h
        m_gated += dm_gated * params.dt / params.tau_h
        z += dz * params.dt / params.tau_h
        
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
                 m: torch.Tensor,
                 m_gated: torch.Tensor,
                 z: torch.Tensor,
                 s_i: torch.Tensor,
                 r0: float,
                 alpha: float,
                 beta: float,
                 gamma_val: float,
                 logit_w: float,
                 dt: float,
                 threshold: float,
                 eps: float):

    presence = torch.sigmoid(gamma_val * u)  # presence

    presence_gated = torch.where(presence < threshold,
                                 torch.tensor(
                                     1e-5, dtype=presence.dtype, device=presence.device),
                                 torch.tensor(1.0,  dtype=presence.dtype, device=presence.device))


    dc = (
        presence_gated
        * (torch.matmul(A.t(), m_gated - 1.0)
        #    + (alpha - 1.0) / (c + eps)
           - beta)
        * dt
    )

    du = (
        gamma_val
        * (
            presence * (1.0 - presence)
            * (c * torch.matmul(A.t(), m - 1.0) + logit_w)
            - 2.0 * presence
            + 1.0
        )
        * dt
    )
    
    dm = s_i - m * (
        r0 + torch.matmul(A, c * presence))
    dm_gated = s_i - m_gated * (
        r0 + torch.matmul(A, c * presence_gated))
    dz = (alpha - 1.0) - z * (c + eps)

    return dc, du, presence, dm, dm_gated, dz
