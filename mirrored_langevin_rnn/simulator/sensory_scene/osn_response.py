from calendar import c
import torch


def generate_osn_response_matrix(*, num_potential_odors, num_low, num_high, c_low, c_med, c_high,
                                 time_vector, t_on_low, t_on_med, t_on_high, t_off, sensing_matrix, device='cpu'):
    """
    Generate OSN response matrix by expanding Poisson matrix in time.

    :param num_potential_odors: Total number of potential odors
    :param num_low: Number of low concentration odors
    :param num_high: Number of high concentration odors
    :param c_low: Low concentration value
    :param c_med: Medium concentration value
    :param c_high: High concentration value
    :param t_on_low: Onset time for low concentration
    :param t_on_med: Onset time for medium concentration
    :param t_on_high: Onset time for high concentration
    :param t_off: Offset time
    :param time_vector: Time vector spanning the indices
    :param sensing_matrix: Sensing matrix (num_osn x num_potential_odors)
    :param device: torch.device

    :return: OSN response matrix (nT x num_osn)
    """
    c_matrix = _gen_concentration_matrix(
        num_potential_odors, num_low, num_high, c_low, c_med, c_high, t_on_med, device=device)
    p_matrix = _gen_presence_matrix(num_potential_odors, num_low, num_high, t_on_med,
                                    device=device)
    poisson_mat = _gen_poisson_response(
        sensing_matrix=sensing_matrix,
        concentration_matrix=c_matrix,
        presence_matrix=p_matrix,
        r0=1.0,  # Baseline firing rate
        device=device
    )
    t = time_vector
    if t_on_med == 0:
        osn_response_mat = (
            (t < t_on_low)[:, None] * poisson_mat[0, :] +
            ((t >= t_on_low) & (t < t_on_high))[:, None] * poisson_mat[1, :] +
            ((t >= t_on_high) & (t < t_off))[:, None] * poisson_mat[2, :] +
            (t >= t_off)[:, None] * poisson_mat[3, :]
        )
    else:
        osn_response_mat = (
            (t < t_on_low)[:, None] * poisson_mat[0, :] +
            ((t >= t_on_low) & (t < t_on_med))[:, None] * poisson_mat[1, :] +
            ((t >= t_on_med) & (t < t_on_high))[:, None] * poisson_mat[2, :] +
            ((t >= t_on_high) & (t < t_off))[:, None] * poisson_mat[3, :] +
            (t >= t_off)[:, None] * poisson_mat[4, :]
        )
    return osn_response_mat, c_matrix, p_matrix


def gen_true_time_series(*, time_vector, t_on_low, t_on_med, t_on_high, t_off,
                         num_low, num_high, num_potential_odors,
                         c_low, c_med, c_high, device='cpu',
                         c_matrix=None, p_matrix=None):
    """
    Generate true concentration and presence time series of odors.

    :param time_vector: Time vector
    :param t_on_low: Onset time for low concentration
    :param t_on_med: Onset time for medium concentration
    :param t_on_high: Onset time for high concentration
    :param t_off: Offset time
    :param num_low: Number of low concentration odors
    :param num_high: Number of high concentration odors
    :param num_potential_odors: Total number of potential odors
    :param c_low: Low concentration value
    :param c_high: High concentration value
    :param device: Device to run on
    :return: c_mat_true: True concentration matrix (nT x nOdor), 
             p_mat_true: True presence matrix (nT x nOdor)
    """
    t = time_vector
    t_length = len(t)
    if t_on_med == 0:
        # Generate the true concentration matrix (nT x nOdor) by expanding concentration vectors at four time phases.
        c_mat_true = torch.hstack([
            c_low * ((t >= t_on_low) & (t <= t_off))[:, None].float() * torch.ones(
                (t_length, num_low), dtype=torch.float32, device=device),
            c_high * ((t >= t_on_high) & (t <= t_off))[:, None].float() * torch.ones(
                (t_length, num_high), dtype=torch.float32, device=device),
            torch.zeros((t_length, num_potential_odors - num_low -
                        num_high), dtype=torch.float32, device=device)
        ])

        # Generate the true presence matrix (nT x nOdor) by expanding presence vectors at four time phases.
        p_mat_true = torch.hstack([
            ((t >= t_on_low) & (t <= t_off))[:, None].float(
            ) * torch.ones((t_length, num_low), dtype=torch.float32, device=device),
            ((t >= t_on_high) & (t <= t_off))[:, None].float(
            ) * torch.ones((t_length, num_high), dtype=torch.float32, device=device),
            torch.zeros((t_length, num_potential_odors - num_low -
                        num_high), dtype=torch.float32, device=device)
        ])
        return c_mat_true, p_mat_true # (nT x nOdor)
    else:
        t_vec_1 = torch.ones(len(t[t < t_on_low]), dtype=torch.float32, device=device)
        t_vec_2 = torch.ones(len(t[(t >= t_on_low) & (t < t_on_med)]), dtype=torch.float32, device=device)
        t_vec_3 = torch.ones(len(t[(t >= t_on_med) & (t < t_on_high)]), dtype=torch.float32, device=device)
        t_vec_4 = torch.ones(len(t[(t >= t_on_high) & (t < t_off)]), dtype=torch.float32, device=device)
        t_vec_5 = torch.ones(len(t[t >= t_off]), dtype=torch.float32, device=device)

        c_mat_1 = torch.outer(t_vec_1, c_matrix[0, :])
        c_mat_2 = torch.outer(t_vec_2, c_matrix[1, :])
        c_mat_3 = torch.outer(t_vec_3, c_matrix[2, :])
        c_mat_4 = torch.outer(t_vec_4, c_matrix[3, :])
        c_mat_5 = torch.outer(t_vec_5, c_matrix[4, :])

        c_mat_true = torch.vstack([c_mat_1, c_mat_2, c_mat_3, c_mat_4, c_mat_5])

        p_mat_1 = torch.outer(t_vec_1, p_matrix[0, :])
        p_mat_2 = torch.outer(t_vec_2, p_matrix[1, :])
        p_mat_3 = torch.outer(t_vec_3, p_matrix[2, :])
        p_mat_4 = torch.outer(t_vec_4, p_matrix[3, :])
        p_mat_5 = torch.outer(t_vec_5, p_matrix[4, :])
        
        p_mat_true = torch.vstack([p_mat_1, p_mat_2, p_mat_3, p_mat_4, p_mat_5])

        return c_mat_true, p_mat_true # (nT x nOdor)


def _gen_concentration_matrix(num_potential_odors, num_low, num_high, c_low, c_med, c_high, t_on_med, device='cpu'):
    """
    Generate concentration matrix (4 x nOdor) for different time phases.

    :param num_potential_odors: Total number of potential odors
    :param num_low: Number of low concentration odors
    :param num_high: Number of high concentration odors  
    :param c_low: Low concentration value
    :param c_high: High concentration value
    :param device: Device to run on
    :return: Concentration matrix (4 x nOdor) representing true concentration at four stages of the simulation.
    """
    # concentration vectors for each time phase
    if c_med == 0 and t_on_med == 0:
        # phase 1: no odors present
        row1 = torch.zeros(num_potential_odors, dtype=torch.float32, device=device)
        # phase 2: low concentration odors present
        row2 = torch.cat([
            torch.full((num_low,), c_low, dtype=torch.float32, device=device),
            torch.zeros(num_potential_odors - num_low,
                        dtype=torch.float32, device=device)
        ])
        # phase 3: low and high concentration odors present
        row3 = torch.cat([
            torch.full((num_low,), c_low, dtype=torch.float32, device=device),
            torch.full((num_high,), c_high, dtype=torch.float32, device=device),
            torch.zeros(num_potential_odors - num_low - num_high,
                        dtype=torch.float32, device=device)
        ])
        # phase 4: no odors present
        row4 = torch.zeros(num_potential_odors, dtype=torch.float32, device=device)

        return torch.vstack([row1, row2, row3, row4])
    
    else:
        print(c_low, c_med, c_high)
        # phase 1: no odors present
        row1 = torch.zeros(num_potential_odors, dtype=torch.float32, device=device)
        # phase 2: low concentration odors present
        row2 = torch.cat([
            torch.full((num_low,), c_med, dtype=torch.float32, device=device),
            torch.full((num_high,), c_low, dtype=torch.float32, device=device),
            torch.zeros(num_potential_odors - num_high - num_low,
                        dtype=torch.float32, device=device)
        ])
        # phase 3: med concentration
        row3 = torch.cat([
            torch.full((num_low,), c_high, dtype=torch.float32, device=device),
            torch.full((num_high,), c_med, dtype=torch.float32, device=device),
            torch.zeros(num_potential_odors - num_low - num_high,
                        dtype=torch.float32, device=device)
        ])
        # phase 4: high concentration
        row4 = torch.cat([
            torch.full((num_low,), c_low, dtype=torch.float32, device=device),
            torch.full((num_high,), c_high, dtype=torch.float32, device=device),
            torch.zeros(num_potential_odors - num_low - num_high,
                        dtype=torch.float32, device=device)
        ])
        
        # phase 5: no odors present
        row5= torch.zeros(num_potential_odors, dtype=torch.float32, device=device)

        return torch.vstack([row1, row2, row3, row4, row5])


def _gen_presence_matrix(num_potential_odors, num_low, num_high, t_med, device='cpu'):
    """
    Generate presence matrix (4 x nOdor): ones if odor is present, zeros otherwise.

    :param num_potential_odors: Total number of potential odors
    :param num_low: Number of low concentration odors
    :param num_high: Number of high concentration odors
    :param device: Device to run on
    :return: Presence matrix (4 x nOdor) representing four stages of the simulation.
    """
    # presence vectors for each time phase
    # phase 1: no odors present
    row1_p = torch.zeros(num_potential_odors,
                        dtype=torch.float32, device=device)
    # phase 2: low concentration odors present
    row2_p = torch.cat([
        torch.ones(num_low, dtype=torch.float32, device=device),
        torch.ones(num_high, dtype=torch.float32, device=device),
        torch.zeros(num_potential_odors - num_high - num_low,
                    dtype=torch.float32, device=device)
    ])
    # phase 3: low and high concentration odors present
    row3_p = torch.cat([
        torch.ones(num_low, dtype=torch.float32, device=device),
        torch.ones(num_high, dtype=torch.float32, device=device),
        torch.zeros(num_potential_odors - num_low - num_high,
                    dtype=torch.float32, device=device)
    ])
    # phase 4: no odors present
    row4_p = torch.zeros(num_potential_odors,
                        dtype=torch.float32, device=device)
    if t_med == 0:
        return torch.vstack([row1_p, row2_p, row3_p, row4_p])
    else: 
        return torch.vstack([row1_p, row2_p, row3_p, row3_p, row4_p])

def _gen_poisson_response(sensing_matrix, concentration_matrix, presence_matrix, r0, device='cpu'):
    """
    Generate Poisson response from sensing matrix and odor information.

    :param sensing_matrix: Sensing matrix (num_osn x num_potential_odors)
    :param concentration_matrix: Concentration matrix (4 x nOdor)
    :param presence_matrix: Presence matrix (4 x nOdor) 
    :param r0: Baseline firing rate
    :param device: Device to run on
    :return: Poisson sampled matrix (4 x num_osn) representing the OSN response at four different time phases.
    """
    # Convert sensing matrix to tensor if needed
    if not isinstance(sensing_matrix, torch.Tensor):
        A = torch.tensor(sensing_matrix, dtype=torch.float32, device=device)
    else:
        A = sensing_matrix.to(device)
    
    # Calculate Poisson rate parameter
    lambda_mat = r0 + (concentration_matrix *
                       presence_matrix) @ A.T 

    # Sample from Poisson distribution
    poisson_mat = torch.poisson(lambda_mat)

    return poisson_mat
