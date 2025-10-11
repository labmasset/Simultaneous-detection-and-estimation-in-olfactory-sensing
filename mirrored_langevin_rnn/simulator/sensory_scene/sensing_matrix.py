import numpy as np
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)

def gen_sensing_matrix(num_osn, num_potential_odors, matrix_type: Literal["dense_gamma", "sparse_gamma", "sparse_binary"] = "dense_gamma",
                       sparsity: float = 0.1, gamma_alpha: float = 0.37, gamma_beta: float = 0.36) -> np.ndarray:
    """
    Returns a sensing matrix of shape (num_osn, num_potential_odors) with the specified type and sparsity.
    :param num_osn: Number of OSNs (Olfactory Sensory Neurons).
    :param num_potential_odors: Number of potential odors.
    :param matrix_type: Type of the sensing matrix. Options are "dense_gamma", "sparse_gamma", "sparse_binary".
    :param sparisity: Sparsity level for sparse matrices (between 0 and 1).
    :param gamma_alpha: Alpha parameter for the gamma distribution.
    :param gamma_beta: Beta parameter for the gamma distribution.
    :return: A sensing matrix of shape (num_osn, num_potential_odors).
    """

    if matrix_type == "dense_gamma":
        sensing_matrix = _affinity_matrix_dense_gamma(num_osn, num_potential_odors, gamma_alpha, gamma_beta)
    elif matrix_type == "sparse_binary":
        sensing_matrix = _affinity_matrix_sparse_binary(num_osn, num_potential_odors, sparsity)
    elif matrix_type == "sparse_gamma":
        sensing_matrix = _affinity_matrix_sparse_gamma(num_osn, num_potential_odors, sparsity, gamma_alpha, gamma_beta)
    return sensing_matrix

def _affinity_matrix_dense_gamma(num_osn, num_potential_odors, gamma_alpha, gamma_beta):
    """
    Generates a dense affinity matrix using a gamma distribution.
    Normalize by the maximum value in each row.
    :param num_osn: Number of OSNs.
    :param num_potential_odors: Number of potential odors.
    :param gamma_alpha: Alpha parameter for the gamma distribution.
    :param gamma_beta: Beta parameter for the gamma distribution.
    :return: A dense affinity matrix of shape (num_osn, num_potential_odors).
    """
    gamma_matrix = np.random.gamma(gamma_alpha, gamma_beta, (num_osn, num_potential_odors))
    normalization_factor = np.max(gamma_matrix, axis=1, keepdims=True)
    sensing_matrix = gamma_matrix / normalization_factor
    return sensing_matrix

def _affinity_matrix_sparse_binary(num_osn, num_potential_odors, sparsity=0.1):
    """
    Generates a sparse binary affinity matrix.
    :param num_osn: Number of OSNs.
    :param num_potential_odors: Number of potential odors.      
    :param sparsity: Sparsity level (between 0 and 1).
    :return: A sparse binary affinity matrix of shape (num_osn, num_potential_odors).
    """
    logger.debug(f"Generating sparse binary sensing matrix with sparsity {sparsity}")
    sensing_matrix = (np.random.rand(num_osn, num_potential_odors) < sparsity).astype(float)
    return sensing_matrix

def _affinity_matrix_sparse_gamma(num_osn, num_potential_odors, sparsity, gamma_alpha, gamma_beta):
    """
    Generates a sparse affinity matrix using a gamma distribution.
    :param num_osn: Number of OSNs.
    :param num_potential_odors: Number of potential odors.
    :param sparsity: Sparsity level (between 0 and 1).
    :return: A sparse affinity matrix of shape (num_osn, num_potential_odors).
    """
    gamma_matrix = np.random.gamma(shape=gamma_alpha, scale=gamma_beta, size=(num_osn, num_potential_odors))
    sparsity_filter = (np.random.rand(num_osn, num_potential_odors) < sparsity).astype(float)
    sensing_matrix = gamma_matrix * sparsity_filter
    sensing_matrix = sensing_matrix / np.maximum(np.max(sensing_matrix, axis=1, keepdims=True), 1e-6)  # Normalize by the max value in each row, avoid division by zero
    return sensing_matrix

