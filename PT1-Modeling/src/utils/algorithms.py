import pandas as pd
import numpy as np
from numpy import linalg

# ISTA algorithm returns the estimated x and its support
def ISTA(x_0, C, tau, lam, y):
    """
    Implements the Iterative Soft Thresholding Algorithm (ISTA) for solving the L1-regularized least squares problem.

    Parameters:
    - x_0: Initial state.
    - C: Coefficient matrix.
    - tau: Step size parameter.
    - lam: Regularization parameter.
    - y: Measurements vector.

    Returns:
    - x_k_1: The estimated x vector.
    - support: The support of the estimated x vector (indices of non-zero elements).
    - num_iterations: The number of iterations performed.

    """
    x_k = x_0
    l_2_norm = 1
    gamma = tau * lam
    num_iterations=0
    
    while (l_2_norm >= (10**(-12))):
        # Note the inverted sign in the update step compared to the original report
        # Note that A is considered as the identity matrix here
        z = x_k + (np.dot(tau, np.dot(C.T, (y - np.dot(C, x_k)))))
        x_k_1 = IST(z, gamma)
        l_2_norm = linalg.norm(x_k_1 - x_k)
        x_k = x_k_1
        num_iterations += 1
    #calculating the support of x
    support = np.where(x_k_1 != 0)[0]
    return x_k_1, support, num_iterations

def IST(x, gamma):
    """
    Implements the Iterative Soft Thresholding (IST).

    Parameters:
        x : Input array.
        gamma : Threshold value.

    Returns:
        x : Output array after applying the IST algorithm.
    """
    return np.where(np.abs(x) > gamma, np.sign(x) * (np.abs(x) - gamma), 0)

# ISTA algorithm returns the estimated x and its support
def ISTA_task_5(x_0, C, tau, lam, y):
    """
    Implements the Iterative Soft Thresholding Algorithm (ISTA) for solving the L1-regularized least squares problem.

    Parameters:
    - x_0: Initial state.
    - C: Coefficient matrix.
    - tau: Step size parameter.
    - lam: Regularization parameter.
    - y: Measurements vector.

    Returns:
    - x_k_1: The estimated x vector.
    - support: The support of the estimated x vector (indices of non-zero elements).
    - num_iterations: The number of iterations performed.

    """
    x_k = x_0
    stop_criteria = 1
    gamma = tau * lam
    num_iterations=0
    estimates_history = []
    
    while (stop_criteria >= (10**(-8))):
        # Note the inverted sign in the update step compared to the original report
        # Note that A is considered as the identity matrix here
        z = x_k + (np.dot(tau, np.dot(C.T, (y - np.dot(C, x_k)))))
        x_k_1 = IST(z, gamma)
        stop_criteria = np.sum(np.linalg.norm(x_k_1 - x_k,2)**2)
        estimates_history.append(np.copy(x_k_1))
        x_k = x_k_1
        num_iterations += 1
    #calculating the support of x
    support = np.where(x_k_1 != 0)[0]
    return x_k_1, support, num_iterations, estimates_history