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
        z = x_k + (np.dot(tau, np.dot(C.T, (y - np.dot(C, x_k)))))
        x_k_1 = IST(z, gamma)
        l_2_norm = linalg.norm(x_k_1 - x_k, ord=2)
        x_k = x_k_1
        num_iterations += 1
    #calculating the support of x
    support = np.where(x_k_1 != 0)[0]
    return x_k_1, support, num_iterations

def DISTA(z_0, G, Q, tau, lam, y):
    z_k = z_0
    delta = 10**(-8)
    q = y.shape[0]
    print("q: ", q)
    n = z_0.shape[1]
    num_iterations = 0
    while True:
        l_2_norm = 0
        z_k_1 = np.zeros_like(z_k)
        for i in range(q):
            cons = np.zeros((n,))
            grad = np.dot(tau,np.dot(G[i].T, (y[i] - np.dot(G[i], z_k[i]))))
            for j in range(q):
                cons += np.dot(Q[i,j], z_k[j])
            merge = np.sum([grad, cons], axis=0)
            z_k_1[i] = IST(merge, tau * lam)
        
        # set to zero all the values of the last q elements of z_k_1 if they are smaller than 0.002
        # z_k_1[:,q:] = np.where(z_k_1[:,q:] < 0.002, 0, z_k_1[:,q:])
        
        # Stopping criterion where the norm of the difference between z_k_1 and z_k is less than 10^-8 for every i in q
        for i in range(q):
            l_2_norm += np.linalg.norm(z_k_1[i] - z_k[i])
        print("l_2_norm: ", l_2_norm)
        if l_2_norm < delta:
            break
        # calculate the support of z
        
        z_k = z_k_1.copy()
        num_iterations += 1

    support = np.where(z_k_1 != 0)[0]
    
    return z_k, support, num_iterations
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