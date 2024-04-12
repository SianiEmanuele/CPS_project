import pandas as pd
import numpy as np
from numpy import linalg

# ISTA algorithm returns the estimated x and its support
def ISTA(x_0, C, tau, lam, y):
    x_k = x_0
    l_2_norm = 1
    gamma = tau * lam
    num_iterations=0
    # TODO: Remove
    print("C SHAPE: ", C.shape)
    print("y SHAPE: ", y.shape)
    print("x_0 SHAPE: ", x_0.shape)
    print("tau shape: ", tau.shape) 
    
    while (l_2_norm >= (10**(-12))):
        z = x_k + (np.dot(tau, np.dot(C.T, (y - np.dot(C, x_k)))))
        x_k_1 = IST(z, gamma)
        l_2_norm = linalg.norm(x_k_1 - x_k)
        x_k = x_k_1
        num_iterations += 1
    #calculating the support of x
    support = np.where(x_k_1 != 0)[0]
    return x_k_1, support, num_iterations

def IST(x, gamma): 
    return np.where(np.abs(x) > gamma, np.sign(x) * (np.abs(x) - gamma), 0)
 