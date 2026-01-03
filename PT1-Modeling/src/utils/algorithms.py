import pandas as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

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
    - x_k_1: The final estimated x vector.
    - support: The support of the estimated x vector (indices of non-zero elements).
    - num_iterations: The number of iterations performed
    - estimates_history: A list containing the estimated x vector at each iteration step.
    """
    x_k = x_0
    stop_criteria = 1
    gamma = tau * lam
    num_iterations=0
    estimates_history = []
    
    while (stop_criteria >= (10**(-8))):
        z = x_k + (np.dot(tau, np.dot(C.T, (y - np.dot(C, x_k)))))
        x_k_1 = IST(z, gamma)
        stop_criteria = np.sum(np.linalg.norm(x_k_1 - x_k,2)**2)
        estimates_history.append(np.copy(x_k_1))
        x_k = x_k_1
        num_iterations += 1
    #calculating the support of x
    support = np.where(x_k_1 != 0)[0]
    return x_k_1, support, num_iterations, estimates_history

def localization_plot(true_location, estimated_targets_location, estimated_attacked_sensors, sensor_coords, title=''):
    """
    Visualizes the spatial results of the localization algorithm within a 2D room grid.

    Parameters:
        true_location: Indices of the grid cells corresponding to the ground truth target positions.
        estimated_targets_location: Indices of the grid cells corresponding to the estimated target positions.
        estimated_attacked_sensors: Indices of the sensors identified as attacked.
        sensor_coords: An array containing the (x, y) coordinates of all sensors in the network.
        title: The title of the plot
    """
    H = 10  # Grid's height (# celle)
    L = 10  # Grid's length (# celle)
    W = 100  # Cell's width (cm)
    n = H * L

    room_grid = np.zeros((2, n))

    for i in range(n):
        room_grid[0, i] = W//2 + (i % L) * W
        room_grid[1, i] = W//2 + (i // L) * W

    # --- 2. Plotting ---
    plt.figure(figsize=(7, 7))
    plt.grid(True)
    plt.title(title)
    # True targets location
    plt.plot(room_grid[0, true_location], room_grid[1, true_location], 's', markersize=9, 
            markeredgecolor=np.array([40, 208, 220])/255, 
            markerfacecolor=np.array([40, 208, 220])/255,
            label='True Targets')    
    
    # Estimated targets location
    plt.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 'x', markersize=9, 
            markeredgecolor=np.array([255, 0, 0])/255, 
            markerfacecolor=np.array([255, 255, 255])/255,
            label='Estimated Targets')

    # Sensors
    plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')
    
    # Estimated attacked sensors
    if len(estimated_attacked_sensors) > 0:
        plt.plot(sensor_coords[estimated_attacked_sensors, 0], sensor_coords[estimated_attacked_sensors, 1], 'o', markersize=12, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor='none',
                label='Attacked sensors')

    plt.xticks(np.arange(100, 1001, 100))
    plt.yticks(np.arange(100, 1001, 100))
    plt.xlabel('(cm)')
    plt.ylabel('(cm)')
    plt.axis([0, 1000, 0, 1000])
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')

def tracking_plot(n, true_location, x_hat, a_hat, sensor_coords, title=''):
    H = 10
    L = 10
    W = 100
    room_grid = np.zeros((2, n))
    for i in range(n):
        room_grid[0, i] = W//2 + (i % L) * W
        room_grid[1, i] = W//2 + (i // L) * W
    
    fig, ax = plt.subplots()

    for i in range(50):
        true_location.append([x-1 for x in true_location[i]])

    for x,true_x,a in zip(x_hat,true_location, a_hat):
        estimated_targets_location = np.argsort(x)[-3:]
        estimated_attacked_sensors = np.argsort(np.abs(a))[-2:]
        print("Estimated attacked sensors: ", estimated_attacked_sensors)
        ax.clear()
        # Real targets
        ax.plot(room_grid[0, true_x], room_grid[1, true_x], 's', markersize=9, 
                markeredgecolor=np.array([40, 208, 220])/255, 
                markerfacecolor=np.array([40, 208, 220])/255)
        #  Estimated targets
        ax.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 'x', markersize=9, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor=np.array([255, 255, 255])/255)

        # Plot of sensors
        ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')
        # Plot of estimated sensors under attack
        ax.plot(sensor_coords[estimated_attacked_sensors[0], 0], sensor_coords[estimated_attacked_sensors[0], 1], 'o', markersize=12, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor='none')
        ax.plot(sensor_coords[estimated_attacked_sensors[1], 0], sensor_coords[estimated_attacked_sensors[1], 1], 'o', markersize=12, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor='none')
        ax.grid(True)
        ax.legend(['True Targets', 'Estimated Targets', 'Sensors', 'Attacked sensors'], loc='best')
        ax.set_xticks(np.arange(100, 1001, 100))
        ax.set_yticks(np.arange(100, 1001, 100))
        ax.set_xlabel('(cm)')
        ax.set_ylabel('(cm)')
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_aspect('equal', adjustable='box')
        plt.pause(0.5)