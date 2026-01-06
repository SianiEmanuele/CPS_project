from numpy import linalg

from .utilities import *

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

def sparse_observer(n, q, A, G, tau, lam, y, K):
    # Estimate x_tilda using ISTA
    lam_weights = np.concatenate((np.full(n, 10), np.full(q,20)))
    x_hat = []
    a_hat = []
    z_hat = []

    z_0 = np.zeros(n+q)
    x_hat.append(z_0[:n])
    a_hat.append(z_0[n:])
    z_hat.append(z_0)

    for k in range(K-1):
        z = z_hat[k] + (np.dot(tau, np.dot(G.T, (y[:,k] - np.dot(G, z_hat[k]))))) # Shrinkage and Threshold argument
        gamma = tau * lam * lam_weights
        z_hat_plus = IST(z, gamma)
        x_hat.append(np.dot(A,z_hat_plus[:n]))
        a_hat.append(z_hat_plus[n:])
        z_hat.append(np.hstack((x_hat[k+1], a_hat[k+1])))
    return x_hat, a_hat

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

def DISTA(n, q, D, y, Q, tau, lam_vec, true_location_targets, true_attack_indices, max_iter=1000, tol=1e-8):
    """
    Implements the Distributed ISTA (DISTA) algorithm for target localization and attack detection
    """
    z_nodes = np.zeros((q, n + q))
    x_true = np.zeros(n)
    for i in true_location_targets: x_true[i] = 1  # Creating the target ground truth matrix for targets
    x_accuracy_list_main = []
    # Values to determine if sistem reach consensus and converge and when
    k_x_consensus = -1
    flag_x_cons = False
    k_a_consensus = -1
    flag_a_cons = False
    k_x_conver = -1
    flag_x_conv = False
    k_a_conver = -1
    flag_a_conv = False

    # ====== DISTA algorithm ======
    # Local augmented matrices G_i
    G_list = []
    for i in range(q):
        e_i = np.zeros(q)
        e_i[i] = 1
        G_i = np.hstack((D[i, :], e_i))
        G_list.append(G_i)

    # Main Loop
    for k in range(max_iter):
        z_prev = np.copy(z_nodes)
        z_new = np.zeros_like(z_nodes)
        x_accuracy_list_local = []
        # Consensus Step (Matrix Multiplication for efficiency)
        Qz = np.dot(Q, z_prev)

        # Local Loop (on each sensor)
        for i in range(q):
            G_i = G_list[i]
            y_i = y[i]
            z_i_k = z_prev[i, :]

            gradient_step = tau * G_i * (y_i - np.dot(G_i, z_i_k))
            # Local Soft Thresholding argument
            z = Qz[i, :] + gradient_step
            # Local Soft Thresholding
            z_new[i, :] = IST(z, tau * lam_vec)

            # State accuracy calculation with l2-norm^2
            x_accuracy = np.linalg.norm(z_new[i, :n] - x_true, 2) ** 2
            x_accuracy_list_local.append(x_accuracy)

        x_accuracy_list_main.append(np.mean(x_accuracy_list_local))
        # Stop Criterion calculation
        diff_norm = np.sum([np.linalg.norm(z_new[i] - z_prev[i], 2) ** 2 for i in range(q)])

        # ====== PERFORMANCE METRICS ==========
        if not (flag_x_conv and flag_a_conv):
            x_is_cons, a_is_cons, x_idxs, a_idxs = check_support_consensus(z_new, n,k_elements=2)  # Check if system reached consensus
            # --- State ---
            if x_is_cons:
                if not flag_x_cons:  # consensus
                    k_x_consensus = k
                    flag_x_cons = True
                if not flag_x_conv:  # convergence
                    if np.array_equal(x_idxs, true_location_targets):
                        k_x_conver = k
                        flag_x_conv = True
            # --- Attacks ---
            if a_is_cons:
                if not flag_a_cons:
                    k_a_consensus = k
                    flag_a_cons = True
                if not flag_a_conv:
                    if np.array_equal(a_idxs, true_attack_indices):
                        k_a_conver = k
                        flag_a_conv = True

        if diff_norm < tol:  # Stop criterion reached
            return z_new, k, x_accuracy_list_main, k_x_consensus, k_a_consensus, k_x_conver, k_a_conver  # Return values if converge

        z_nodes = z_new

        if k > 0 and k % 5000 == 0:
            print(f"      Iter {k}: Diff Norm {diff_norm:.2e}")

    return z_nodes, max_iter, x_accuracy_list_main, k_x_consensus, k_a_consensus, k_x_conver, k_a_conver  # Return values if does not converge
