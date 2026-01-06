import os as os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from scipy import stats

# List of sensors positions for localization plot
sensor_coords = np.array([
    [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
    [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
    [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
])

# =================== BASE ALGOTITHMS ==================================

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

# ==================================================================================================================

def ISTA_runs(runs, p, q, C, tau, lam, x_sparsity):
    # parameters definition with suggested settings
    q = q
    p = p

    x_tilda_supports = []
    x_estimated_supports = []
    num_iterations = []

    # running the simulation 20 times
    for _ in range(runs):
        # generating x_tilda with n-sparsity
        x_tilda = np.zeros(p)
        x_tilda[np.random.choice(p, x_sparsity, replace=False)] = np.random.choice([-1, 1], 2) * np.random.uniform(1, 2)
        x_tilda_supports.append(np.where(x_tilda != 0)[0])  # real x_tilda support

        eta = 10 ** (-2) * np.random.randn(q)

        y = np.dot(C, x_tilda) + eta
        x_estimated, x_estimated_supp, iterations = ISTA(np.zeros(p), C, tau, lam, y)
        x_estimated_supports.append(x_estimated_supp)
        num_iterations.append(iterations)

    correct_estimations = 0
    for j in range(runs):
        if np.array_equal(x_tilda_supports[j], x_estimated_supports[j]):
            correct_estimations += 1
    return correct_estimations, num_iterations, x_estimated

def ISTA_runs_with_attacks(runs, n, q, C, tau, lam, x_sparsity, a_sparsity, attack_type, noisy):
    correct_estimations = 0
    num_iterations = []
    estimation_accuracy = []

    for _ in range(20, runs):
        # Generate x_tilda with standard uniform distribution
        x_tilda = np.random.randn(n)

        # Generate the sparse attack vector a
        a = np.zeros(q)
        attack_indices = np.random.choice(q, a_sparsity, replace=False)
        a[attack_indices] = np.random.choice([-2, -1, 1, 2], a_sparsity)

        if (noisy):
            eta = 10 ** (-2) * np.random.randn(q)
        else:
            eta = np.zeros(q)

        # Generate the measurements y
        if attack_type == "UNAWARE":
            y = np.dot(C, x_tilda) + eta + a
        elif attack_type == "AWARE":
            y = np.dot(C, x_tilda) + eta
            y[attack_indices] += 0.5 * y[attack_indices]

        # Estimate xe using the weighted ISTA_runs
        lam_weights = np.concatenate((np.zeros(n), np.ones(q)))
        # print("lam :", lam_weights * lam)
        G = np.hstack((C, np.eye(q)))
        w = np.zeros(n + q)
        w_estimated, w_estimated_supp, iterations = ISTA(w, G, tau, lam * lam_weights, y)

        # Extract the estimated x
        x_estimated = w_estimated[:n]

        # Retrieve the estimated attack vector
        a_estimated = w_estimated[n:]

        # Calculate the estimation accuracy
        estimation_accuracy.append(np.linalg.norm(x_tilda - x_estimated) ** 2)

        # Check if the attack was correctly detected
        if np.array_equal(np.where(a != 0)[0], np.where(a_estimated != 0)[0]):
            correct_estimations += 1

        num_iterations.append(iterations)

    attack_detection_rate = correct_estimations / (runs - 20)

    return attack_detection_rate, num_iterations, estimation_accuracy

def Localization_with_attacks(n, q, G, tau, lam, y):
    # Estimate x_tilda using ISTA
    lam_weights = np.concatenate((np.full(n, 10), np.full(q, 20)))
    w = np.zeros(n + q)
    w_estimated, w_estimated_supp, iterations = ISTA(w, G, tau, lam * lam_weights, y)

    return w_estimated, w_estimated_supp, iterations

def check_support_consensus(z_nodes, n_state, k_elements=2):
    """
    Checks if all nodes in the network agree on the support (indices of the largest elements)
    for both the state vector (x) and the attack vector (a).
    """
    x_estimates = z_nodes[:, :n_state]
    a_estimates = z_nodes[:, n_state:]
    # Find indices of the k largest values (magnitude)
    x_est_idx = np.argsort(np.abs(x_estimates), axis=1)[:, -k_elements:]
    a_est_idx = np.argsort(np.abs(a_estimates), axis=1)[:, -k_elements:]

    x_est_idx = np.sort(x_est_idx, axis=1)
    a_est_idx = np.sort(a_est_idx, axis=1)

    x_first = x_est_idx[0]
    a_first = a_est_idx[0]
    x_cons = np.all(x_est_idx == x_first)
    a_cons = np.all(a_est_idx == a_first)

    return x_cons, a_cons, x_first, a_first

def check_convergence(x_estimates, a_estimates, true_targets_locations, true_attacked_sensors, n_targets=2, n_attacks=2, n_change_attacks=0):
    """
    Evaluates the convergence and stability of state (x) and attack (a) estimates. for task 4 optional
    """
    i = 0
    x_converged_status = False
    a_converged_status = False
    attacks_convergence_iteration = []
    state_convergence_iteration = 50
    prev_a_est_idx = []
    total_iterations = len(x_estimates)
    interval_step = total_iterations // (n_change_attacks) if (n_change_attacks) > 0 else total_iterations
    prev_config = 0

    for x_est, a_est, true_x in zip(x_estimates, a_estimates, true_targets_locations):
        # Calculating indexes
        x_est_idx = np.argsort(np.abs(x_est))[-n_targets:]
        a_est_idx = np.argsort(np.abs(a_est))[-n_attacks:]
        x_est_idx = np.sort(x_est_idx)
        a_est_idx = np.sort(a_est_idx)
        true_targets = np.sort(np.argsort(true_x)[-n_targets:])

        # If sensors under attacks change during execution (part 2)
        if n_change_attacks > 0:
            current_config_idx = (i) // interval_step
            current_config_idx = min(current_config_idx, len(true_attacked_sensors) - 1)
            current_true_a = true_attacked_sensors[current_config_idx]
            attacked_sensors = np.sort(current_true_a).tolist()
        else:
            current_config_idx = 0
            if len(true_attacked_sensors) > 0 and isinstance(true_attacked_sensors[0], int): # Checking if the sensors under attack list is a list of int (part 3)
                attacked_sensors = np.sort(true_attacked_sensors).tolist()
            elif len(true_attacked_sensors) > 0 and isinstance(true_attacked_sensors[0], tuple): # Checking if the sensors under attack list is a list of tuple
                attacked_sensors = np.sort(true_attacked_sensors[0]).tolist()
            else:
                attacked_sensors = []

        expected_x = (prev_x_est_idx - 1) % 100 if i != 0 else x_est_idx # Calculating current x estimation index from previous
        prev_x_est_idx = x_est_idx
        # --- State convergence check ---
        is_x_correct = np.array_equal(x_est_idx, true_targets)
        is_x_maintenance = np.array_equal(x_est_idx, np.sort(expected_x))
        if not x_converged_status and is_x_correct: # First time estimation is correct
            x_converged_status = True
            state_convergence_iteration = i
        elif x_converged_status and (not is_x_maintenance or not is_x_correct): # If the estimation is uncorrect
            x_converged_status = False
            state_convergence_iteration = -1

        # --- Attacks convergence check ---
        is_a_correct = np.array_equal(a_est_idx, attacked_sensors)
        is_a_maintenance = np.array_equal(a_est_idx, prev_a_est_idx)
        if n_change_attacks > 0 and current_config_idx != prev_config: # Changed sensors under attack
            a_converged_status = False
            if len(attacks_convergence_iteration) != current_config_idx: # Changed sensors under attacks without attacks estimation convergence
                attacks_convergence_iteration.append('-')
        if not a_converged_status and is_a_correct:
            a_converged_status = True
            attacks_convergence_iteration.append(i)
        elif a_converged_status and (not is_a_maintenance or (n_change_attacks > 0 and current_config_idx == prev_config and not is_a_correct)):
            a_converged_status = False
            attacks_convergence_iteration.pop()

        prev_a_est_idx = a_est_idx
        prev_config = current_config_idx if n_change_attacks != 0 else 0 # Calculating current x estimation index from previous
        i = i+1

    print(f'x_conv: {x_converged_status}, state_convergence_iteration: {state_convergence_iteration}')
    print(f'a_conv: {a_converged_status}, attacks_convergence_iteration: {attacks_convergence_iteration}')
    print('-----------------------------------------------\n')
    return x_converged_status, a_converged_status, state_convergence_iteration, attacks_convergence_iteration

def Localization_with_attacks_task_5(n, q, G, tau, lam, y, true_location_targets, true_attack_indices):
    lam_weights = np.concatenate((np.full(n, 10), np.full(q, 0.1)))
    final_lam = lam * lam_weights
    x_true = np.zeros(n)
    for i in true_location_targets: x_true[i] = 1

    a_true = np.zeros(q)
    for i in true_attack_indices: a_true[i] = 1

    w = np.zeros(n + q)

    w_estimated, w_estimated_supp, iterations, history = ISTA_task_5(w, G, tau, final_lam, y)

    x_acc_hist = []

    for w_step in history:
        x_est = w_step[:n]
        # Calculating accuracy
        x_acc_hist.append(np.linalg.norm(x_est - x_true, 2)**2)

    return w_estimated, w_estimated_supp, iterations, x_acc_hist

def distributed_localization():
    """
    Distributed target localization under sparse sensor attacks using DISTA.
    """
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    mat = sio.loadmat(cwd + r'/utils/distributed_localization_data.mat')
    y = np.squeeze(mat['y'])
    D = mat['D']
    Q12 = mat['Q_12']
    Q18 = mat['Q_18']
    Q4 = mat['Q_4']
    Q8 = mat['Q_8']
    matrices_list = [Q4, Q8, Q12, Q18]
    topologies_names = ["TOPOLOGY 1 (Q4)", "TOPOLOGY 2 (Q8)", "TOPOLOGY 3 (Q12)", "TOPOLOGY 4 (Q18)"]

    n = D.shape[1]
    q = D.shape[0]
    true_location = [13, 24]
    true_attack_indices = [7, 22]

    # Parameters
    tau = 4e-7
    lam_vec = np.concatenate((np.full(n, 10), np.full(q, 0.1)))
    attack_threshold = 0.002

    # List to store accuracy curves for final comparison
    x_all_topologies_accuracy = []

    # --- LOOP OVER ALL TOPOLOGIES ---
    for i, Q_curr in enumerate(matrices_list):
        print(f"--- {topologies_names[i]} ---")

        # Eigenvalue analysis
        evals = np.abs(np.linalg.eigvals(Q_curr))
        lambda_2 = np.sort(evals)[::-1][1]
        print(f"   |lambda_2|: {lambda_2:.5f}")
        iterations = 15000

        # Run DISTA
        z_nodes, stop_criteria_iter, x_accuracy, k_x_cons, k_a_cons, k_x_conv, k_a_conv = DISTA(n, q, D, y, Q_curr, tau,
                                                                                                lam_vec, true_location,
                                                                                                true_attack_indices,
                                                                                                max_iter=iterations)

        print("\n--- Performance Metrics ---")
        print(f"   X Consensus (k_x_cons)   : {k_x_cons if k_x_cons != -1 else 'Not Reached'}")
        print(f"   A Consensus (k_a_cons)   : {k_a_cons if k_a_cons != -1 else 'Not Reached'}")
        print(f"   X Converged (k_x_conv)   : {k_x_conv if k_x_conv != -1 else 'Not Reached'}")
        print(f"   A Converged (k_a_conv)   : {k_a_conv if k_a_conv != -1 else 'Not Reached'}")
        # Check if the consensus algorithm reached stop condition
        if stop_criteria_iter < iterations:
            print(f"   Reached stop criteria at iteration: {stop_criteria_iter}")
        else:
            print(f"   Reached MAX ITERATIONS ({stop_criteria_iter}) without reach stop criteria")

        z_final = np.mean(z_nodes, axis=0)
        x_est = z_final[:n]
        a_est = z_final[n:]

        # Refinement of a values
        a_est_refined = np.copy(a_est)
        a_est_refined[np.abs(a_est_refined) < attack_threshold] = 0

        # Extract Indices
        estimated_targets_location = np.argsort(x_est)[-2:]
        estimated_attacked_sensors = np.where(a_est_refined != 0)[0]
        est_attack_values = a_est_refined[estimated_attacked_sensors]

        print(f"   Estimated Targets: {estimated_targets_location} (True: {true_location})")
        print(f"   Estimated Attacks: {estimated_attacked_sensors} (True: {true_attack_indices})")
        if len(estimated_attacked_sensors) > 0:
            print("   Estimated Attack Values:")
            for idx, val in zip(estimated_attacked_sensors, est_attack_values):
                print(f"      -> Sensor {idx}: {val:.4f}")
        else:
            print("      -> No attacks detected.")

        # Process Accuracy for the state global plot
        x_acc_array = np.array(x_accuracy)
        # Calculate MEAN error across all nodes for each iteration
        x_all_topologies_accuracy.append(x_acc_array)

        localization_plot(true_location, true_attack_indices, estimated_targets_location, estimated_attacked_sensors,
                          title=f"{topologies_names[i]}\nStop criteria reached at iter: {stop_criteria_iter}")
        print('\n --------------------------------------------------- \n')

    # STATE ACCURACY PLOT
    colors = ['b', 'g', 'r', 'm']
    # Determine the maximum number of iterations any topology ran for
    max_len_x = max(len(curve) for curve in x_all_topologies_accuracy)

    plt.figure(figsize=(10, 6))

    for i, acc_curve in enumerate(x_all_topologies_accuracy):
        current_len = len(acc_curve)
        plt.plot(acc_curve, label=topologies_names[i], color=colors[i % len(colors)], linewidth=0.5)
        plt.plot(current_len - 1, acc_curve[-1], 'o', color=colors[i % len(colors)])

    plt.title('State Accuracy (Distributed)')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Error (L2 Norm)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return x_all_topologies_accuracy, topologies_names

def centralized_localization():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()

    mat = sio.loadmat(cwd + r'/utils/distributed_localization_data.mat')
    y = np.squeeze(mat['y'])
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]

    G = np.hstack((D, np.eye(q)))

    true_location = [13, 24]
    true_attack_indices = [7, 22]

    # Parameters
    attack_threshold = 0.0015
    tau = 1 / (np.linalg.norm(G, ord=2) ** 2) - 10 ** (-8)
    lam_scalar = 1

    z_est, support, stop_iter, x_acc_hist = Localization_with_attacks_task_5(n, q, G, tau, lam_scalar, y, true_location,
                                                                             true_attack_indices)

    print(f"Centralized localization converged at iteration: {stop_iter}")

    x_est = z_est[:n]
    a_est = z_est[n:]
    attacks = np.sort(a_est)[-2:]
    a_est_refined = np.copy(a_est)
    a_est_refined[np.abs(a_est_refined) < attack_threshold] = 0

    estimated_targets_location = np.argsort(x_est)[-2:]
    estimated_attacked_sensors = np.where(a_est_refined != 0)[0]

    print(f"   Estimated Targets: {estimated_targets_location} (True: {true_location})")
    print(f"   Estimated Attacks: {estimated_attacked_sensors} (True: {true_attack_indices})")

    print('\n --------------------------------------------------- \n')
    print("Generating Plots...")

    # --- PLOTTING ---
    localization_plot(true_location, true_attack_indices, estimated_targets_location, estimated_attacked_sensors)

    # State Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_acc_hist, label='Centralized ISTA', color='b', linewidth=0.5)
    plt.plot(len(x_acc_hist) - 1, x_acc_hist[-1], 'o', color='b')
    plt.title('State Accuracy (Centralized)')
    plt.xlabel('Iterations')
    plt.ylabel('Error (L2 Norm)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return x_acc_hist

# ==================================================================================================================

# =================== PLOTS ==================================

def localization_plot(true_location, true_attacked_sensors, estimated_targets_location, estimated_attacked_sensors,
                        title=''):
    """
    Visualizes the spatial results of the localization algorithm within a 2D room grid.

    Parameters:
        true_location: Indices of the grid cells corresponding to the ground truth target positions.
        estimated_targets_location: Indices of the grid cells corresponding to the estimated target positions.
        estimated_attacked_sensors: Indices of the sensors identified as attacked.
        true_attacked_sensors: Indices of attacked sensors.
        sensor_coords: An array containing the (x, y) coordinates of all sensors in the network.
        title: The title of the plot
    """
    H, L, W = 10, 10, 100
    n = H * L
    room_grid = np.zeros((2, n))
    for i in range(n):
        room_grid[0, i] = W // 2 + (i % L) * W
        room_grid[1, i] = W // 2 + (i // L) * W

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.grid(True)
    ax.set_title(title)

    # Colore azzurro target
    target_color = np.array([40, 208, 220]) / 255

    # True Targets
    ax.scatter(room_grid[0, true_location].flatten(),
               room_grid[1, true_location].flatten(),
               marker='s', s=100, c=[target_color], edgecolors=target_color,
               label='True Targets', zorder=3)

    # Estimated Targets
    ax.scatter(room_grid[0, estimated_targets_location].flatten(),
               room_grid[1, estimated_targets_location].flatten(),
               marker='x', s=100, c='red',
               label='Estimated Targets', zorder=4)

    # All Sensors
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
               s=60, c='pink', alpha=0.4, label='Sensors', zorder=1)

    # Estimated Attacked Sensors
    if len(estimated_attacked_sensors) > 0:
        # Quelli che l'algoritmo PENSA siano attaccati (Cerchio Rosso)
        ax.scatter(sensor_coords[estimated_attacked_sensors, 0],
                   sensor_coords[estimated_attacked_sensors, 1],
                   marker='o', s=200, facecolors='none', edgecolors='red',
                   linewidths=1.5, label='Estimated Attacked Sensors', zorder=2)

    # True attacked sensors
    if len(true_attacked_sensors) > 0:
        ax.scatter(sensor_coords[true_attacked_sensors, 0],
                   sensor_coords[true_attacked_sensors, 1],
                   marker='*', s=60, c=[target_color],
                   label='True Attacked Sensors', zorder=5)

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

    # Formattazione assi
    ax.set_xticks(np.arange(0, 1001, 100))
    ax.set_yticks(np.arange(0, 1001, 100))
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_aspect('equal')
    plt.show()


def tracking_plot(n, true_location, x_hat, a_hat, true_attacked_sensors, n_attacks=2, title=''):
    L = 10
    W = 100
    k = 0
    room_grid = np.zeros((2, n))
    for i in range(n):
        room_grid[0, i] = W // 2 + (i % L) * W
        room_grid[1, i] = W // 2 + (i // L) * W

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(50):
        true_location.append([x - 1 for x in true_location[i]])

    for x, true_x, a in zip(x_hat, true_location, a_hat):

        estimated_targets_location = np.argsort(x)[-3:]
        estimated_attacked_sensors = np.argsort(np.abs(a))[-n_attacks:]
        ax.clear()
        # Real targets
        ax.plot(room_grid[0, true_x], room_grid[1, true_x], 's', markersize=9,
                markeredgecolor=np.array([40, 208, 220]) / 255,
                markerfacecolor=np.array([40, 208, 220]) / 255)
        #  Estimated targets
        ax.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 'x', markersize=9,
                markeredgecolor=np.array([255, 0, 0]) / 255,
                markerfacecolor=np.array([255, 255, 255]) / 255)
        ax.set_title(f'Iteration: {k}')

        # Plot of sensors
        ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')

        # Plot of estimated sensors under attack
        for attack_number in range(n_attacks):
            ax.plot(sensor_coords[estimated_attacked_sensors[attack_number], 0],
                    sensor_coords[estimated_attacked_sensors[attack_number], 1], 'o', markersize=12,
                    markeredgecolor=np.array([255, 0, 0]) / 255,
                    markerfacecolor='none')
            ax.plot(sensor_coords[true_attacked_sensors[k][attack_number], 0],
                    sensor_coords[true_attacked_sensors[k][attack_number], 1], '*', markersize=5,
                    markeredgecolor=np.array([40, 208, 220]) / 255,
                    markerfacecolor=np.array([40, 208, 220]) / 255)
        ax.grid(True)
        ax.legend(['True Targets', 'Estimated Targets', 'Sensors', 'Estimated attacked sensors', 'Attacked sensors'],
                  loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        ax.set_xticks(np.arange(100, 1001, 100))
        ax.set_yticks(np.arange(100, 1001, 100))
        ax.set_xlabel('(cm)')
        ax.set_ylabel('(cm)')
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_aspect('equal', adjustable='box')
        plt.pause(0.5)
        k += 1


def plot_convergence_iterations(attacked_sensors, x_results, a_results, attack_type):
    x = range(1,len(attacked_sensors))

    plt.figure()
    plt.plot(x, x_results, marker='o', label='x convergence iterations')
    plt.plot(x, a_results, marker='s', label='a convergence iterations')

    plt.xlabel('Number of Attacked Sensors')
    plt.ylabel('Convergence Iterations')
    plt.title(f'{attack_type} | Convergence vs Number of Attacked Sensors')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==================================================================================================================
