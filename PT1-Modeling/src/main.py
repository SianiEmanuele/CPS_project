import os as os
from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
from kornia.augmentation.auto.autoaugment.ops import color

from utils import ISTA, IST, ISTA_task_5, localization_plot, tracking_plot
import scipy.io as sio
from scipy import stats
import networkx as nx

# List of sensors positions for localization plot
sensor_coords = np.array([
    [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
    [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
    [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
])
        
def ISTA_runs( runs, p, q, C, tau, lam, x_sparsity):
    #parameters definition with suggested settings
    q=q
    p=p

    x_tilda_supports = []
    x_estimated_supports = []
    num_iterations = []

    #running the simulation 20 times
    for _ in range(runs):        
        # generating x_tilda with n-sparsity
        x_tilda = np.zeros(p)
        x_tilda[np.random.choice(p, x_sparsity, replace=False)] = np.random.choice([-1, 1], 2) * np.random.uniform(1, 2)
        x_tilda_supports.append (np.where(x_tilda != 0)[0]) #real x_tilda support
        
        eta = 10**(-2) * np.random.randn(q)

        y = np.dot(C, x_tilda) + eta
        x_estimated, x_estimated_supp, iterations = ISTA(np.zeros(p),C,tau,lam,y)
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

    for _ in range(20,runs):
        # Generate x_tilda with standard uniform distribution        
        x_tilda = np.random.randn(n)

        # Generate the sparse attack vector a
        a = np.zeros(q)
        attack_indices = np.random.choice(q, a_sparsity, replace=False)
        a[attack_indices] = np.random.choice([-2, -1, 1, 2], a_sparsity)

        if(noisy):
            eta = 10**(-2) * np.random.randn(q)
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
        #print("lam :", lam_weights * lam)
        G = np.hstack((C, np.eye(q)))
        w = np.zeros(n+q)
        w_estimated, w_estimated_supp, iterations = ISTA(w, G, tau, lam * lam_weights, y)

        # Extract the estimated x
        x_estimated = w_estimated[:n]

        # Retrieve the estimated attack vector
        a_estimated = w_estimated[n:]

        # Calculate the estimation accuracy
        estimation_accuracy.append(np.linalg.norm(x_tilda - x_estimated)**2)

        # Check if the attack was correctly detected
        if np.array_equal(np.where(a != 0)[0], np.where(a_estimated != 0)[0]):
            correct_estimations += 1

        num_iterations.append(iterations)

    attack_detection_rate = correct_estimations / (runs-20)

    return attack_detection_rate, num_iterations, estimation_accuracy

def Localization_with_attacks(n, q, G, tau, lam, y):
    # Estimate x_tilda using ISTA
    lam_weights = np.concatenate((np.full(n, 10), np.full(q,20)))
    w = np.zeros(n+q)
    w_estimated, w_estimated_supp, iterations = ISTA(w, G, tau, lam * lam_weights, y)

    return w_estimated, w_estimated_supp, iterations

def observer(n, q, A, G, tau, lam, y, K):
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

def DISTA(n, q, D, y, Q, tau, lam_vec, true_location_targets, true_attack_indices, max_iter=1000, tol=1e-8):
    """
    Implements the Distributed ISTA (DISTA) algorithm for target localization and attack detection
    """
    z_nodes = np.zeros((q, n + q)) 
    x_true = np.zeros(n)
    for i in true_location_targets: x_true[i]=1 # Creating the target ground truth matrix for targets
    x_accuracy_list_main = []
    # Values to determine if sistem reach consensus and converge and when
    k_x_consensus = -1; flag_x_cons = False
    k_a_consensus = -1; flag_a_cons = False
    k_x_conver = -1; flag_x_conv = False
    k_a_conver = -1; flag_a_conv = False
    
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
            x_accuracy = np.linalg.norm(z_new[i, :n] - x_true, 2)**2
            x_accuracy_list_local.append(x_accuracy)

        x_accuracy_list_main.append(np.mean(x_accuracy_list_local))
        # Stop Criterion calculation
        diff_norm = np.sum([np.linalg.norm(z_new[i] - z_prev[i],2)**2 for i in range(q)])

        # ====== PERFORMANCE METRICS ==========
        if not (flag_x_conv and flag_a_conv):
            x_is_cons, a_is_cons, x_idxs, a_idxs = check_support_consensus(z_new, n, k_elements=2) # Chec if system reacked consensus
            # --- State ---
            if x_is_cons:
                if not flag_x_cons: # consensus
                    k_x_consensus = k
                    flag_x_cons = True
                if not flag_x_conv: # convergence
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
        
        if diff_norm < tol: # Staop criterion reached
            return z_new, k, x_accuracy_list_main, k_x_consensus, k_a_consensus, k_x_conver, k_a_conver # Return values if converge
        
        z_nodes = z_new
        
        if k > 0 and k % 5000 == 0:
            print(f"      Iter {k}: Diff Norm {diff_norm:.2e}")
            
    return z_nodes, max_iter, x_accuracy_list_main, k_x_consensus, k_a_consensus, k_x_conver, k_a_conver # Return values if does not converge

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

def Localization_with_attacks_task_5(n, q, G, tau, lam, y, true_location_targets, true_attack_indices):
    lam_weights = np.concatenate((np.full(n, 10), np.full(q, 0.1)))
    final_lam = lam * lam_weights

    x_true = np.zeros(n)
    for i in true_location_targets: x_true[i]=1

    a_true = np.zeros(q)
    for i in true_attack_indices: a_true[i]=1
    
    # Inizializzazione
    w = np.zeros(n + q)
    
    # Chiamata a ISTA (Firma originale rispettata)
    w_estimated, w_estimated_supp, iterations, history = ISTA_task_5(w, G, tau, final_lam, y)
    
    x_acc_hist = []
    
    for w_step in history:
        x_est = w_step[:n]
        # Calcola errore norma L2
        x_acc_hist.append(np.linalg.norm(x_est - x_true, 2))

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
        z_nodes, stop_criteria_iter, x_accuracy, k_x_cons, k_a_cons, k_x_conv, k_a_conv = DISTA(n, q, D, y, Q_curr, tau, lam_vec, true_location, true_attack_indices, max_iter=iterations)
        
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

        localization_plot(true_location, estimated_targets_location, estimated_attacked_sensors, sensor_coords, title=f"{topologies_names[i]}\nStop criteria reached at iter: {stop_criteria_iter}")

        print('\n --------------------------------------------------- \n')

    # STATE ACCURACY PLOT
    colors = ['b', 'g', 'r', 'm'] 
    # Determine the maximum number of iterations any topology ran for
    max_len_x = max(len(curve) for curve in x_all_topologies_accuracy)
    
    plt.figure(figsize=(10, 6))
    
    for i, acc_curve in enumerate(x_all_topologies_accuracy):
        current_len = len(acc_curve)
        pad_width = max_len_x - current_len
        # Add padding to the curves to make them equal length
        if pad_width > 0:
            padded_acc = np.pad(acc_curve, (0, pad_width), mode='edge')
        else:
            padded_acc = acc_curve
            
        plt.plot(padded_acc, label=topologies_names[i], color=colors[i % len(colors)], linewidth=0.5)
        plt.plot(current_len-1, acc_curve[-1], 'o', color=colors[i % len(colors)])

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
    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam_scalar = 1

    z_est, support, stop_iter, x_acc_hist = Localization_with_attacks_task_5(n, q, G, tau, lam_scalar, y, true_location, true_attack_indices)

    print(f"Centralized localization converged at iteration: {stop_iter}")

    x_est = z_est[:n]
    a_est = z_est[n:]
    attacks = np.sort(a_est)[-2:]
    print('ATTACKS: ', attacks)
    a_est_refined = np.copy(a_est)
    a_est_refined[np.abs(a_est_refined) < attack_threshold] = 0
    
    estimated_targets_location = np.argsort(x_est)[-2:]
    estimated_attacked_sensors = np.where(a_est_refined != 0)[0]
    
    print(f"   Estimated Targets: {estimated_targets_location} (True: {true_location})")
    print(f"   Estimated Attacks: {estimated_attacked_sensors} (True: {true_attack_indices})")
    
    print('\n --------------------------------------------------- \n')
    print("Generating Plots...")

    # --- PLOTTING ---
    localization_plot(true_location, estimated_targets_location, estimated_attacked_sensors, sensor_coords)

    # State Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_acc_hist, label='Centralized ISTA', color='b', linewidth=0.5)
    plt.plot(len(x_acc_hist)-1, x_acc_hist[-1], 'o', color='b')
    plt.title('State Accuracy (Centralized)')
    plt.xlabel('Iterations')
    plt.ylabel('Error (L2 Norm)') 
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return x_acc_hist

############################### TASK 1 ##################################################
def task_1():

    q=10
    p=20
    C = np.random.randn(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)
    sparsity = 2

    runs = 1000

    print("\nFIRST EXERCISE WITH SUGGESTED PARAMETERS (q=10, p=20)\n")

    print("             QUESTION 1\n- Support recovery rate: how many times the support of x_tilda is correctly estimated?")
    correct_estimations, num_iterations, _ = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
    print("The support of x_tilda is correctly estimated in ", correct_estimations, " out of", runs ," runs" ,
            "\nMin iterations = ", min(num_iterations), " || Max iterations = ", max(num_iterations), " || Mean Iterations = ", np.mean(num_iterations), "\n")
    ##################################### QUESTION 2 ##############################################################
    q_list = range(10, 51)
    correct_estimations_percentage = []
    max_iterations = []
    min_iterations = []
    mean_iterations = []
    print("             QUESTION 2\n- Can we obtain 100% of success in the support recovery by increasing q?\n")
    # Simulation
    for q in q_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        tau = 1 / (C_l_2_norm**2) - 10**(-8)
        lam = 1 / (100*tau)
        correct_estimations, num_iterations, _ = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage.append(correct_estimations*100/runs)
        max_iterations.append(np.max(num_iterations))
        min_iterations.append(np.min(num_iterations))
        mean_iterations.append(np.mean(num_iterations))
    #plotting correct estimations percentage
    plt.plot(q_list, correct_estimations_percentage)
    plt.xlabel("q")
    plt.ylabel("Correct estimations percentage")
    plt.title("Percentage of correct estimations in function of q")
    plt.grid()
    plt.show()
    #plotting min, max and mean iterations in 3 different lines
    fig, axs = plt.subplots(3)
    fig.suptitle('Iterations in function of q')
    axs[0].plot(q_list, min_iterations)
    axs[0].set_title('Min iterations')
    axs[1].plot(q_list, max_iterations)
    axs[1].set_title('Max iterations')
    axs[2].plot(q_list, mean_iterations)
    axs[2].set_title('Mean iterations')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    plt.show()

    ##################################### QUESTION 4 ##############################################################

    # In order to highlight the fact that the value of q is the determining factor in the correct estimation of the support, we will conduct two analysis.
    #     1. Varying tau keeping q equal to 10
    #     2. Varying tau keeping q equal to 20 (order of x)
    # For the comparison to be fair we used the same tau with both q==10 and q==20

    print("             QUESTION 4\n q = 10 | Try different values for τ , by keeping τλ constant\n")
    
    ##### q = 10 ######
    q = 10
    tau_list = []
    C = np.random.randn(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    for i in range (0, 10):
        tau_list.append(1 / (C_l_2_norm**2) - 10**(-8) - i * 10**(-3))
    correct_estimations_percentage = []
    max_iterations = []
    min_iterations = []
    mean_iterations = []

    for tau in tau_list:
        lam = 1 / (100 * tau)
        correct_estimations, num_iterations, _ = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage.append(correct_estimations*100/runs)
        max_iterations.append(np.max(num_iterations))
        min_iterations.append(np.min(num_iterations))
        mean_iterations.append(np.mean(num_iterations))
        print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")
    
    #plotting correct estimations percentage
    plt.figure()
    plt.ylim(0, 100)
    plt.plot(tau_list, correct_estimations_percentage)
    plt.gca().invert_xaxis()
    plt.xlabel("tau")
    plt.ylabel("Correct estimations percentage")
    plt.title("q = 10 | Percentage of correct estimations in function of tau")
    plt.grid()
    plt.show()

    plt.figure()
    plt.yscale('log')
    plt.plot(tau_list, min_iterations, color='b', label="Min")
    plt.plot(tau_list, max_iterations, color='g', label="Max")
    plt.plot(tau_list, mean_iterations, color='m', label="Mean")
    plt.gca().invert_xaxis()
    plt.xlabel("tau")
    plt.ylabel("Number of iterations")
    plt.title("q = 10 | Convergence time in function of tau")
    plt.legend()
    plt.grid()
    plt.show()

    ##################################### QUESTION 5 ##############################################################
    # In order to highlight the fact that the value of q is the determining factor in the correct estimation of the support, we will conduct two analysis.
    # 1. Varying lambda keeping q equal to 10 and tau constant
    # 2. Varying lambda keeping q equal to 24 and tau constant
    # For the comparison to be fair we used the same lambda with both q==10 and q==24
    print("             QUESTION 5\nq=10 | Try different values for λ , by keeping τ constant\n")

    q = 10

    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam_list = []
    for i in range(0, 10):
        lam_list.append(1 / (100*tau) - i * 10**(-2))

    correct_estimations_percentage = []
    max_iterations = []
    min_iterations = []
    mean_iterations = []

    for lam in lam_list:
        correct_estimations, num_iterations, _ = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage.append(correct_estimations*100/runs)
        max_iterations.append(np.max(num_iterations))
        min_iterations.append(np.min(num_iterations))
        mean_iterations.append(np.mean(num_iterations))
        print("q=10 | lam = ", lam, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")


    #plotting correct estimations percentage
    plt.figure()
    plt.ylim(0, 100)
    plt.plot(lam_list, correct_estimations_percentage, color='r')
    plt.gca().invert_xaxis()
    plt.xlabel("lambda")
    plt.ylabel("Correct estimations percentage")
    plt.title("Percentage of correct estimations in function of lambda")
    plt.grid()
    plt.show()

    plt.figure()
    plt.yscale('log')
    plt.plot(lam_list, min_iterations, color='b', label="Min")
    plt.plot(lam_list, max_iterations, color='g', label="Max")
    plt.plot(lam_list, mean_iterations, color='m', label="Mean")
    plt.gca().invert_xaxis()
    plt.xlabel("lam")
    plt.ylabel("Number of iterations")
    plt.title("q = 10 | Convergence time in function of lambda")
    plt.legend()
    plt.grid()
    plt.show()

############################### TASK 2 ##################################################
def task_2():
    runs=1000
    q=20
    p=10
    x_sparsity = p
    a_sparsity = 2
    C = np.random.randn(q, p)
    #calculate tau as a vector with q zeroes and p ones
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 2 * 10**(-3) / tau

    # Case with unaware attacks and without noise
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=False)
    
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("UNAWARE - CLEAN | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    # Case with unaware attacks and with noise
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=True)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("UNAWARE - NOISY | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    # Case with aware attacks and without noise
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=False)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("AWARE - CLEAN | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    # Case with aware attacks and with noise
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=True)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("AWARE - NOISY | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

############################### TASK 3 ##################################################
def task_3():
    true_location = []
    true_location.append([22,35,86])
    cwd = os.getcwd()
    # original matrices
    mat = sio.loadmat(cwd + r'/utils/localization.mat')

    A = mat['A']
    y = np.squeeze(mat['y'])
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]

    G = np.hstack((D, np.eye(q)))
    # normalize G
    G = stats.zscore(G, axis=0)

    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam = 1
    
    w_estimated, w_estimated_supp, iterations = Localization_with_attacks(n, q, G, tau, lam, y)

    # Extract the estimated targets' location by taking the 3 greatest values of the first n elements of w_estimated
    estimated_targets_location = np.argsort(w_estimated[:n])[-3:]
    x_estimated = w_estimated[:n]
    print('x_estimated: ', x_estimated)

    x_true = np.zeros(n)
    for i in true_location: x_true[i]=1
    accuracy = np.linalg.norm(x_true - x_estimated)**2
    print('accuracy: ', accuracy)


    # Extract the estimated attacked vectors from the support of the last q eleemnts of w_estimated
    estimated_attacked_sensors = np.where(w_estimated[n:] != 0)[0]
    
    print("Estimated targets location: ", estimated_targets_location)
    print("Estimated attacked sensors: ", estimated_attacked_sensors)

    localization_plot(true_location, estimated_targets_location, estimated_attacked_sensors, sensor_coords)
    plt.show()

############################### TASK 4 ##################################################
def task_4():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    # mat = sio.loadmat(cwd + r'/CPS_project/PT1-Modeling/src/utils/tracking_moving_targets.mat')
    mat = sio.loadmat(cwd + r'/utils/tracking_moving_targets.mat')

    A = mat['A']
    y = mat['Y']
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]
    K = y.shape[1]

    G = np.hstack((D, np.eye(q)))
    G = stats.zscore(G, axis=0)

    true_location = []
    true_location.append([22,35,86])

    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam = 1
    x_hat, a_hat = observer(n, q, A, G, tau, lam, y, K)
    # Create the graph with moving targets
    tracking_plot(n, true_location, x_hat, a_hat, sensor_coords, title='')
    plt.show()
    return

############################### TASK 4 OPTIONAL #########################################
def task_4_optional():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    # mat = sio.loadmat(cwd + r'/CPS_project/PT1-Modeling/src/utils/tracking_moving_targets.mat')
    mat = sio.loadmat(cwd + r'/utils/tracking_moving_targets.mat')

    A = mat['A']
    y = mat['Y']
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]
    K = y.shape[1]
    G = np.hstack((D, np.eye(q)))
    G = stats.zscore(G, axis=0)

    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam = 1
    attacked_sensors = [(11, 15)] 
    x_true = np.zeros((n, K))
    true_location = []
    true_location.append([22,35,86])

    # Set the ground truth state vector
    for loc in true_location:
        x_true[loc, 0] = 1
    # Simulate the dynamics of the targets for the entire duration K
    for i in range(K-1):
        x_true[:,i+1] = np.dot(A, x_true[:,i])
        
    # Create the vector of measurement corrupted with attacks
    y = np.zeros((q, K))
    for i in range(K):
        # Calculate the "clean" measurements
        y[:, i] = np.dot(D, x_true[:, i])
        # Add the attacks onn sensor 11 and 15
        # Attack on Sensor 11
        y[attacked_sensors[0][0], i] += 0.5 * y[attacked_sensors[0][0], i]
        # Attack on Sensor 15
        y[attacked_sensors[0][1], i] += 0.5 * y[attacked_sensors[0][1], i]

    x_hat, a_hat = observer(n, q, A, G, tau, lam, y, K)
    tracking_plot(n, true_location, x_hat, a_hat, sensor_coords, title='')
    plt.show()
    return

############################### TASK 5 ##################################################
def task_5():
    print("DISTRIBUTED SYSTEM TASK 5")
    dist_x_curves, top_names = distributed_localization()

    print("CENTRAALIZED SYSTEM TASK 5")
    cent_x_curve = centralized_localization()

    colors = ['b', 'g', 'r', 'm'] 
    cent_color = 'c'
    
    # PLOT STATE ACCURACY
    plt.figure(figsize=(12, 7))
    all_curves_x = dist_x_curves + [cent_x_curve]
    max_len_x = max(len(c) for c in all_curves_x)

    # Plot distributed curves
    for i, acc_curve in enumerate(dist_x_curves):
        # Add padding to curves
        current_len = len(acc_curve)
        pad_width = max_len_x - current_len
        
        if pad_width > 0:
            padded_acc = np.pad(acc_curve, (0, pad_width), mode='edge')
        else:
            padded_acc = acc_curve
            
        plt.plot(padded_acc, label=f"Dist. {top_names[i]}", color=colors[i % len(colors)], linewidth=0.5, alpha=0.7)
        plt.plot(current_len-1, acc_curve[-1], 'o', color=colors[i % len(colors)], alpha=0.6, markersize=3)

    # Plot centralized curves
    curr_len_c = len(cent_x_curve)
    pad_width_c = max_len_x - curr_len_c
    # Add padding to curves
    if pad_width_c > 0:
        padded_cent = np.pad(cent_x_curve, (0, pad_width_c), mode='edge')
    else:
        padded_cent = cent_x_curve

    plt.plot(padded_cent, label="Centralized (Fusion Center)", color=cent_color, linewidth=1)
    plt.plot(curr_len_c-1, cent_x_curve[-1], 'D', color=cent_color, markersize=3)

    plt.title('GRAND FINAL: State Accuracy (Distributed vs Centralized)')
    plt.xlabel('Iterations')
    plt.ylabel('Error (L2 Norm)') 
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_4_optional()
    # task_5()