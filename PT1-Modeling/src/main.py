import random
from utils.utilities import *
import scipy.io as sio
from scipy import stats


############################### TASK 1 ##################################################
def task_1():
    print('\n========== TASK 1 ==========')

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
    # plotting iterations
    plt.figure()
    plt.yscale('log')
    plt.plot(q_list, min_iterations, color='b', label="Min")
    plt.plot(q_list, max_iterations, color='g', label="Max")
    plt.plot(q_list, mean_iterations, color='m', label="Mean")
    plt.xlabel("q")
    plt.ylabel("Number of iterations")
    plt.title("Convergence time in function of q")
    plt.legend()
    plt.grid()
    plt.show()

    ##################################### QUESTION 4 ##############################################################
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
    print('\n========== TASK 2 ==========')

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
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam, x_sparsity, a_sparsity, "UNAWARE", noisy=False)
    
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("UNAWARE - CLEAN | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    # Case with unaware attacks and with noise
    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam, x_sparsity, a_sparsity, "UNAWARE", noisy=True)

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
    print('\n========== TASK 3 ==========')

    true_location = []
    true_attacked_sensors = []
    true_location.append([22,35,86])
    true_attacked_sensors.append([11,15])
    cwd = os.getcwd()
    # original matrices
    mat = sio.loadmat(cwd + r'/../data/localization.mat')

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

    # Extract the estimated attacked vectors from the support of the last q eleemnts of w_estimated
    estimated_attacked_sensors = np.where(w_estimated[n:] != 0)[0]
    
    print("Estimated targets location: ", estimated_targets_location)
    print("Estimated attacked sensors: ", estimated_attacked_sensors)

    localization_plot(true_location, true_attacked_sensors, estimated_targets_location, estimated_attacked_sensors)
    plt.show()

############################### TASK 4 ##################################################
def task_4():

    #################################################### MANDATORY PART ###############################################
    print('\n===== TASK 4 - MANDATORY PART =====')

    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    mat = sio.loadmat(cwd + r'/../data/tracking_moving_targets.mat')

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

    true_location = []
    attacked_sensors = [(11, 15)]
    x_true = np.zeros((K,n))
    true_location.append([21,34,85]) # Changed targets values due to python and matlab mismatch (-1 index)

    attack_threshold_percentage = 0.35 #threshold percentage of the maximum estimated value (for a) after which the sensor is considered attacked

    # Set the ground truth state vector
    for loc in true_location:
        x_true[0, loc] = 1
    # Simulate the dynamics of the targets for the entire duration K
    for i in range(K-1):
        x_true[i+1,:] = np.dot(A, x_true[i,:])


    # Calculates estimates
    x_hat, a_hat = sparse_observer(n, q, A, G, tau, lam, y, K)

    # Check convergence
    _, _, _, _, estimated_targets, estimated_attacked_sensors = check_convergence(x_hat, a_hat, x_true, attacked_sensors, n_targets=3, attack_threshold_percentage=attack_threshold_percentage)

    # Plot results
    tracking_plot(n, true_location, estimated_targets, estimated_attacked_sensors, true_attacked_sensors=K*attacked_sensors, title='')

    plt.show()



    ############################# OPTIONAL TASK PART 1 - Aware time-invariant attacks ##################################
    print('\n===== OPTIONAL TASK PART 1 - Aware time-invariant attacks =====')
    # Create the vector of measurement corrupted with attacks
    y = np.zeros((q, K))
    for i in range(K):
        eta = 10**(-2) * np.random.randn()
        # Calculate the "clean" measurements
        y[:, i] = np.dot(D, x_true[i, :]) + eta

        # AWARE attacks
        y[attacked_sensors[0][0], i] += 0.5 * y[attacked_sensors[0][0], i]
        y[attacked_sensors[0][1], i] += 0.5 * y[attacked_sensors[0][1], i]

    x_hat, a_hat = sparse_observer(n, q, A, G, tau, lam, y, K)
    _, _, _, _, estimated_targets, estimated_attacked_sensors = check_convergence(x_hat, a_hat, x_true, attacked_sensors, n_targets=3, attack_threshold_percentage=attack_threshold_percentage)

    tracking_plot(n, true_location, estimated_targets, estimated_attacked_sensors, true_attacked_sensors=K * [attacked_sensors[0]],
                  title='OPTIONAL TASK PART 1 WITH AWARE ATTACKS')
    plt.show()



    ############################# OPTIONAL TASK PART 2 - Aware time-varying attacks ##################################
    print('\n===== OPTIONAL TASK PART 2 - Aware time-varying attacks =====')
    attacked_sensors = [(11, 15), (8, 17)]

    # 2 phases -> 2 couples of attacked sensors, attack is held for K/2
    num_phases = len(attacked_sensors)
    interval_step = K // num_phases
    y = np.zeros((q, K))
    for i in range(K):
        # Calculate the "clean" measurements
        eta = 10**(-2) * np.random.randn()
        y[:, i] = np.dot(D, x_true[i, :]) + eta

        # Change the attacked sensors for the second half iterations
        current_phase_idx = i // interval_step
        current_phase_idx = min(current_phase_idx, num_phases - 1)
        current_sensors = attacked_sensors[current_phase_idx]

        # AWARE attacks
        y[current_sensors[0], i] += 0.5 * y[current_sensors[0], i]
        y[current_sensors[1], i] += 0.5 * y[current_sensors[1], i]

    # Calculate estimates
    x_hat, a_hat = sparse_observer(n, q, A, G, tau, lam, y, K)
    # Calculate convergence
    _, _, _, _, estimated_targets, estimated_attacked_sensors = check_convergence(x_hat, a_hat, x_true, attacked_sensors, n_targets=3, attack_threshold_percentage=attack_threshold_percentage, n_change_attacks=len(attacked_sensors))

    # Plot
    tracking_plot(n, true_location, estimated_targets, estimated_attacked_sensors,
                  true_attacked_sensors=int(K / 2) * [attacked_sensors[0]] + int(K / 2) * [attacked_sensors[1]],
                  title='OPTIONAL TASK PART 2')
    plt.show()


    ################### OPTIONAL TASK PART 3 - Aware vs Unaware attacks on increasing number of sensors #########################
    print('===== OPTIONAL TASK PART 3 - Aware vs Unaware attacks on increasing number of sensors =====')

   

    for attack_type in ["AWARE", "UNAWARE"]:
        print("======= Attack type: ", attack_type, " =======")
        # Attack sensors in random order
        attacked_sensors = []
        all_sensors = [i for i in range(25)]
        random.seed(42)
        random.shuffle(all_sensors)

        for _ in range(2):
            attacked_sensors.append(all_sensors.pop(0))

        # continue attacks until all sensors are under attack
        x_convergence_iterations = []
        a_convergence_iterations= []
        while len(attacked_sensors) <= q:
            print(f"\n--- Iteration with {len(attacked_sensors)} Attacked Sensors: {attacked_sensors} ---")

            y = np.zeros((q, K))
            for i in range(K):
                eta = 10**(-2) * np.random.randn()
                y[:, i] = np.dot(D, x_true[i, :]) + eta
                for sensor_idx in attacked_sensors:
                    # Aware => a==0.5y
                    if attack_type == 'AWARE':
                        a = 0.5 * y[sensor_idx, i]
                    
                    # Unaware -> a== random between 90% and 110% of the mean of the measurements
                    else:
                        y_mean = np.mean(y[:, i])
                        a = np.random.uniform(0.9 * y_mean, 1.1 * y_mean)

                    y[sensor_idx, i] += a

            # Calculates estimates
            x_hat, a_hat = sparse_observer(n, q, A, G, tau, lam, y, K)
            (x_converged,
             a_converged,
             x_iteration,
             a_iteration,
             estimated_targets,
             estimated_attacked_sensors) = check_convergence(x_hat, a_hat, x_true, attacked_sensors, n_targets=3, attack_threshold_percentage=attack_threshold_percentage)
            
            x_convergence_iterations.append(x_iteration if x_converged else None)
            a_convergence_iterations.append(a_iteration.pop() if a_converged else None)


            tracking_plot(n, true_location, estimated_targets, estimated_attacked_sensors, true_attacked_sensors=K *[attacked_sensors],
                           n_attacks=len(attacked_sensors), title='OPTIONAL TASK PART 3')

            plt.show()

            if len(attacked_sensors) == q:
                break

            next_sensor_to_attack = all_sensors.pop(0)
            attacked_sensors.append(next_sensor_to_attack)
            print('\n===========================================================\n')

        plot_convergence_iterations(attacked_sensors, x_convergence_iterations, a_convergence_iterations, attack_type)
    return

############################### TASK 5 ##################################################
def task_5():
    print('\n========== TASK 5 ==========')

    ########################### CENTRALIZED ISTA (Just for comparison) #######################
    print("CENTRALIZED SYSTEM TASK 5")
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()

    mat = sio.loadmat(cwd + r'/../data/distributed_localization_data.mat')
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

    z_est, support, stop_iter = Localization_with_attacks(n, q, G, tau, lam_scalar, y, lam_2=0.1, delta=10**-8, norm_exp=2)

    print(f"Centralized localization converged at iteration: {stop_iter}")

    x_est = z_est[:n]
    a_est = z_est[n:]
    a_est_refined = np.copy(a_est)
    a_est_refined[np.abs(a_est_refined) < attack_threshold] = 0

    estimated_targets_location = np.argsort(x_est)[-2:]
    estimated_attacked_sensors = np.where(a_est_refined != 0)[0]

    print(f"   Estimated Targets: {estimated_targets_location} (True: {true_location})")
    print(f"   Estimated Attacks: {estimated_attacked_sensors} (True: {true_attack_indices})")

    print('\n --------------------------------------------------- \n')
    print("DISTRIBUTED SYSTEM TASK 5")

    ############################################## DISTA  ##########################################

    Q12 = mat['Q_12']
    Q18 = mat['Q_18']
    Q4 = mat['Q_4']
    Q8 = mat['Q_8']
    matrices_list = [Q4, Q8, Q12, Q18]
    topologies_names = ["TOPOLOGY 1 (Q4)", "TOPOLOGY 2 (Q8)", "TOPOLOGY 3 (Q12)", "TOPOLOGY 4 (Q18)"]
    tau = 4e-7
    lam_vec = np.concatenate((np.full(n, 10), np.full(q, 0.1)))

    # Parameters
    attack_threshold = 0.002 # DISTA needs a bigger threshold

    # --- LOOP OVER ALL TOPOLOGIES ---
    for i, Q_curr in enumerate(matrices_list):
        print(f"--- {topologies_names[i]} ---")

        # Eigenvalue analysis
        evals = np.abs(np.linalg.eigvals(Q_curr))
        lambda_2 = np.sort(evals)[::-1][1]
        print(f"   |lambda_2|: {lambda_2:.5f}")
        iterations = 15000

        # Run DISTA
        z_nodes, stop_criteria_iter, k_x_cons, k_a_cons, k_x_conv, k_a_conv = DISTA(n, q, D, y, Q_curr, tau,
                                                                                                lam_vec, true_location,
                                                                                                true_attack_indices,
                                                                                                attack_threshold,
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

        localization_plot(true_location, true_attack_indices, estimated_targets_location, estimated_attacked_sensors,
                          title=f"{topologies_names[i]}\nStop criteria reached at iter: {stop_criteria_iter}")
        print('\n --------------------------------------------------- \n')


    return


if __name__ == "__main__":
    task_1()
    task_2()
    task_3()
    task_4()
    task_5()