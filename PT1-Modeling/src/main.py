import os as os
import numpy as np
import matplotlib.pyplot as plt
from utils import ISTA, IST
import scipy.io as sio
from scipy import stats

        
def ISTA_runs( runs, p, q, C, tau, lam, x_sparsity):
    #parameters definition with suggested settings
    q=q
    p=p

    x_tilda_supports = []
    x_estimated_supports = []
    num_iterations = []

    #running the simulation 20 times
    for _ in range(runs):        
        # generating x_tilda with 2-sparsity
        x_tilda = np.zeros(p)
        x_tilda[np.random.choice(p, x_sparsity, replace=False)] = np.random.choice([-1, 1], 2) * np.random.uniform(1, 2)
        x_tilda_supports.append (np.where(x_tilda != 0)[0])
        
        eta = 10**(-2) * np.random.randn(q)

        y = np.dot(C, x_tilda) + eta
        x_estimated, x_estimated_supp, iterations = ISTA(np.zeros(p),C,tau,lam,y)
        x_estimated_supports.append(x_estimated_supp)
        num_iterations.append(iterations)

    correct_estimations = 0
    for j in range(runs):
        if np.array_equal(x_tilda_supports[j], x_estimated_supports[j]):
            correct_estimations += 1
    return correct_estimations, num_iterations


def ISTA_runs_with_attacks(runs, n, q, C, tau, lam, x_sparsity, a_sparsity, attack_type, noisy):
    correct_estimations = 0
    num_iterations = []
    estimation_accuracy = []

    for _ in range(20,runs):
        # Generate x_tilda with standard uniform distribution        
        x_tilda = np.random.randn(n)

        # Generate the attack vector a
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
        z = z_hat[k] + (np.dot(tau, np.dot(G.T, (y[:,k] - np.dot(G, z_hat[k])))))
        gamma = tau * lam * lam_weights
        z_hat_plus = IST(z, gamma)
        x_hat.append(np.dot(A,z_hat_plus[:n]))
        a_hat.append(z_hat_plus[n:])
        z_hat.append(np.hstack((x_hat[k+1], a_hat[k+1])))
    return x_hat, a_hat

#task 1
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
    correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
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
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
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
    for i in range (0, 10):
        tau_list.append(1 / (C_l_2_norm**2) - 10**(-8) - i * 10**(-3))
    correct_estimations_percentage_q_10 = []
    max_iterations_q_10 = []
    min_iterations_q_10 = []
    mean_iterations_q_10 = []

    for tau in tau_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        print(tau)
        lam = 1 / (100*tau)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_10.append(correct_estimations*100/runs)
        max_iterations_q_10.append(np.max(num_iterations))
        min_iterations_q_10.append(np.min(num_iterations))
        mean_iterations_q_10.append(np.mean(num_iterations))
        # print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")
    
    ##### q = 20 ######
    
    #plotting correct estimations percentage
    plt.plot(tau_list, correct_estimations_percentage)
    plt.xlabel("tau")
    plt.ylabel("Correct estimations percentage")
    plt.title("q = 10 | Percentage of correct estimations in function of tau")
    plt.show()

    q = 20
    correct_estimations_percentage_q_20 = []
    max_iterations_q_20 = []
    min_iterations_q_20 = []
    mean_iterations_q_20 = []

    for tau in tau_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        lam = 1 / (100*tau)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_20.append(correct_estimations*100/runs)
        max_iterations_q_20.append(np.max(num_iterations))
        min_iterations_q_20.append(np.min(num_iterations))
        mean_iterations_q_20.append(np.mean(num_iterations))

    #plotting correct estimations percentage with both q=10 and q=20
    plt.plot(tau_list, correct_estimations_percentage_q_10, label="q=10", color='r')
    plt.plot(tau_list, correct_estimations_percentage_q_20, label="q=20", color='b')
    plt.xlabel("tau")
    plt.ylabel("Correct estimations percentage")
    plt.title("q = 20 | Percentage of correct estimations in function of tau")
    plt.legend()
    plt.grid()
    plt.show()

    #plotting min, max and mean iterations in funcrtion of tau with q=10 and q=20
    fig, axs = plt.subplots(3, figsize=(7, 10))
    fig.suptitle('Iterations in function of tau')
    axs[0].plot(tau_list, min_iterations_q_10, label="q=10", color='r')
    axs[0].plot(tau_list, min_iterations_q_20, label="q=20", color='b')
    axs[0].set_title('Min iterations')
    axs[0].legend()
    axs[1].plot(tau_list, max_iterations_q_10, label="q=10", color='r')
    axs[1].plot(tau_list, max_iterations_q_20, label="q=20", color='b')
    axs[1].legend()
    axs[1].set_title('Max iterations')
    axs[2].plot(tau_list, mean_iterations_q_10, label="q=10", color='r')
    axs[2].plot(tau_list, mean_iterations_q_20, label="q=20", color='b')
    axs[2].set_title('Mean iterations')
    axs[2].legend()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    plt.show()

    ##################################### QUESTION 5 ##############################################################
    # In order to highlight the fact that the value of q is the determining factor in the correct estimation of the support, we will conduct two analysis.
    # 1. Varying lambda keeping q equal to 10 and tau constant
    # 2. Varying lambda keeping q equal to 20 and tau constant
    # For the comparison to be fair we used the same lambda with both q==10 and q==20
    print("             QUESTION 5\nq=10 | Try different values for λ , by keeping τ constant\n")

    q = 10

    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam_list = []
    for i in range(0, 10):
        lam_list.append(1 / (100*tau) - i * 10**(-3))

    correct_estimations_percentage_q_10 = []
    max_iterations_q_10 = []
    min_iterations_q_10 = []
    mean_iterations_q_10 = []

    for lam in lam_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_10.append(correct_estimations*100/runs)
        max_iterations_q_10.append(np.max(num_iterations))
        min_iterations_q_10.append(np.min(num_iterations))
        mean_iterations_q_10.append(np.mean(num_iterations))
    
    q = 20

    correct_estimations_percentage_q_20 = []
    max_iterations_q_20 = []
    min_iterations_q_20 = []
    mean_iterations_q_20 = []

    for lam in lam_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_20.append(correct_estimations*100/runs)
        max_iterations_q_20.append(np.max(num_iterations))
        min_iterations_q_20.append(np.min(num_iterations))
        mean_iterations_q_20.append(np.mean(num_iterations))
    

    #plotting correct estimations percentage with both q=10 and q=20
    plt.plot(lam_list, correct_estimations_percentage_q_10, label="q=10", color='r')
    plt.plot(lam_list, correct_estimations_percentage_q_20, label="q=20", color='b')
    plt.xlabel("lambda")
    plt.ylabel("Correct estimations percentage")
    plt.title("Percentage of correct estimations in function of lambda")
    plt.legend()
    plt.grid()
    plt.show()

    #plotting min, max and mean iterations in funcrtion of tau with q=10 and q=20
    fig, axs = plt.subplots(3, figsize=(7, 10))
    fig.suptitle('Iterations in function of lambda')
    axs[0].plot(lam_list, min_iterations_q_10, label="q=10", color='r')
    axs[0].plot(lam_list, min_iterations_q_20, label="q=20", color='b')
    axs[0].set_title('Min iterations')
    axs[0].legend()
    axs[1].plot(lam_list, max_iterations_q_10, label="q=10", color='r')
    axs[1].plot(lam_list, max_iterations_q_20, label="q=20", color='b')
    axs[1].legend()
    axs[1].set_title('Max iterations')
    axs[2].plot(lam_list, mean_iterations_q_10, label="q=10", color='r')
    axs[2].plot(lam_list, mean_iterations_q_20, label="q=20", color='b')
    axs[2].set_title('Mean iterations')
    axs[2].legend()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
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

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=False)
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("UNAWARE - CLEAN | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=True)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("UNAWARE - NOISY | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=False)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("AWARE - CLEAN | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()

    print("Attack detection rate: ", attack_detection_rate)

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_runs_with_attacks(runs, p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=True)

    #plot the estimation accuracy in function of the number of runs
    plt.scatter(range(20,runs), estimation_accuracy, s=1, c='blue', marker='o')
    plt.xlabel("Number of runs")
    plt.ylabel("Estimation accuracy")
    plt.title("AWARE - NOISY | Estimation accuracy in function of the number of runs")
    plt.grid()
    plt.show()


############################### TASK 3 ##################################################
# Regarding the estimated state vector, our results indicated the need for refinement, as we observed more than three non-zero elements, even though their values were low. In a real-world scenario, knowing the number of targets to localize is feasible, so we selected the top three non-zero elements from the estimated $x$ vector, setting the others to 0. This approach ensured that the estimated support was accurate.

# In contrast, the attacked sensors were clearly identified without requiring any additional cleaning.
def task_3():

    cwd = os.getcwd()
    #original matrices
    mat = sio.loadmat(cwd + r'/src/utils/localization.mat')

    A = mat['A']
    y = np.squeeze(mat['y'])
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]

    G = np.hstack((D, np.eye(q)))
    #normalize G
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

    H = 10  # Grid's height (# celle)
    L = 10  # Grid's length (# celle)
    W = 100  # Cell's width (cm)

    room_grid = np.zeros((2, n))

    for i in range(n):
        room_grid[0, i] = W//2 + (i % L) * W
        room_grid[1, i] = W//2 + (i // L) * W

    # Plots
    plt.figure()
    plt.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 's', markersize=9, 
            markeredgecolor=np.array([40, 208, 220])/255, 
            markerfacecolor=np.array([40, 208, 220])/255)
    plt.grid(True)
    plt.legend(['Targets'], loc='best')
#     plt.plot(room_grid[0, estimated_attacked_sensors], room_grid[1, estimated_attacked_sensors], 's', markersize=9, 
#             markeredgecolor=np.array([255, 0, 0])/255, 
#             markerfacecolor=np.array([255, 0, 0])/255)

    plt.xticks(np.arange(100, 1001, 100))
    plt.yticks(np.arange(100, 1001, 100))
    plt.xlabel('(cm)')
    plt.ylabel('(cm)')
    plt.axis([0, 1000, 0, 1000])
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

def task_4():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    #original matrices
    mat = sio.loadmat(cwd + r'/src/utils/tracking_moving_targets.mat')
    A = mat['A']
    y = mat['Y']
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]
    K = y.shape[1]

    G = np.hstack((D, np.eye(q)))
    #normalize G
    G = stats.zscore(G, axis=0)

    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam = 1
    x_hat, a_hat = observer(n, q, A, G, tau, lam, y, K)


    # # Extract the estimated targets' location by taking the 3 greatest values of the first n elements of w_estimated
    # estimated_targets_location = np.argsort(w_estimated[:n])[-3:]

    # # Extract the estimated attacked vectors from the support of the last q eleemnts of w_estimated
    # estimated_attacked_sensors = np.where(w_estimated[n:] != 0)[0]
    
    # print("Estimated targets location: ", estimated_targets_location)
    # print("Estimated attacked sensors: ", estimated_attacked_sensors)

    H = 10  # Grid's height (# celle)
    L = 10  # Grid's length (# celle)
    W = 100  # Cell's width (cm)

    room_grid = np.zeros((2, n))
    for i in range(n):
        room_grid[0, i] = W//2 + (i % L) * W
        room_grid[1, i] = W//2 + (i // L) * W
    
    fig, ax = plt.subplots()
    true_location = []
    true_location.append([21,34,85])
    # append other 49 true locations by subtracting 1 from each element
    for i in range(49):
        true_location.append([x-1 for x in true_location[i]])

    


    for x,true_x,a in zip(x_hat,true_location, a_hat):
        estimated_targets_location = np.argsort(x)[-3:]
        estimated_attacked_sensors = np.where(a != 0)[0]
        print("Estimated attacked sensors: ", estimated_attacked_sensors)

        # Pulisci il grafico precedente
        ax.clear()

        # Plotta i nuovi dati
        ax.plot(room_grid[0, true_x], room_grid[1, true_x], 's', markersize=9, 
                markeredgecolor=np.array([40, 208, 220])/255, 
                markerfacecolor=np.array([40, 208, 220])/255)
        # update true location by subtracting 1 from each element as red circles
        ax.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 'x', markersize=9, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor=np.array([255, 255, 255])/255)


        ax.grid(True)
        ax.legend(['True Targets', 'Estimated Targets'], loc='best')

        ax.set_xticks(np.arange(100, 1001, 100))
        ax.set_yticks(np.arange(100, 1001, 100))
        ax.set_xlabel('(cm)')
        ax.set_ylabel('(cm)')
        ax.set_xlim([0, 1000])
        ax.set_ylim([0, 1000])
        ax.set_aspect('equal', adjustable='box')

        # Aggiorna la figura
        plt.pause(0.5)
        

    # Mostra il grafico finale
    plt.show()
    return 

if __name__ == "__main__":
    #task_1()
    #task_2()
    #task_3()
    task_4()