import os as os
import numpy as np
import matplotlib.pyplot as plt
from utils.algorithms import *
        
def ISTA( runs, p, q, C, tau, lam, x_sparsity):
    #parameters definition with suggested settings
    q=q
    p=p
    i=0 #counter

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


def ISTA_with_attacks(n, q, C, tau, lam, x_sparsity, a_sparsity, attack_type, noisy):
    correct_estimations = 0
    num_iterations = []
    estimation_accuracy = []

    for _ in range(20):
        # Generate x_tilda with standard normal distribution
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

        # Estimate xe using the weighted ISTA
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

    attack_detection_rate = correct_estimations / 20
    mean_estimation_accuracy = np.mean(estimation_accuracy)

    return attack_detection_rate, num_iterations, mean_estimation_accuracy

#task 1
def task_1():

    q=10
    p=20
    C = np.random.rand(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)
    sparsity = 2

    runs = 1000

    print("\nFIRST EXERCISE WITH SUGGESTED PARAMETERS (q=10, p=20)\n")

    print("             QUESTION 1\n- Support recovery rate: how many times the support of x_tilda is correctly estimated?")
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("The support of x_tilda is correctly estimated in ", correct_estimations, " out of", runs ," runs" , 
          "\nMin iterations = ", min(num_iterations), " || Max iterations = ", max(num_iterations), " || Mean Iterations = ", np.mean(num_iterations), "\n")
    
    # # #parameters calculation with q=15
    # q = 20
    # C = np.random.rand(q, p)
    # C_l_2_norm = np.linalg.norm(C, ord=2)
    # tau = 1 / (C_l_2_norm**2) - 10**(-8)
    # lam = 1 / (100*tau)
    
    # #simulation 
    # correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    # print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")

    # #parameters calculation with q=20
    # q = 30
    # C = np.random.rand(q, p)
    # C_l_2_norm = np.linalg.norm(C, ord=2)
    # tau = 1 / (C_l_2_norm**2) - 10**(-8)
    # lam = 1 / (100*tau)

    # #simulation 
    # correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    # print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")
    # #parameters calculation with q=25
    # q = 100
    # C = np.random.rand(q, p)
    # C_l_2_norm = np.linalg.norm(C, ord=2)
    # tau = 1 / (C_l_2_norm**2) - 10**(-8)
    # lam = 1 / (100*tau)

    # GRAPHS
    q_list = range(10, 51)
    correct_estimations_percentage = []

    print("             QUESTION 2\n- Can we obtain 100% of success in the support recovery by increasing q?\n")
    for q in q_list:
        C = np.random.rand(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        tau = 1 / (C_l_2_norm**2) - 10**(-8)
        lam = 1 / (100*tau)
        correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage.append(correct_estimations*100/runs)
        print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")
    #plotting the results
    plt.plot(q_list, correct_estimations_percentage)
    plt.xlabel("q")
    plt.ylabel("Correct estimations percentage")
    plt.title("Percentage of correct estimations in function of q")
    plt.show()



    #simulation
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")

    ##################################### QUESTION 4 ##############################################################

    print("             QUESTION 4\nTry different values for τ , by keeping τλ constant\n")
    
    #Suggested settings
    q = 10
    C = np.random.rand(q, p)                
    C_l_2_norm = np.linalg.norm(C, ord=2)
    
    #tau calculation
    gamma = tau * lam
    tau = 1 / (C_l_2_norm**2) - 10**(-10)
    lam = gamma / tau

    #simulation 
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")
    
    #different tau
    tau = 1 / (C_l_2_norm**2) - 10**(-12)
    lam = gamma / tau

    #simulation
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")

    ##################################### QUESTION 5 ##############################################################
    print("             QUESTION 5\nTry different values for λ , by keeping τ constant\n")
    
    #suggested tau
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    
    #smaller lambda
    lam = 1 / (1000*tau)
    
    #simulation
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("lam = ", lam, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")

    #bigger lambda
    lam = 1 / (10*tau)

    #simulation
    correct_estimations, num_iterations = ISTA(runs, p, q, C, tau, lam, sparsity)
    print("lam = ", lam, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of ", runs, " runs" , "\n")


    

# task_2
def task_2():
    q=20
    p=10
    x_sparsity = p
    a_sparsity = 2
    C = np.random.rand(q, p)
    #calculate tau as a vector with q zeroes and p ones
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 2 * 10**(-3) / tau

    print("\nSECOND EXERCISE WITH SUGGESTED PARAMETERS (q=10, p=20)\n")

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_with_attacks(p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=False)
    print("| UNAWARE - CLEAN | \n1. The rate of attack detection is: ", attack_detection_rate, " calculated over 20 run\n2. The mean estimation accuracy is: ", estimation_accuracy, "\n")

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_with_attacks(p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=False)
    print("| AWARE - CLEAN | \n1. The rate of attack detection is: ", attack_detection_rate, " calculated over 20 run\n2.  The mean estimation accuracy is: ", estimation_accuracy, "\n")

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_with_attacks(p, q, C, tau, lam,x_sparsity,a_sparsity, "UNAWARE", noisy=True)
    print("| UNAWARE - NOISY | \n1. The rate of attack detection is: ", attack_detection_rate, " calculated over 20 run\n2. The mean estimation accuracy is: ", estimation_accuracy, "\n")

    attack_detection_rate, num_iterations, estimation_accuracy  = ISTA_with_attacks(p, q, C, tau, lam,x_sparsity,a_sparsity, "AWARE", noisy=True)
    print("| AWARE - CLEAN | \n1. The rate of attack detection is: ", attack_detection_rate, " calculated over 20 run\n2. The mean estimation accuracy is: ", estimation_accuracy, "\n")


task_1()
#task_2()