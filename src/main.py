import os as os
import numpy as np
from utils.algorithms import *

        
def ISTA_20_runs(p,q,C,tau,lam):
    #parameters definition with suggested settings
    q=q
    p=p
    i=0 #counter

    x_tilda_supports = []
    x_estimated_supports = []
    num_iterations = []

    #running the simulation 20 times
    while(i < 20):        
        # generating x_tilda with 2-sparsity
        x_tilda = np.zeros(p)
        x_tilda[np.random.choice(p, 2, replace=False)] = np.random.choice([-1, 1], 2) * np.random.uniform(1, 2)
        x_tilda_supports.append (np.where(x_tilda != 0)[0])
        
        eta = 10**(-2) * np.random.randn(q)

        y = np.dot(C, x_tilda) + eta
        x_estimated, x_estimated_supp, iterations = ISTA(np.zeros(p),C,tau,lam,y)
        x_estimated_supports.append(x_estimated_supp)
        num_iterations.append(iterations)

        i+=1
    correct_estimations = 0
    for j in range(20):
        if np.array_equal(x_tilda_supports[j], x_estimated_supports[j]):
            correct_estimations += 1
    return correct_estimations, num_iterations

#Exercise 1
def exercise_1():

    q=10
    p=20
    C = np.random.rand(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)

    print("\nFIRST EXERCISE WITH SUGGESTED PARAMETERS (q=10, p=20)\n")

    print("             QUESTION 1\n- Support recovery rate: how many times the support of x_tilda is correctly estimated?")
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , 
          "\nMin iterations = ", min(num_iterations), " || Max iterations = ", max(num_iterations), " || Mean Iterations = ", np.mean(num_iterations), "\n")
    print("             QUESTION 2\n- Can we obtain 100% of success in the support recovery by increasing q?\n")
    
    #parameters calculation with q=15
    q = 20
    C = np.random.rand(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)
    
    #simulation 
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")

    #parameters calculation with q=20
    q = 30
    C = np.random.rand(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)

    #simulation 
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")
    #parameters calculation with q=25
    q = 100
    C = np.random.rand(q, p)
    C_l_2_norm = np.linalg.norm(C, ord=2)
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    lam = 1 / (100*tau)

    #simulation
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("q = ", q, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")

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
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")
    
    #different tau
    tau = 1 / (C_l_2_norm**2) - 10**(-12)
    lam = gamma / tau

    #simulation
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("tau = ", tau, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")

    ##################################### QUESTION 5 ##############################################################
    print("             QUESTION 5\nTry different values for λ , by keeping τ constant\n")
    
    #suggested tau
    tau = 1 / (C_l_2_norm**2) - 10**(-8)
    
    #smaller lambda
    lam = 1 / (1000*tau)
    
    #simulation
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("lam = ", lam, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")

    #bigger lambda
    lam = 1 / (10*tau)

    #simulation
    correct_estimations, num_iterations = ISTA_20_runs(p, q, C, tau, lam)
    print("lam = ", lam, "|| The support of x_tilda is correctly estimated in ", correct_estimations, " out of 20 runs" , "\n")


    
exercise_1()