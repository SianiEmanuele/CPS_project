import os as os
import numpy as np
import matplotlib.pyplot as plt
from utils import ISTA, IST
import scipy.io as sio
from scipy import stats
import networkx as nx

        
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
    return correct_estimations, num_iterations

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
        z = z_hat[k] + (np.dot(tau, np.dot(G.T, (y[:,k] - np.dot(G, z_hat[k])))))
        gamma = tau * lam * lam_weights
        z_hat_plus = IST(z, gamma)
        x_hat.append(np.dot(A,z_hat_plus[:n]))
        a_hat.append(z_hat_plus[n:])
        z_hat.append(np.hstack((x_hat[k+1], a_hat[k+1])))
    return x_hat, a_hat

def plot_abstract_comparison(matrices):
    """
    Plots the abstract topology of sensor networks based on adjacency or stochastic matrices.
    Each topology is displayed in a separate, independent pop-up window using a force-directed layout.

    Args:
        matrices (list of np.array): A list containing the Q matrices (stochastic/adjacency matrices).
                                     Each matrix represents the connectivity of a specific network topology.
                                     Shape of each matrix should be (N, N), where N is the number of sensors.

    Returns:
        None: The function generates and displays matplotlib figures directly.
    """
    
    titles = ["Topologia 4", "Topologia 8", "Topologia 12", "Topologia 18"]

    # Iterate through each matrix in the provided list
    for i, Q in enumerate(matrices):
        
        # 1. Create a NEW dedicated window for each iteration
        # The 'i' argument assigns a unique ID to the figure
        plt.figure(i, figsize=(10, 8)) 
        
        # 2. Create the Directed Graph object from the numpy matrix
        # create_using=nx.DiGraph ensures edge direction is preserved (asymmetric connections)
        G_raw = nx.from_numpy_array(Q.T, create_using=nx.DiGraph)
        
        # 3. Node Relabeling (0-based to 1-based)
        # Create a mapping dictionary: {0: 1, 1: 2, ..., 24: 25}
        mapping = {node: node + 1 for node in G_raw.nodes()}
        G = nx.relabel_nodes(G_raw, mapping)
        
        # 4. Layout Calculation
        # 'spring_layout' positions nodes to minimize edge crossing.
        # 'k' parameter controls the distance between nodes (higher k = more spread out).
        # 'seed' fixes the randomness so different runs look the same.
        pos = nx.spring_layout(G, seed=42, k=0.9)
        
        # Set window title
        plt.title(titles[i], fontsize=14, fontweight='bold')
        
        # --- DRAWING PHASE ---
        
        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, 
                               node_color='lightblue', 
                               node_size=600, 
                               edgecolors='black')
        
        # Draw Node Labels (Numbers inside circles)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Draw Edges (Arrows)
        # 'connectionstyle' creates curved edges, essential to see bidirectional links (A<->B) separately.
        nx.draw_networkx_edges(G, pos, 
                               edge_color='gray', 
                               arrowstyle='->', 
                               arrowsize=20,
                               connectionstyle='arc3, rad=0.1')
        
        # 5. Draw Edge Weights
        # Extract weights from the graph attributes
        edge_labels = nx.get_edge_attributes(G, 'weight')
        
        # Format weights to 2 decimal places to keep the plot clean
        # Filter out very small weights (effectively zero) if any exist
        formatted_edge_labels = {edge: f"{weight:.2f}" 
                                 for edge, weight in edge_labels.items() 
                                 if weight > 0.001}
        
        # Draw the labels on the edges
        # label_pos=0.3 places the text closer to the source, avoiding overlap with the arrowhead
        # nx.draw_networkx_edge_labels(G, pos, 
        #                              edge_labels=formatted_edge_labels,
        #                              font_size=8,
        #                              label_pos=0.3)
        
        # Turn off axis (coordinate system is abstract, not physical)
        plt.axis('off')

    # Show all generated windows
    plt.show()
    return

def check_strong_connectivity(matrices):
    """
    Checks if the directed graphs associated with the provided stochastic matrices 
    are Strongly Connected.
    
    A directed graph is strongly connected if every vertex is reachable from 
    every other vertex. This condition ensures there are no isolated nodes and 
    that consensus can be reached (information flows globally).

    Args:
        matrices (list of np.array): A list containing the Q matrices.
                                     Assumes Q[i, j] is the weight for the link j -> i (Row-Stochastic).

    Returns:
        list of bool: A list containing True if the corresponding topology is strongly connected, False otherwise.
    """
    
    results = []
    print("--- Strong Connectivity Check ---")

    for i, Q in enumerate(matrices):
        # 1. Graph Creation
        # We use Q.T to correctly represent the flow direction (Sender -> Receiver)
        G = nx.from_numpy_array(Q.T, create_using=nx.DiGraph)
        
        # 2. Check Strong Connectivity
        # nx.is_strongly_connected returns True if every node can reach every other node
        is_strongly_connected = nx.is_strongly_connected(G)
        results.append(is_strongly_connected)
        
        status = "PASSED" if is_strongly_connected else "FAILED"
        print(f"Topology {i+1}: {status}")

        # 3. Detailed Diagnostics (if check fails)
        if not is_strongly_connected:
            # Check for completely isolated nodes (degree 0)
            isolates = list(nx.isolates(G))
            if isolates:
                # Adjust index to be 1-based for readability
                isolates_1based = [n + 1 for n in isolates]
                print(f"   -> Warning: Nodes {isolates_1based} are completely isolated (no incoming or outgoing links).")
            
            # Check number of Strongly Connected Components (SCCs)
            # If > 1, the graph is partitioned into islands that don't talk to each other
            num_sccs = nx.number_strongly_connected_components(G)
            print(f"   -> Graph is partitioned into {num_sccs} separate components.")
            
    print("---------------------------------")
    return results

def check_doubly_stochastic(matrices):
    print("--- Doubly Stochastic Check ---")
    
    for i, Q in enumerate(matrices):
        # 1. Controllo Righe (Row Stochastic)
        # axis=1 somma lungo le righe
        row_sums = np.sum(Q, axis=1)
        is_row_stoch = np.allclose(row_sums, 1) # Usa allclose per evitare errori di arrotondamento float
        
        # 2. Controllo Colonne (Column Stochastic)
        # axis=0 somma lungo le colonne
        col_sums = np.sum(Q, axis=0)
        is_col_stoch = np.allclose(col_sums, 1)
        
        # 3. Verdetto
        if is_row_stoch and is_col_stoch:
            print(f"Topology {i+1}: DOUBLY Stochastic (Converges to Average)")
        elif is_row_stoch:
            print(f"Topology {i+1}: ROW Stochastic only (Converges to Weighted Value)")
        elif is_col_stoch:
            print(f"Topology {i+1}: COLUMN Stochastic only (Unstable for standard consensus)")
        else:
            print(f"Topology {i+1}: NOT Stochastic")
            
    print("-------------------------------")
    return

def sufficient_condition_consensus(matrices):
    for i, Q in enumerate(matrices):
        eigenvalues = np.linalg.eigvals(Q)
        abs_eigenvalues = np.abs(eigenvalues)
        # Ordina in modo decrescente (dal più grande al più piccolo)
        # [::-1] serve a invertire l'array sortato (che di base è crescente)
        sorted_evals = np.sort(abs_eigenvalues)[::-1]
        # Estrarre i due più grandi
        lambda_1 = sorted_evals[0]
        lambda_2 = sorted_evals[1]
        
        print(f"\nTopologia {i+1}:")
        print(f"   -> 1° Autovalore (|λ1|): {lambda_1:.6f} (Dovrebbe essere 1.0)")
        print(f"   -> 2° Autovalore (|λ2|): {lambda_2:.6f}")
        
        # Check if there is sufficient condition for convergence λ1 = 1 && |λ1|>|λ2|>=...>=|λq|
        if np.allclose(lambda_1, 1):
            print("THE SYSTEM CONVERGES")
        else:
            print("THE SYSTEM DOES NOT CONVERGE")
            return

        # Check Row Stochastic
        # axis=1 somma lungo le righe
        row_sums = np.sum(Q, axis=1)
        is_row_stoch = np.allclose(row_sums, 1) # Usa allclose per evitare errori di arrotondamento float
        
        # Check Column Stochastic
        # axis=0 somma lungo le colonne
        col_sums = np.sum(Q, axis=0)
        is_col_stoch = np.allclose(col_sums, 1)
        
        # Q characteristic
        if is_row_stoch and is_col_stoch:
            print(f"Topology {i+1}: DOUBLY Stochastic (Converges to Average)")
        elif is_row_stoch:
            print(f"Topology {i+1}: ROW Stochastic only (Converges to Weighted Value)")
        elif is_col_stoch:
            print(f"Topology {i+1}: COLUMN Stochastic only (Unstable for standard consensus)")
        else:
            print(f"Topology {i+1}: NOT Stochastic")

        # Convergence rate
        rho = lambda_2
        
        if np.isclose(rho, 1.0):
            print("   -> THE SYSTEM DOES NOT CONVERGE")
        elif rho > 0.9:
            print(f"   -> SLOW CONVERGENCE (Rate: {rho:.4f}).")
        elif rho < 0.5:
             print(f"   -> FAST CONVERGENCE (Rate: {rho:.4f}).")
        else:
             print(f"   -> MODERATE CONVERGENCE (Rate: {rho:.4f}).")
        print("\n--------------------------------------------------")
    return

def DISTA(n, q, D, y, Q, tau, lam_vec, max_iter=1000, tol=1e-8):
    """
    Implements the Distributed ISTA (DISTA) algorithm (Algorithm 3).
    
    Returns:
        z_nodes (np.array): Final estimates matrix (q x (n+q)).
        k (int): The iteration number where convergence was reached.
    """
    
    # 1. Initialization
    z_nodes = np.zeros((q, n + q)) 
    
    # Pre-compute local augmented matrices G_i
    G_list = []
    for i in range(q):
        e_i = np.zeros(q)
        e_i[i] = 1 
        G_i = np.hstack((D[i, :], e_i)) 
        G_list.append(G_i)

    print(f"   -> Starting DISTA (Max Iter: {max_iter})...")

    # 2. Main Loop
    for k in range(max_iter):
        z_prev = np.copy(z_nodes)
        z_new = np.zeros_like(z_nodes)
        
        # Consensus Step (Matrix Multiplication for efficiency)
        consensus_block = np.dot(Q, z_prev) 

        # Local Update Loop
        for i in range(q):
            G_i = G_list[i]
            y_i = y[i]
            z_i_k = z_prev[i, :]
            
            # --- CORREZIONE GRADIENTE (Fix Shape Mismatch) ---
            # Calcolo residuo scalare
            residual = y_i - np.dot(G_i, z_i_k)
            
            # Moltiplicazione scalare per il vettore G_i
            gradient_correction = tau * G_i * residual
            # -------------------------------------------------
            
            # Combine
            v = consensus_block[i, :] + gradient_correction
            
            # Soft Thresholding
            z_new[i, :] = IST(v, tau * lam_vec)
            
        # Stop Criterion
        diff_norm = np.sum([np.linalg.norm(z_new[i] - z_prev[i],2) for i in range(q)])
        
        # --- CORREZIONE RETURN (Fix Unpack Error) ---
        if diff_norm < tol:
            return z_new, k # Restituisce DUE valori se converge
        
        z_nodes = z_new
        
        if k > 0 and k % 5000 == 0:
            print(f"      Iter {k}: Diff Norm {diff_norm:.2e}")
            
    return z_nodes, max_iter # Restituisce DUE valori se finisce i cicli

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
    
    #plotting correct estimations percentage
    plt.plot(tau_list, correct_estimations_percentage)
    plt.xlabel("tau")
    plt.ylabel("Correct estimations percentage")
    plt.title("q = 10 | Percentage of correct estimations in function of tau")
    plt.show()

    ##### q = 24 ######
    q = 24
    correct_estimations_percentage_q_24 = []
    max_iterations_q_24 = []
    min_iterations_q_24 = []
    mean_iterations_q_24 = []

    for tau in tau_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        lam = 1 / (100*tau)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_24.append(correct_estimations*100/runs)
        max_iterations_q_24.append(np.max(num_iterations))
        min_iterations_q_24.append(np.min(num_iterations))
        mean_iterations_q_24.append(np.mean(num_iterations))

    #plotting correct estimations percentage with both q=10 and q=24
    plt.plot(tau_list, correct_estimations_percentage_q_10, label="q=10", color='r')
    plt.plot(tau_list, correct_estimations_percentage_q_24, label="q=24", color='b')
    plt.xlabel("tau")
    plt.ylabel("Correct estimations percentage")
    plt.title("q = 24 | Percentage of correct estimations in function of tau")
    plt.legend()
    plt.grid()
    plt.show()

    #plotting min, max and mean iterations in funcrtion of tau with q=10 and q=24
    fig, axs = plt.subplots(3, figsize=(7, 10))
    fig.suptitle('Iterations in function of tau')
    axs[0].plot(tau_list, min_iterations_q_10, label="q=10", color='r')
    axs[0].plot(tau_list, min_iterations_q_24, label="q=24", color='b')
    axs[0].set_title('Min iterations')
    axs[0].legend()
    axs[1].plot(tau_list, max_iterations_q_10, label="q=10", color='r')
    axs[1].plot(tau_list, max_iterations_q_24, label="q=24", color='b')
    axs[1].legend()
    axs[1].set_title('Max iterations')
    axs[2].plot(tau_list, mean_iterations_q_10, label="q=10", color='r')
    axs[2].plot(tau_list, mean_iterations_q_24, label="q=24", color='b')
    axs[2].set_title('Mean iterations')
    axs[2].legend()
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
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
    
    q = 24

    correct_estimations_percentage_q_24 = []
    max_iterations_q_24 = []
    min_iterations_q_24 = []
    mean_iterations_q_24 = []

    for lam in lam_list:
        C = np.random.randn(q, p)
        C_l_2_norm = np.linalg.norm(C, ord=2)
        correct_estimations, num_iterations = ISTA_runs(runs, p, q, C, tau, lam, sparsity)
        correct_estimations_percentage_q_24.append(correct_estimations*100/runs)
        max_iterations_q_24.append(np.max(num_iterations))
        min_iterations_q_24.append(np.min(num_iterations))
        mean_iterations_q_24.append(np.mean(num_iterations))
    

    #plotting correct estimations percentage with both q=10 and q=24
    plt.plot(lam_list, correct_estimations_percentage_q_10, label="q=10", color='r')
    plt.plot(lam_list, correct_estimations_percentage_q_24, label="q=24", color='b')
    plt.xlabel("lambda")
    plt.ylabel("Correct estimations percentage")
    plt.title("Percentage of correct estimations in function of lambda")
    plt.legend()
    plt.grid()
    plt.show()

    #plotting min, max and mean iterations in funcrtion of tau with q=10 and q=24
    fig, axs = plt.subplots(3, figsize=(7, 10))
    fig.suptitle('Iterations in function of lambda')
    axs[0].plot(lam_list, min_iterations_q_10, label="q=10", color='r')
    axs[0].plot(lam_list, min_iterations_q_24, label="q=24", color='b')
    axs[0].set_title('Min iterations')
    axs[0].legend()
    axs[1].plot(lam_list, max_iterations_q_10, label="q=10", color='r')
    axs[1].plot(lam_list, max_iterations_q_24, label="q=24", color='b')
    axs[1].legend()
    axs[1].set_title('Max iterations')
    axs[2].plot(lam_list, mean_iterations_q_10, label="q=10", color='r')
    axs[2].plot(lam_list, mean_iterations_q_24, label="q=24", color='b')
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
# Regarding the estimated state vector, our results indicated the need for refinement, as we observed more than three non-zero elements, even though their values were low. In a real-world scenario, knowing the number of targets to localize is feasible, so we selected the top three non-zero elements from the estimated $x$ vector, setting the others to 0. This approach ensured that the estimated support was accurate.

# In contrast, the attacked sensors were clearly identified without requiring any additional cleaning.
def task_3():

    sensor_coords = np.array([
        [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
        [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
        [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
    ])
    true_location = []
    true_location.append([22,35,86])
    cwd = os.getcwd()
    #original matrices
    mat = sio.loadmat(cwd + r'/utils/localization.mat')

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
    plt.grid(True)

    # True targets plot
    plt.plot(room_grid[0, true_location], room_grid[1, true_location], 's', markersize=9, 
            markeredgecolor=np.array([40, 208, 220])/255, 
            markerfacecolor=np.array([40, 208, 220])/255)    
    # Estimated targets plot
    plt.plot(room_grid[0, estimated_targets_location], room_grid[1, estimated_targets_location], 'x', markersize=9, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor=np.array([255, 255, 255])/255)

    # Sensors plot
    plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')
    
    # Attacked sensors plot
    plt.plot(sensor_coords[estimated_attacked_sensors[0], 0], sensor_coords[estimated_attacked_sensors[0], 1], 'o', markersize=12, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor='none')
    plt.plot(sensor_coords[estimated_attacked_sensors[1], 0], sensor_coords[estimated_attacked_sensors[1], 1], 'o', markersize=12, 
            markeredgecolor=np.array([255, 0, 0])/255, 
            markerfacecolor='none')

    plt.xticks(np.arange(100, 1001, 100))
    plt.yticks(np.arange(100, 1001, 100))
    plt.xlabel('(cm)')
    plt.ylabel('(cm)')
    plt.axis([0, 1000, 0, 1000])
    plt.legend(['True Targets', 'Estimated Targets', 'Sensors', 'Attacked sensors'], loc='best')
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()

############################### TASK 4 ##################################################
def task_4():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    #original matrices
    # mat = sio.loadmat(cwd + r'/CPS_project/PT1-Modeling/src/utils/tracking_moving_targets.mat')
    mat = sio.loadmat(cwd + r'/utils/tracking_moving_targets.mat')

    A = mat['A']
    y = mat['Y']
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]
    K = y.shape[1]
    sensor_coords = np.array([
        [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
        [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
        [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
    ])

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
    true_location.append([22,35,86])
    # append other 49 true locations by subtracting 1 from each element
    for i in range(50):
        true_location.append([x-1 for x in true_location[i]])
    
    # skip first element
    true_location = true_location[1:]

    for x,true_x,a in zip(x_hat,true_location, a_hat):
        estimated_targets_location = np.argsort(x)[-3:]
        estimated_attacked_sensors = np.argsort(a)[-2:]
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

        # Plot of sensors
        ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')

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

        # Aggiorna la figura
        plt.pause(0.5)
        

    # Mostra il grafico finale
    plt.show()
    return

############################### TASK 4 OPTIONAL #########################################
def task_4_optional():
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    #original matrices
    # mat = sio.loadmat(cwd + r'/CPS_project/PT1-Modeling/src/utils/tracking_moving_targets.mat')
    mat = sio.loadmat(cwd + r'/utils/tracking_moving_targets.mat')

    A = mat['A']
    y = mat['Y']
    D = mat['D']
    n = D.shape[1]
    q = D.shape[0]
    K = y.shape[1]
    sensor_coords = np.array([
        [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
        [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
        [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
    ])

    G = np.hstack((D, np.eye(q)))
    #normalize G
    G = stats.zscore(G, axis=0)

    tau = 1 / (np.linalg.norm(G, ord=2)**2) - 10**(-8)
    lam = 1

    attacked_sensors = [(11, 15)] 

    x_true = np.zeros((n, K))

    true_location = []
    true_location.append([22,35,86])
    # Set the initial state vector: place a '1' in the cells occupied by targets
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
        # Add the attacks
        # Attack on Sensor 11
        y[attacked_sensors[0][0], i] += 0.5 * y[attacked_sensors[0][0], i]
        # Attack on Sensor 15
        y[attacked_sensors[0][1], i] += 0.5 * y[attacked_sensors[0][1], i]

    x_hat, a_hat = observer(n, q, A, G, tau, lam, y, K)

    H = 10  # Grid's height (# celle)
    L = 10  # Grid's length (# celle)
    W = 100  # Cell's width (cm)

    room_grid = np.zeros((2, n))
    for i in range(n):
        room_grid[0, i] = W//2 + (i % L) * W
        room_grid[1, i] = W//2 + (i // L) * W
    
    fig, ax = plt.subplots()

    # append other 49 true locations by subtracting 1 from each element
    for i in range(50):
        true_location.append([x-1 for x in true_location[i]])

    for x,true_x,a in zip(x_hat,true_location, a_hat):
        estimated_targets_location = np.argsort(x)[-3:]
        estimated_attacked_sensors = np.argsort(np.abs(a))[-2:]
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

        # Plot of sensors
        ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')

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

        # Aggiorna la figura
        plt.pause(0.5)
        

    # Mostra il grafico finale
    plt.show()
    return

############################### TASK 5 ##################################################
def task_5():
    """
    Task 5: Distributed target localization under sparse sensor attacks using DISTA.
    
    This function:
    1. Loads distributed data (y, D, Q matrices).
    2. Analyzes the connectivity of the provided network topologies (Eigenvalues).
    3. Solves the localization problem using the Distributed ISTA (DISTA) algorithm.
    4. Refines the attack vector estimation.
    5. Plots the results for a selected valid topology.
    """
    np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
    cwd = os.getcwd()
    
    # Load Data
    mat = sio.loadmat(cwd + r'/utils/distributed_localization_data.mat')
    # IMPORTANT: Squeeze y to make it shape (25,) instead of (25,1)
    y = np.squeeze(mat['y']) 
    D = mat['D']
    
    # Load Topologies
    Q12 = mat['Q_12']
    Q18 = mat['Q_18']
    Q4 = mat['Q_4']
    Q8 = mat['Q_8']
    
    matrices_list = [Q4, Q8, Q12, Q18] 
    topo_names = ["Topology 1 (Q4)", "Topology 2 (Q8)", "Topology 3 (Q12)", "Topology 4 (Q18)"]

    n = D.shape[1] 
    q = D.shape[0] 

    # Sensor Coordinates
    sensor_coords = np.array([
        [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
        [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
        [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
    ])
    
    # Ground Truth Indices (0-based)
    # Based on text solution: supp(x)={14,25} -> [13, 24]
    # Based on text solution: supp(a)={8,23}  -> [7, 22]
    true_target_indices = [13, 24]
    true_attack_indices = [7, 22]

    # Parameters
    tau = 4e-7
    lam_vec = np.concatenate((np.full(n, 10), np.full(q, 0.1)))
    attack_threshold = 0.002
    
    # --- LOOP OVER ALL TOPOLOGIES ---
    print("\n================ STARTING SIMULATION FOR ALL TOPOLOGIES ================")
    
    for i, Q_curr in enumerate(matrices_list):
        print(f"\n--- Testing {topo_names[i]} ---")
        
        # 1. Analyze Eigenvalues
        evals = np.abs(np.linalg.eigvals(Q_curr))
        lambda_2 = np.sort(evals)[::-1][1]
        print(f"   |lambda_2|: {lambda_2:.5f} (Convergence Rate)")
        
        # 2. Run DISTA
        z_nodes, converged_iter = DISTA(n, q, D, y, Q_curr, tau, lam_vec, max_iter=15000)
        
        if converged_iter < 15000:
            print(f"CONVERGED at iteration: {converged_iter}")
        else:
            print(f"Reached MAX ITERATIONS ({converged_iter}) without full convergence.")

        # 3. Consensus & Refinement
        z_final = np.mean(z_nodes, axis=0)
        x_est = z_final[:n]
        a_est = z_final[n:]
        
        # Refinement
        a_est_refined = np.copy(a_est)
        a_est_refined[np.abs(a_est_refined) < attack_threshold] = 0
        
        # Extract Indices
        est_targets = np.argsort(x_est)[-2:]
        est_attacks = np.where(a_est_refined != 0)[0]

        # Extract Values for Attacks
        est_attack_values = a_est_refined[est_attacks]
        
        print(f"   Estimated Targets: {est_targets} (True: {true_target_indices})")
        print(f"   Estimated Attacks: {est_attacks} (True: {true_attack_indices})")
        if len(est_attacks) > 0:
            print("   Estimated Attack Values:")
            for idx, val in zip(est_attacks, est_attack_values):
                print(f"      -> Sensor {idx}: {val:.4f}")
            else:
                print("      -> No attacks detected.")
        # ---------------------------------------------
        
        # 4. Plotting (Separate Figure for each)
        H, L, W = 10, 10, 100 
        room_grid = np.zeros((2, n))
        for k in range(n):
            room_grid[0, k] = W//2 + (k % L) * W
            room_grid[1, k] = W//2 + (k // L) * W

        plt.figure(i, figsize=(7, 7))
        plt.grid(True)
        plt.title(f"{topo_names[i]}\nConverged at iter: {converged_iter}")

        # True targets
        plt.plot(room_grid[0, true_target_indices], room_grid[1, true_target_indices], 's', markersize=9, 
                markeredgecolor=np.array([40, 208, 220])/255, 
                markerfacecolor=np.array([40, 208, 220])/255, label='True Targets')    
        
        # Estimated targets
        plt.plot(room_grid[0, est_targets], room_grid[1, est_targets], 'x', markersize=9, 
                markeredgecolor=np.array([255, 0, 0])/255, 
                markerfacecolor=np.array([255, 255, 255])/255, label='Est. Targets')

        # Sensors
        plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], s=50, c='pink', alpha=0.5, label='Sensors')
        
        # Attacked sensors
        if len(est_attacks) > 0:
            # Plot first one with label
            plt.plot(sensor_coords[est_attacks[0], 0], sensor_coords[est_attacks[0], 1], 'o', markersize=12, 
                    markeredgecolor=np.array([255, 0, 0])/255, markerfacecolor='none', label='Attacked')
            # Plot others
            for idx in est_attacks[1:]:
                plt.plot(sensor_coords[idx, 0], sensor_coords[idx, 1], 'o', markersize=12, 
                        markeredgecolor=np.array([255, 0, 0])/255, markerfacecolor='none')

        plt.axis([0, 1000, 0, 1000])
        plt.legend(loc='upper right', fontsize='small')
        plt.gca().set_aspect('equal', adjustable='box')
        
    plt.show()
    return

if __name__ == "__main__":
    # task_1()
    # task_2()
    # task_3()
    # task_4()
    # task_4_optional()
    task_5()