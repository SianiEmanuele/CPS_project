import numpy as np
import matplotlib.pyplot as plt

# List of sensors positions for localization plot
sensor_coords = np.array([
    [80,  750],[100,  345],[70, 170],[190, 930],[170, 30],[240, 320],[260, 360],[260, 460],[350, 700],[370, 410],
    [400, 950],[330, 640],[410, 650],[550, 20],[620, 750],[760, 760],[650,  10],[660, 230],[710, 195],[870, 650],
    [920, 950],[930, 610],[960, 190],[970, 260],[970, 980]
])


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