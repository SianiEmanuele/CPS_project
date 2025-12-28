# Magnetic Levitator Multi-Agent Control

**Authors:** Emanuele Giuseppe Siani, Laura Scigliano

## Project Structure
The project is orchestrated by a single MATLAB Live Script, **`main.mlx`**, which is logically divided into two sequential parts. Each part implements a specific observation strategy and executes a corresponding Simulink model:

1.  **Part 1: Neighborhood Observer** 
    * **Focus:** Followers estimate the leader's state using information shared between neighbors.
    * **Simulink Model:** Executes `model_neighborhood_observer.slx`.
2.  **Part 2: Local Observer** 
    * **Focus:** Followers estimate the state using only their own local measurements.
    * **Simulink Model:** Executes `model_local_observers.slx`.

## Overview
This simulation studies cooperative control for a multi-agent system (1 leader, 6 followers) based on **magnetic levitator dynamics**. It iterates through various configurations to evaluate stability (Hurwitz checks), consensus, and control energy.

## Simulation Scenarios
The script automatically tests the system under the following conditions defined in the code:
* **Topologies:** Iterates through 3 different network structures.
* **References:** Tests Constant, Sinusoidal, and Ramp signals.
* **LQR Weights:** Varies $Q$ and $R$ pairs (e.g., $Q=I, R=100$ vs $Q=100I, R=1$).
* **Noise:** Supports noise injection via the `noise_free` flag ($\sigma_{followers}=1$, $\sigma_{leader}=0.1$).

## Requirements
* MATLAB & Simulink
* **Core Files:** `main.mlx`, `model_neighborhood_observer.slx`, `model_local_observers.slx` 
* **Helpers:** `compute_topology_matrices.m`, `create_folder.m` 

## Usage
1.  Open **`main.mlx`**.
2.  Adjust the setup parameters if needed (e.g., set `automatically_save_plots = true` ).
2. Adjust the `noise_free` flag (default is true)
3.  Run the script. It will execute **Part 1** followed immediately by **Part 2**, logging results to text files (`*_simulation_log.txt`) and generating performance plots.