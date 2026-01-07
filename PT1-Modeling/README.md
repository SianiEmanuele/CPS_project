# Modeling and control of cyber-physical systems - Part 1

**Authors:** Emanuele Giuseppe Siani, Laura Scigliano

## Project Structure
The project is implemented in **Python**. The workflow is centered around a main execution script and a library of utility functions:

1.  **Main Execution (`main.py`)**
    * **Focus:** Orchestrates the simulation flow and executes **Tasks 1 to 5** (including the optional **Task 4**).
    * **Function:** Loads data, configures parameters, calls the specific algorithms, and manages the result visualization.

2.  **Utilities (`src/utils/utilities.py`)**
    * **Focus:** Contains the core logic, algorithms, and helper tools required by the main script.
    * **Components:**
        * **Algorithms:** Implementation of optimization algorithms such as **ISTA** (Iterative Shrinkage-Thresholding Algorithm) and **DISTA** (Distributed ISTA).
        * **Runners:** Functions to execute multiple simulation runs or handle specific scenarios like the **Localization Problem**.
        * **Helpers:** Utility functions for convergence checks, **consensus** verification, and error tracking.
        * **Plotting:** Specialized functions to generate graphs and visualize the estimation results.

## Requirements
* **Python 3.x**
* **Dependencies:** Listed in `requirements.txt`. Common libraries include:
    * `numpy`
    * `scipy`
    * `matplotlib`

## Usage
1.  Navigate to the `PT1-Modeling` directory.
2.  Install the dependencies (if not already installed):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main script:
    ```bash
    python PT1-Modeling/src/main.py
    ```
4.  The script will execute the configured Tasks (1 through 5) and generate the corresponding plots and console outputs for analysis.