import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os

# Set font sizes for plot readability
plt.rcParams.update({'font.size': 20})

def run_min_width_experiment():
    """
    Study impact of minimal interconnect width on optimal delay in the 6-interconnect circuit.
    
    Returns:
        tuple: (delay_values, width_values) for plotting and analysis
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters - these are constants in the optimization
    alpha_param = np.random.uniform(0.5, 2.0, 6)
    alpha = cp.Parameter(6, pos=True)
    alpha.value = alpha_param

    beta_param = np.random.uniform(0.5, 2.0, 6)
    beta = cp.Parameter(6, pos=True)
    beta.value = beta_param

    gamma_param = np.random.uniform(0.5, 2.0, 6)
    gamma = cp.Parameter(6, pos=True)
    gamma.value = gamma_param

    C_L_param = np.random.uniform(0.5, 2.0, 6)
    C_L = cp.Parameter(6, pos=True)
    C_L.value = C_L_param

    W_min_param = np.random.uniform(0.05, .1, 6)
    W_min = cp.Parameter(6, pos=True)
    W_min.value = W_min_param

    W_max_param = np.random.uniform(.3, .5, 6)
    W_max = cp.Parameter(6, pos=True)
    W_max.value = W_max_param

    L_min_param = np.random.uniform(2, 3, 6)
    L_min = cp.Parameter(6, pos=True)
    L_min.value = L_min_param

    L_max_param = np.random.uniform(4, 5, 6)
    L_max = cp.Parameter(6, pos=True)
    L_max.value = L_max_param

    Volume_max = 10000  # Volume constraint

    # Variables for optimization
    l = cp.Variable(6, pos=True)  # Length
    w = cp.Variable(6, pos=True)  # Width

    # Compute R and C using DGP-compatible operations
    # R = alpha * l/w (resistance)
    R = cp.multiply(alpha, cp.multiply(l, cp.inv_pos(w)))
    # C = beta * l * w + gamma * l (capacitance)
    C = cp.multiply(beta, cp.multiply(l, w)) + cp.multiply(gamma, l)

    # Total capacitance calculation
    C_tot = cp.vstack([
        C_L[0] + C[0] + C[1],
        C_L[1] + C[1] + C[2] + C[5],
        C_L[2] + C[2] + C[3] + C[4],
        C_L[3] + C[3],
        C_L[4] + C[4],
        C_L[5] + C[5]
    ])
    C_tot = cp.reshape(C_tot, (6,))

    # Delay calculations
    D1 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[2], (C_tot[2] + C_tot[3] + C_tot[4])) + cp.multiply(R[3], C_tot[3])
    D2 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[2], (C_tot[2] + C_tot[3] + C_tot[4])) + cp.multiply(R[4], C_tot[4])
    D3 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[5], C_tot[5])

    # Take maximum delay as objective
    Delays = cp.vstack([D1, D2, D3])
    D_max = cp.max(Delays)

    # Set basic constraints
    constraints = [
        w >= W_min, # Width constraints
        w <= W_max,
        l >= L_min, # Length constraints 
        l <= L_max,
        cp.sum(cp.multiply(l, cp.multiply(w, w))) <= Volume_max  # Volume constraint
    ]

    # Run experiment with different min width for wire 1
    wire_delay = []
    wire_min_values = [5 + 0.1 * i for i in range(0, 49)]
    
    for i in range(0, 49):
        print(f"Running with min width: {wire_min_values[i]}")
        W_min.value[1] = wire_min_values[i]
        # Set max width to 20 as in the notebook
        W_max.value[1] = 20.0
        
        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(D_max), constraints)
        
        solve_success = False
        # Try with multiple solvers
        solvers_to_try = [
            # Default solver (ECOS)
            {"gp": True},
            # ECOS with tighter tolerances
            {"gp": True, "abstol": 1e-8, "reltol": 1e-8},
            # SCS solver
            {"gp": True, "solver": cp.SCS, "eps": 1e-3, "max_iters": 5000},
            # Try with more iterations if needed
            {"gp": True, "solver": cp.SCS, "eps": 1e-4, "max_iters": 10000},
            # Try MOSEK if available
            {"gp": True, "solver": cp.MOSEK}
        ]
        
        for solver_params in solvers_to_try:
            try:
                prob.solve(**solver_params)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and prob.value is not None:
                    solve_success = True
                    break
            except Exception as e:
                continue
        
        if solve_success:
            # Calculate total volume
            Total_Volume = np.sum(l.value * w.value * w.value)
            print(f"  Optimal delay: {D_max.value}, Total volume: {Total_Volume:.2f}")
            
            # Store the optimal delay
            wire_delay.append(D_max.value)
        else:
            print(f"  Problem could not be solved with any solver")
            wire_delay.append(np.nan)  # Add NaN for failed problems

    # Convert to Python floats for plotting
    wire_delay_plot = [float(x) for x in wire_delay]

    # Create output directory if needed
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.plot(wire_min_values, wire_delay_plot, 'b-', linewidth=3)

    # Add labels with larger font size (all lowercase)
    plt.xlabel('minimal interconnect width', fontsize=24)
    plt.ylabel('optimal delay', fontsize=24)

    # Add grid on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Set font size for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'interconnect_min_width_styled.pdf'))
    plt.savefig(os.path.join(output_dir, 'interconnect_min_width_styled.png'))
    print(f"Minimum width plot saved to {output_dir}/interconnect_min_width_styled.pdf")
    
    # Save the experiment data for later use
    np.save(os.path.join(output_dir, 'interconnect_min_width_delays.npy'), wire_delay_plot)
    np.save(os.path.join(output_dir, 'interconnect_min_width_values.npy'), wire_min_values)
    print(f"Experiment data saved to {output_dir}/interconnect_min_width_delays.npy and {output_dir}/interconnect_min_width_values.npy")
    
    return wire_delay_plot, wire_min_values


def run_max_width_experiment():
    """
    Study impact of maximum interconnect width on optimal delay in the 6-interconnect circuit.
    
    Returns:
        tuple: (delay_values, width_values) for plotting and analysis
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters - these are constants in the optimization
    alpha_param = np.random.uniform(0.5, 2.0, 6)
    alpha = cp.Parameter(6, pos=True)
    alpha.value = alpha_param

    beta_param = np.random.uniform(0.5, 2.0, 6)
    beta = cp.Parameter(6, pos=True)
    beta.value = beta_param

    gamma_param = np.random.uniform(0.5, 2.0, 6)
    gamma = cp.Parameter(6, pos=True)
    gamma.value = gamma_param

    C_L_param = np.random.uniform(0.5, 2.0, 6)
    C_L = cp.Parameter(6, pos=True)
    C_L.value = C_L_param

    W_min_param = np.random.uniform(0.05, .1, 6)
    W_min = cp.Parameter(6, pos=True)
    W_min.value = W_min_param

    W_max_param = np.random.uniform(.3, .5, 6)
    W_max = cp.Parameter(6, pos=True)
    W_max.value = W_max_param

    L_min_param = np.random.uniform(2, 3, 6)
    L_min = cp.Parameter(6, pos=True)
    L_min.value = L_min_param

    L_max_param = np.random.uniform(4, 5, 6)
    L_max = cp.Parameter(6, pos=True)
    L_max.value = L_max_param

    Volume_max = 10000  # Use 10000 as in the notebook

    # Variables for optimization
    l = cp.Variable(6, pos=True)  # Length
    w = cp.Variable(6, pos=True)  # Width

    # Compute R and C using DGP-compatible operations
    R = cp.multiply(alpha, cp.multiply(l, cp.inv_pos(w)))
    C = cp.multiply(beta, cp.multiply(l, w)) + cp.multiply(gamma, l)

    # Total capacitance calculation - exact formula from notebook
    C_tot = cp.vstack([
        C_L[0] + C[0] + C[1],
        C_L[1] + C[1] + C[2] + C[5],
        C_L[2] + C[2] + C[3] + C[4],
        C_L[3] + C[3],
        C_L[4] + C[4],
        C_L[5] + C[5]
    ])
    C_tot = cp.reshape(C_tot, (6,))

    # Delay calculations - exactly as in notebook
    D1 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[2], (C_tot[2] + C_tot[3] + C_tot[4])) + cp.multiply(R[3], C_tot[3])
    D2 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[2], (C_tot[2] + C_tot[3] + C_tot[4])) + cp.multiply(R[4], C_tot[4])
    D3 = cp.multiply(R[0], cp.sum(C_tot)) + cp.multiply(R[1], cp.sum(C_tot[1:])) + cp.multiply(R[5], C_tot[5])

    # Use maximum of delays as objective
    Delays = cp.vstack([D1, D2, D3])
    D_max = cp.max(Delays)

    # Set basic constraints
    constraints = [
        w >= W_min, # Width constraints
        w <= W_max,
        l >= L_min, # Length constraints 
        l <= L_max,
        cp.sum(cp.multiply(l, cp.multiply(w, w))) <= Volume_max  # Volume constraint
    ]

    # Max width experiment ranges - exactly match notebook
    wire_delay = []
    wire_max_values = [.5 + 0.1 * i for i in range(1, 50)]
    
    for i in range(1, 50):
        print(f"Running with max width: {wire_max_values[i-1]}")
        W_max.value[1] = wire_max_values[i-1]
        # W_min.value[1] remains at its original value (no changes needed)
        
        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(D_max), constraints)
        
        solve_success = False
        # Try with multiple solvers
        solvers_to_try = [
            # Default solver (ECOS)
            {"gp": True},
            # ECOS with tighter tolerances
            {"gp": True, "abstol": 1e-8, "reltol": 1e-8},
            # SCS solver
            {"gp": True, "solver": cp.SCS, "eps": 1e-3, "max_iters": 5000},
            # Try with more iterations if needed
            {"gp": True, "solver": cp.SCS, "eps": 1e-4, "max_iters": 10000},
            # Try MOSEK if available
            {"gp": True, "solver": cp.MOSEK}
        ]
        
        for solver_params in solvers_to_try:
            try:
                prob.solve(**solver_params)
                if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and prob.value is not None:
                    solve_success = True
                    break
            except Exception as e:
                continue
        
        if solve_success:
            # Store the optimal delay
            wire_delay.append(D_max.value)
        else:
            print(f"  Problem could not be solved with any solver")
            wire_delay.append(np.nan)  # Add NaN for failed problems

    # Convert to Python floats for plotting
    wire_delay_plot = [float(x) for x in wire_delay]

    # Create output directory if needed
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.plot(wire_max_values, wire_delay_plot, 'b-', linewidth=3)

    # Add labels with larger font size (all lowercase)
    plt.xlabel('maximum interconnect width', fontsize=24)
    plt.ylabel('optimal delay', fontsize=24)

    # Add grid on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Set font size for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(output_dir, 'interconnect_max_width_styled.pdf'))
    plt.savefig(os.path.join(output_dir, 'interconnect_max_width_styled.png'))
    print(f"Maximum width plot saved to {output_dir}/interconnect_max_width_styled.pdf")
    
    # Save the experiment data for later use
    np.save(os.path.join(output_dir, 'interconnect_max_width_delays.npy'), wire_delay_plot)
    np.save(os.path.join(output_dir, 'interconnect_max_width_values.npy'), wire_max_values)
    print(f"Experiment data saved to {output_dir}/interconnect_max_width_delays.npy and {output_dir}/interconnect_max_width_values.npy")
    
    return wire_delay_plot, wire_max_values


if __name__ == "__main__":
    # Run both experiments
    print("Running experiment 1: Min Width Variation")
    min_width_delays, min_width_values = run_min_width_experiment()
    
    print("\nRunning experiment 2: Max Width Variation")
    max_width_delays, max_width_values = run_max_width_experiment() 