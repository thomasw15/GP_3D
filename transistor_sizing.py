import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def run_transistor_sizing_experiment():
    """
    This experiment optimizes transistor sizing to minimize delay under
    various volume constraints. It models a circuit with constraints on
    transistor sizes, capacitance, and power.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define parameters for the transistor circuit
    # Set up device parameters
    num_devices = 9  # Total number of transistors
    
    # Create variables for transistor sizes (vector of scaling factors)
    x = cp.Variable(num_devices, pos=True)
    
    # Device parameters (capacitance, resistance, etc.)
    V_dd = 1.0
    I_leak = np.random.uniform(0.1, 0.3, num_devices)
    R_0 = np.random.uniform(1.0, 3.0, num_devices)
    C_g = np.random.uniform(0.5, 1.5, num_devices) 
    C_d = np.random.uniform(0.5, 1.5, num_devices)
    
    # Activity factors (switching frequency)
    Freq = np.random.uniform(0.2, 0.5, num_devices)
    Freq_PI = np.random.uniform(0.2, 0.5, 2)
    
    # Primary input capacitances
    C_in = cp.Parameter(9, pos=True)
    C_in.value = np.random.uniform(0.5, 1.5, 9)
    
    C_in_PO = cp.Parameter(2, pos=True)
    C_in_PO.value = np.random.uniform(0.5, 1.5, 2)
    
    # Volume calculations
    vol_per_x = np.random.uniform(1.0, 3.0, num_devices)
    Total_Volume = cp.sum(cp.multiply(vol_per_x, x))
    
    # Calculate resistance (inverse scaling with transistor size)
    R = cp.multiply(R_0, cp.inv_pos(x))
    
    # Calculate intrinsic capacitance (scales with transistor size)
    C_int = cp.multiply(C_g, x) + cp.multiply(C_d, x)
    
    # Primary input capacitances
    C_L_PI = cp.vstack([C_in[0] + C_in[1], C_in[2] + C_in[3]])
    C_L_PI = cp.reshape(C_L_PI, (2,))
    
    # Load capacitances of circuit blocks
    C_L_CB = cp.vstack([
            C_in[4],
            C_in[4] + C_in[5] + C_in[6],
            C_in[4] + C_in[5] + C_in[6],
            C_in[4] + C_in[5] + C_in[6],
            C_in[7],
            C_in[8],
            C_in[8],
            C_in_PO[0],
            C_in_PO[1]
    ])
    C_L_CB = cp.reshape(C_L_CB, (9,))
    
    # Total capacitance per transistor
    C_total = C_L_CB + C_int
    
    # Power calculation
    Power = (
       (cp.sum(cp.multiply(Freq, C_total)) + cp.sum(cp.multiply(Freq_PI, C_L_PI))) * V_dd**2
        + V_dd * cp.sum(cp.multiply(x, I_leak))
    )
    
    # Compute the delay for each transistor using the RC delay model
    Delay = cp.multiply(0.69, cp.multiply(R, C_total))
    
    # Calculate delays along different paths in the circuit
    Delay_path = cp.vstack([
            Delay[0] + Delay[4] + Delay[7],
            Delay[1] + Delay[4] + Delay[7],
            Delay[1] + Delay[5] + Delay[8],
            Delay[1] + Delay[6] + Delay[8],
            Delay[2] + Delay[4] + Delay[7],
            Delay[2] + Delay[5] + Delay[8],
            Delay[2] + Delay[6] + Delay[8],
            Delay[3] + Delay[4] + Delay[7],
            Delay[3] + Delay[5] + Delay[8],
            Delay[3] + Delay[6] + Delay[8],
    ])
    
    # Worst-case delay is the maximum delay across all paths
    D_max = cp.max(Delay_path)
    
    # Track optimal delay for different volume constraints
    optimal_delay = []
    optimal_volume = []
    
    # Vary the maximum volume constraint
    for i in range(1, 50):
        print(f"Iteration {i}")
        Volume_max = 50 + 2*i
        
        # Set constraints
        constraints = [
            #Power <= 1000.0,  # Optional power limit
            Total_Volume <= Volume_max,  # Volume constraint
            x >= 1  # Minimum transistor size
        ]
        
        # Define and solve the optimization problem
        prob = cp.Problem(cp.Minimize(D_max), constraints)
        prob.solve(solver="SCS", gp=True)  # Use geometric programming
        
        # Record results
        optimal_delay.append(D_max.value)
        optimal_volume.append(Total_Volume.value)
    
    # Convert to Python float for plotting
    optimal_delay = [float(d) for d in optimal_delay]
    volumes = [50 + 2*i for i in range(1, 50)]
    
    # Create the plot
    plt.figure(figsize=(20, 10))
    
    # Set larger default font sizes
    plt.rcParams.update({'font.size': 20})
    
    # Plot the data
    plt.plot(volumes, optimal_delay, 'b-', linewidth=3)
    
    # Labels - all lowercase
    plt.xlabel("volume constraint", fontsize=24)
    plt.ylabel("delay", fontsize=24)
    
    # Add grid matching other plots
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set font size for tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("transistor_sizing.pdf", format="pdf")
    plt.savefig("transistor_sizing.png", format="png")
    
    # Return data instead of showing the plot
    # plt.show()  # Commented out to prevent blocking when called from other scripts
    
    return optimal_delay, volumes

if __name__ == "__main__":
    run_transistor_sizing_experiment() 