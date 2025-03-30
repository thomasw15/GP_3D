import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from parse_iscas85 import parse_iscas85_verilog

def run_iscas85_interconnect_sizing(benchmark_file, min_width_values=None):
    """
    Perform interconnect sizing optimization for ISCAS85 circuits.
    
    Implements sizing methodology using:
    1. Interconnect network extraction
    2. RC tree with π-models
    3. Elmore delay model
    4. Geometric programming optimization
    
    Args:
        benchmark_file: Path to ISCAS85 Verilog benchmark file
        min_width_values: Optional array of min width values to test
        
    Returns:
        tuple: (delay_values, width_values) for result analysis
    """
    # Parse the benchmark file to get circuit topology
    circuit_name = os.path.basename(benchmark_file).split('.')[0]
    print(f"Parsing {benchmark_file}...")
    
    # Extract circuit graph
    G, primary_inputs, primary_outputs = parse_iscas85_verilog(benchmark_file)
    print(f"Circuit {circuit_name}: {len(primary_inputs)} inputs, {len(primary_outputs)} outputs")
    print(f"Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    
    # Step 1: Create interconnect network (tree-structured)
    # We'll start from primary inputs and build trees to outputs
    interconnect_network = extract_interconnect_network(G, primary_inputs)
    unique_interconnects = list(interconnect_network.edges())
    
    print(f"Extracted {len(unique_interconnects)} unique interconnects")
    
    # Step 2: Create RC tree from interconnect network
    rc_tree, leaf_nodes = create_rc_tree(interconnect_network, primary_outputs)
    print(f"Created RC tree with {len(leaf_nodes)} leaf nodes")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of interconnects
    num_wires = len(unique_interconnects)
    
    # Parameters - these are constants in the optimization
    # Physical properties of interconnects
    alpha_param = np.random.uniform(0.5, 2.0, num_wires)
    alpha = cp.Parameter(num_wires, pos=True)
    alpha.value = alpha_param

    beta_param = np.random.uniform(0.5, 2.0, num_wires)
    beta = cp.Parameter(num_wires, pos=True)
    beta.value = beta_param

    gamma_param = np.random.uniform(0.5, 2.0, num_wires)
    gamma = cp.Parameter(num_wires, pos=True)
    gamma.value = gamma_param

    # Load capacitance for each interconnect
    C_L_param = np.random.uniform(0.5, 2.0, num_wires)
    C_L = cp.Parameter(num_wires, pos=True)
    C_L.value = C_L_param

    # Width constraints - for min_width experiment, use much larger max width range
    W_min_param = np.random.uniform(0.05, 0.1, num_wires)
    W_min = cp.Parameter(num_wires, pos=True)
    W_min.value = W_min_param

    # Much larger max width for more flexibility in min width experiments
    W_max_param = np.random.uniform(5.0, 5.5, num_wires)
    W_max = cp.Parameter(num_wires, pos=True)
    W_max.value = W_max_param

    # Length constraints
    L_min_param = np.random.uniform(2, 3, num_wires)
    L_min = cp.Parameter(num_wires, pos=True)
    L_min.value = L_min_param

    L_max_param = np.random.uniform(4, 5, num_wires)
    L_max = cp.Parameter(num_wires, pos=True)
    L_max.value = L_max_param

    # Volume constraint - greatly relaxed for min width experiments
    if circuit_name in ["c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]:
        Volume_max = 50.0 * num_wires  #  Larger constraint for bigger circuits
    else:
        Volume_max = 30.0  # More relaxed constraint for small circuits like c17

    # Optimization variables
    l = cp.Variable(num_wires, pos=True)  # Length
    w = cp.Variable(num_wires, pos=True)  # Width

    # Step 3: Compute R and C using π-model
    # R = alpha * l/w (resistance)
    R = cp.multiply(alpha, cp.multiply(l, cp.inv_pos(w)))
    # C = beta * l * w + gamma * l (capacitance)
    C = cp.multiply(beta, cp.multiply(l, w)) + cp.multiply(gamma, l)

    # Total capacitance for each interconnect (own + load)
    C_total = C + C_L
    
    # Step 4: Apply Elmore delay model
    print("Computing Elmore delays...")
    
    # Create mapping from edge to index
    edge_to_idx = {edge: i for i, edge in enumerate(unique_interconnects)}
    
    # Calculate path delays using Elmore delay model
    path_delays = []
    for leaf in leaf_nodes:
        # Find all input nodes (nodes with no incoming edges)
        input_nodes = [n for n in rc_tree.nodes() if rc_tree.in_degree(n) == 0 and n != leaf]
        
        # For each input, calculate delay to the leaf
        for source in input_nodes:
            try:
                # Find path from input to leaf
                path = list(nx.shortest_path(rc_tree, source, leaf))
                if len(path) <= 1:
                    continue  # Skip if path has only one node
                
                # Initialize delay as zero (scalar value, not CVXPY expression)
                path_delay = 0
                path_has_valid_delay = False
                
                # For each node in path
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    j = edge_to_idx.get(edge)
                    if j is None:
                        continue  # Skip if not an interconnect
                    
                    # Get all downstream nodes from this point in the path
                    downstream_nodes = set()
                    for k in range(i+1, len(path)):
                        downstream_nodes.add(path[k])
                    
                    # Add edges between downstream nodes
                    downstream_edges = []
                    for u, v in rc_tree.edges():
                        if u in downstream_nodes and v in downstream_nodes:
                            downstream_edges.append((u, v))
                    
                    # Get indices for downstream edges
                    downstream_indices = [edge_to_idx.get(e) for e in downstream_edges]
                    downstream_indices = [idx for idx in downstream_indices if idx is not None]
                    
                    # Compute downstream capacitance and add to delay
                    if downstream_indices:
                        downstream_C = sum([C_total[idx] for idx in downstream_indices])
                        contrib = R[j] * downstream_C
                        path_delay += contrib
                        path_has_valid_delay = True
                
                # Only add if we've computed a valid delay
                if path_has_valid_delay:
                    path_delays.append(path_delay)
                    
            except nx.NetworkXNoPath:
                continue  # No path between this input and leaf
    
    # If no valid delays, we can't proceed
    if not path_delays:
        print("No valid path delays found. Check interconnect extraction.")
        return
    
    # Maximum delay across all paths
    if len(path_delays) == 1:
        D_max = path_delays[0]
    else:
        D_max = cp.maximum(*path_delays)
    
    # Basic constraints
    constraints = [
        w >= W_min,  # Width constraints
        w <= W_max,
        l >= L_min,  # Length constraints
        l <= L_max,
        cp.sum(cp.multiply(l, cp.multiply(w, w))) <= Volume_max  # Volume constraint
    ]

    # Run experiment with different min width values
    if min_width_values is None:
        min_width_values = np.linspace(0.05, 1.5, 100)  # Adjusted upper bound to 1.5 as requested
    
    wire_delay = np.full(len(min_width_values), np.nan)  # Pre-allocate array
    
    # Identify 20% of the wires that are likely most critical (randomly for now)
    # In a real design, these would be critical paths identified by timing analysis
    np.random.seed(42)
    critical_wires = np.random.choice(num_wires, size=max(1, int(num_wires * 0.2)), replace=False)
    print(f"Selected {len(critical_wires)} critical wires out of {num_wires} total")
    
    for i, min_value in enumerate(min_width_values):
        print(f"Running with min width: {min_value:.3f} for critical wires only")
        # Apply min_value only to critical wires, not all wires
        W_min_original = W_min.value.copy()  # Save original values
        W_min.value[critical_wires] = min_value  # Set min width only for critical wires
        
        # Define and solve the problem
        prob = cp.Problem(cp.Minimize(D_max), constraints)
        
        solve_success = False
        # Try with multiple solvers
        solvers_to_try = [
            # SCS solver with parameters for large problems
            {"gp": True, "solver": cp.SCS, "eps": 1e-3, "max_iters": 5000, "verbose": True},
            # Try with more iterations if needed
            {"gp": True, "solver": cp.SCS, "eps": 1e-4, "max_iters": 10000},
            # Default solver (ECOS)
            {"gp": True},
            # ECOS with tighter tolerances
            {"gp": True, "abstol": 1e-8, "reltol": 1e-8},
            # Try MOSEK if available
            {"gp": True, "solver": cp.MOSEK}
        ]
        
        for solver_params in solvers_to_try:
            try:
                prob.solve(**solver_params)
                if prob.status in [cp.OPTIMAL, "optimal_inaccurate"] and prob.value is not None:
                    solve_success = True
                    break
            except Exception as e:
                print(f"  Solver failed with parameters {solver_params}: {str(e)}")
                continue
        
        if solve_success:
            # Calculate total volume
            Total_Volume = np.sum(l.value * w.value * w.value)
            print(f"  Optimal delay: {D_max.value:.5f}, Total volume: {Total_Volume:.2f}")
            
            # Check if volume constraint is tight
            if Total_Volume > Volume_max * 0.95:
                print("  Volume constraint is nearly tight")
            
            # Store the optimal delay
            wire_delay[i] = D_max.value
        else:
            print(f"  Problem could not be solved with any solver")
    
    # Create and save the plot with updated styling
    plt.figure(figsize=(20, 10))
    masked_delay = np.ma.masked_invalid(wire_delay)
    plt.plot(min_width_values, masked_delay, 'b-', linewidth=3)
    
    # Add labels with larger font size (all lowercase)
    plt.xlabel('minimal interconnect width', fontsize=24)
    plt.ylabel('optimal delay', fontsize=24)
    
    # Add grid on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set font size for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set title with larger font
    plt.title(f"Delay vs Minimal Width Constraint for {circuit_name}", fontsize=24)
    
    # Adjust layout
    plt.tight_layout()
    
    output_file = f"{circuit_name}_min_width.pdf"
    plt.savefig(output_file, format="pdf")
    print(f"Plot saved to {output_file}")
    
    # Save the experiment data for later use
    np.save(f"{circuit_name}_min_width_delays.npy", masked_delay.filled(np.nan))
    np.save(f"{circuit_name}_min_width_values.npy", min_width_values)
    print(f"Experiment data saved to {circuit_name}_min_width_delays.npy and {circuit_name}_min_width_values.npy")
    
    return masked_delay.filled(np.nan), min_width_values


def extract_interconnect_network(G, primary_inputs):
    """
    Extract a tree-structured interconnect network from the circuit graph
    starting from primary inputs.
    
    Returns a directed tree (forest) of interconnects.
    """
    interconnect_network = nx.DiGraph()
    
    # For each primary input, find all reachable nodes
    for input_node in primary_inputs:
        # Add the primary input
        interconnect_network.add_node(input_node, type='input')
        
        # Add all edges reachable from this input
        for edge in nx.edge_dfs(G, input_node):
            u, v = edge
            interconnect_network.add_edge(u, v)
    
    # Ensure all nodes from G are in the interconnect network
    for node in G.nodes():
        if node not in interconnect_network:
            interconnect_network.add_node(node)
    
    # Debug info
    print(f"Interconnect network has {interconnect_network.number_of_nodes()} nodes and {interconnect_network.number_of_edges()} edges")
    
    return interconnect_network


def create_rc_tree(interconnect_network, primary_outputs):
    """
    Create an RC tree from the interconnect network.
    
    Returns the RC tree and a list of leaf nodes.
    """
    rc_tree = interconnect_network.copy()
    
    # Identify leaf nodes (primary outputs and nodes with no outgoing edges)
    leaf_nodes = set()
    for node in rc_tree.nodes():
        if node in primary_outputs or rc_tree.out_degree(node) == 0:
            leaf_nodes.add(node)
    
    # Make sure we have at least one valid path to each leaf
    valid_leaves = set()
    for leaf in leaf_nodes:
        # Check if there's a path from any primary input to this leaf
        for node in rc_tree.nodes():
            if rc_tree.in_degree(node) == 0 and node != leaf:  # Input node
                try:
                    path = nx.shortest_path(rc_tree, node, leaf)
                    valid_leaves.add(leaf)
                    break
                except nx.NetworkXNoPath:
                    continue
    
    print(f"RC tree has {len(valid_leaves)} valid leaf nodes")
    
    return rc_tree, list(valid_leaves)


def run_multiple_benchmarks(benchmark_files=None):
    """
    Run the interconnect sizing experiment on multiple benchmarks.
    """
    if benchmark_files is None:
        benchmark_files = [
            "ISCAS85/c17.v",
            "ISCAS85/c432.v",
            "ISCAS85/c499.v",
            "ISCAS85/c880.v",
            "ISCAS85/c1355.v",
            "ISCAS85/c1908.v",
        ]
    
    results = {}
    for benchmark in benchmark_files:
        print(f"\nRunning on benchmark: {benchmark}")
        try:
            delay, width = run_iscas85_interconnect_sizing(benchmark)
            results[benchmark] = (delay, width)
        except Exception as e:
            print(f"Error processing {benchmark}: {str(e)}")
    
    return results


def run_iscas85_max_width_experiment(benchmark_file, max_width_values=None):
    """
    Run the interconnect sizing experiment for maximum width constraints.
    Similar to run_iscas85_interconnect_sizing but varies max width instead of min width.
    
    Args:
        benchmark_file: Path to the ISCAS85 Verilog benchmark file
        max_width_values: Optional array of max width values to test
    """
    # Parse the benchmark file to get circuit topology
    circuit_name = os.path.basename(benchmark_file).split('.')[0]
    print(f"Parsing {benchmark_file}...")
    
    # Extract circuit graph
    G, primary_inputs, primary_outputs = parse_iscas85_verilog(benchmark_file)
    print(f"Circuit {circuit_name}: {len(primary_inputs)} inputs, {len(primary_outputs)} outputs")
    print(f"Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    
    # Step 1: Create interconnect network (tree-structured)
    # We'll start from primary inputs and build trees to outputs
    interconnect_network = extract_interconnect_network(G, primary_inputs)
    unique_interconnects = list(interconnect_network.edges())
    
    print(f"Extracted {len(unique_interconnects)} unique interconnects")
    
    # Step 2: Create RC tree from interconnect network
    rc_tree, leaf_nodes = create_rc_tree(interconnect_network, primary_outputs)
    print(f"Created RC tree with {len(leaf_nodes)} leaf nodes")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Number of interconnects
    num_wires = len(unique_interconnects)
    
    # Parameters - these are constants in the optimization
    # Physical properties of interconnects
    alpha_param = np.random.uniform(0.5, 2.0, num_wires)
    alpha = cp.Parameter(num_wires, pos=True)
    alpha.value = alpha_param

    beta_param = np.random.uniform(0.5, 2.0, num_wires)
    beta = cp.Parameter(num_wires, pos=True)
    beta.value = beta_param

    gamma_param = np.random.uniform(0.5, 2.0, num_wires)
    gamma = cp.Parameter(num_wires, pos=True)
    gamma.value = gamma_param

    # Load capacitance for each interconnect
    C_L_param = np.random.uniform(0.5, 2.0, num_wires)
    C_L = cp.Parameter(num_wires, pos=True)
    C_L.value = C_L_param

    # Width constraints - for max_width experiment
    W_min_param = np.random.uniform(0.05, 0.1, num_wires)
    W_min = cp.Parameter(num_wires, pos=True)
    W_min.value = W_min_param

    # Original max width ranges for max width experiment
    W_max_param = np.random.uniform(0.3, 0.5, num_wires)
    W_max = cp.Parameter(num_wires, pos=True)
    W_max.value = W_max_param

    # Length constraints
    L_min_param = np.random.uniform(2, 3, num_wires)
    L_min = cp.Parameter(num_wires, pos=True)
    L_min.value = L_min_param

    L_max_param = np.random.uniform(4, 5, num_wires)
    L_max = cp.Parameter(num_wires, pos=True)
    L_max.value = L_max_param

    # Original volume constraints for max width experiment
    if circuit_name in ["c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]:
        Volume_max = 15.0 * num_wires  # Larger constraint for bigger circuits
    else:
        Volume_max = 10.0  # Relaxed constraint for small circuits like c17

    # Optimization variables
    l = cp.Variable(num_wires, pos=True)  # Length
    w = cp.Variable(num_wires, pos=True)  # Width

    # Step 3: Compute R and C using π-model
    # R = alpha * l/w (resistance)
    R = cp.multiply(alpha, cp.multiply(l, cp.inv_pos(w)))
    # C = beta * l * w + gamma * l (capacitance)
    C = cp.multiply(beta, cp.multiply(l, w)) + cp.multiply(gamma, l)

    # Total capacitance for each interconnect (own + load)
    C_total = C + C_L
    
    # Step 4: Apply Elmore delay model
    print("Computing Elmore delays...")
    
    # Create mapping from edge to index
    edge_to_idx = {edge: i for i, edge in enumerate(unique_interconnects)}
    
    # Calculate path delays using Elmore delay model
    path_delays = []
    for leaf in leaf_nodes:
        # Find all input nodes (nodes with no incoming edges)
        input_nodes = [n for n in rc_tree.nodes() if rc_tree.in_degree(n) == 0 and n != leaf]
        
        # For each input, calculate delay to the leaf
        for source in input_nodes:
            try:
                # Find path from input to leaf
                path = list(nx.shortest_path(rc_tree, source, leaf))
                if len(path) <= 1:
                    continue  # Skip if path has only one node
                
                # Initialize delay as zero (scalar value, not CVXPY expression)
                path_delay = 0
                path_has_valid_delay = False
                
                # For each node in path
                for i in range(len(path)-1):
                    edge = (path[i], path[i+1])
                    j = edge_to_idx.get(edge)
                    if j is None:
                        continue  # Skip if not an interconnect
                    
                    # Get all downstream nodes from this point in the path
                    downstream_nodes = set()
                    for k in range(i+1, len(path)):
                        downstream_nodes.add(path[k])
                    
                    # Add edges between downstream nodes
                    downstream_edges = []
                    for u, v in rc_tree.edges():
                        if u in downstream_nodes and v in downstream_nodes:
                            downstream_edges.append((u, v))
                    
                    # Get indices for downstream edges
                    downstream_indices = [edge_to_idx.get(e) for e in downstream_edges]
                    downstream_indices = [idx for idx in downstream_indices if idx is not None]
                    
                    # Compute downstream capacitance and add to delay
                    if downstream_indices:
                        downstream_C = sum([C_total[idx] for idx in downstream_indices])
                        contrib = R[j] * downstream_C
                        path_delay += contrib
                        path_has_valid_delay = True
                
                # Only add if we've computed a valid delay
                if path_has_valid_delay:
                    path_delays.append(path_delay)
                    
            except nx.NetworkXNoPath:
                continue  # No path between this input and leaf
    
    # If no valid delays, we can't proceed
    if not path_delays:
        print("No valid path delays found. Check interconnect extraction.")
        return
    
    # Maximum delay across all paths
    if len(path_delays) == 1:
        D_max = path_delays[0]
    else:
        D_max = cp.maximum(*path_delays)
    
    # Basic constraints
    constraints = [
        w >= W_min,  # Width constraints
        w <= W_max,
        l >= L_min,  # Length constraints
        l <= L_max,
        cp.sum(cp.multiply(l, cp.multiply(w, w))) <= Volume_max  # Volume constraint
    ]

    # Run experiment with different max width values
    if max_width_values is None:
        max_width_values = np.linspace(0.3, 3.0, 100)  # More steps for smoother plot
    
    wire_delay = np.full(len(max_width_values), np.nan)  # Pre-allocate array
    
    for i, max_value in enumerate(max_width_values):
        print(f"Running with max width: {max_value:.3f}")
        # Apply max_value to all wires
        W_max.value[:] = max_value
        
        # Define and solve the problem
        prob = cp.Problem(cp.Minimize(D_max), constraints)
        
        solve_success = False
        # Try with multiple solvers
        solvers_to_try = [
            # SCS solver with parameters for large problems
            {"gp": True, "solver": cp.SCS, "eps": 1e-3, "max_iters": 5000, "verbose": True},
            # Try with more iterations if needed
            {"gp": True, "solver": cp.SCS, "eps": 1e-4, "max_iters": 10000},
            # Default solver (ECOS)
            {"gp": True},
            # ECOS with tighter tolerances
            {"gp": True, "abstol": 1e-8, "reltol": 1e-8},
            # Try MOSEK if available
            {"gp": True, "solver": cp.MOSEK}
        ]
        
        for solver_params in solvers_to_try:
            try:
                prob.solve(**solver_params)
                if prob.status in [cp.OPTIMAL, "optimal_inaccurate"] and prob.value is not None:
                    solve_success = True
                    break
            except Exception as e:
                print(f"  Solver failed with parameters {solver_params}: {str(e)}")
                continue
        
        if solve_success:
            # Calculate total volume
            Total_Volume = np.sum(l.value * w.value * w.value)
            print(f"  Optimal delay: {D_max.value:.5f}, Total volume: {Total_Volume:.2f}")
            
            # Check if volume constraint is tight
            if Total_Volume > Volume_max * 0.95:
                print("  Volume constraint is nearly tight")
            
            # Store the optimal delay
            wire_delay[i] = D_max.value
        else:
            print(f"  Problem could not be solved with any solver")
    
    # Create and save the plot with updated styling
    plt.figure(figsize=(20, 10))
    masked_delay = np.ma.masked_invalid(wire_delay)
    plt.plot(max_width_values, masked_delay, 'b-', linewidth=3)
    
    # Add labels with larger font size (all lowercase)
    plt.xlabel('maximum interconnect width', fontsize=24)
    plt.ylabel('optimal delay', fontsize=24)
    
    # Add grid on y-axis only
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set font size for ticks
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Set title with larger font
    plt.title(f"Delay vs Maximum Width Constraint for {circuit_name}", fontsize=24)
    
    # Adjust layout
    plt.tight_layout()
    
    output_file = f"{circuit_name}_max_width.pdf"
    plt.savefig(output_file, format="pdf")
    print(f"Plot saved to {output_file}")
    
    # Save the experiment data for later use
    np.save(f"{circuit_name}_max_width_delays.npy", masked_delay.filled(np.nan))
    np.save(f"{circuit_name}_max_width_values.npy", max_width_values)
    print(f"Experiment data saved to {circuit_name}_max_width_delays.npy and {circuit_name}_max_width_values.npy")
    
    return masked_delay.filled(np.nan), max_width_values



# Update existing plotting code for consistent styling
def update_existing_plot_styles():
    # Update the plot styling in run_iscas85_interconnect_sizing
    # Figure size (20, 10)
    # Font size 20 (default)
    # Labels with fontsize 24
    # Grid on y-axis only with linestyle='--' and alpha=0.3
    # Tick font size 20
    # Tight layout
    plt.rcParams.update({'font.size': 20})

# Update the main function to include the length experiments
if __name__ == "__main__":
    import sys
    
    # Apply consistent styling to all plots
    plt.rcParams.update({'font.size': 20})
    
    if len(sys.argv) > 1:
        # Run specific benchmark
        benchmark_file = sys.argv[1]
        print(f"Running interconnect sizing experiments on {benchmark_file}")
        
        # Run all four experiments
        print("\n=== Running minimum width experiment with critical wires ===")
        min_delays, min_widths = run_iscas85_interconnect_sizing(benchmark_file)
        
        print("\n=== Running maximum width experiment ===")
        max_delays, max_widths = run_iscas85_max_width_experiment(benchmark_file)
    
    else:
        # Run on a small benchmark (c17) to test all four experiments
        print("Running on c17.v benchmark as a test")
        
        # Run all four experiments
        print("\n=== Running minimum width experiment with critical wires ===")
        min_delays, min_widths = run_iscas85_interconnect_sizing("ISCAS85/c17.v")
        
        print("\n=== Running maximum width experiment ===")
        max_delays, max_widths = run_iscas85_max_width_experiment("ISCAS85/c17.v")
    