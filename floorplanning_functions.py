import cvxpy as cp
import numpy as np
import random
import time

# Define orientation types
HORIZONTAL = 0  # xy-plane (flat)
VERTICAL_YZ = 1  # yz-plane (standing along x-axis) 
VERTICAL_XZ = 2  # xz-plane (standing along y-axis)

def generate_module_data(num_modules=20, seed=None):
    """
    Generate random module data with variable orientations and physical properties.
    
    Parameters:
        num_modules: Number of modules to generate
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing module parameters (orientations, dimensions, thermal properties)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    print(f"Generating {num_modules} modules...")
    
    # Generate orientations randomly but ensure we have all types
    orientation_counts = {
        HORIZONTAL: max(num_modules // 2, 1),  # At least half horizontal
        VERTICAL_YZ: max(num_modules // 4, 1),  # Some vertical in YZ plane
        VERTICAL_XZ: max(num_modules // 4, 1)   # Some vertical in XZ plane
    }
    
    # Adjust if counts don't add up
    total = sum(orientation_counts.values())
    if total < num_modules:
        orientation_counts[HORIZONTAL] += (num_modules - total)
    elif total > num_modules:
        excess = total - num_modules
        if orientation_counts[VERTICAL_YZ] > excess:
            orientation_counts[VERTICAL_YZ] -= excess
        else:
            orientation_counts[VERTICAL_XZ] -= (excess - orientation_counts[VERTICAL_YZ])
            orientation_counts[VERTICAL_YZ] = 0
    
    # Create a list of orientations
    orientations = []
    for orientation, count in orientation_counts.items():
        orientations.extend([orientation] * count)
    
    # Shuffle the orientations
    random.shuffle(orientations)
    orientations = orientations[:num_modules]  # Ensure exactly num_modules
    
    # Generate random minimum dimensions
    width_min = np.random.uniform(0.5, 1.5, num_modules)
    height_min = np.random.uniform(0.5, 1.5, num_modules)
    depth_min = np.random.uniform(0.3, 0.8, num_modules)
    
    # Generate power consumption and thermal conductivity
    power_consumption = np.random.uniform(0.5, 2.0, num_modules)  # P_i in the paper (power consumption)
    thermal_conductivity = np.random.uniform(0.8, 1.2, num_modules)  # K_i in the paper (thermal conductivity)
    
    # Return as a dictionary
    return {
        'orientations': orientations,
        'width_min': width_min,
        'height_min': height_min,
        'depth_min': depth_min,
        'power_consumption': power_consumption,
        'thermal_conductivity': thermal_conductivity,
        'num_modules': num_modules
    }

def generate_arrangement_constraints(module_data):
    """
    Create spatial arrangement constraints between modules for 3D floorplanning.
    
    Generates constraint groups along each dimension (width, length, height) that define
    how modules must fit together. Includes both pair-wise adjacency constraints and
    multi-module group constraints.
    
    Parameters:
        module_data: Dictionary with module parameters
        
    Returns:
        Dictionary with constraint groups for each dimension
    """
    num_modules = module_data['num_modules']
    print("Generating arrangement constraints...")
    
    # Random module positions in 3D space (for visualization later only)
    num_dim = max(1, int(np.ceil(num_modules ** (1/3))))
    available_positions = [(r, c, l) for r in range(num_dim) for c in range(num_dim) for l in range(num_dim)]
    random.shuffle(available_positions)
    
    module_positions = []
    for i in range(min(num_modules, len(available_positions))):
        module_positions.append(available_positions[i])
    
    # Groups of modules that must fit within the total width W
    width_constraints = []
    
    # Groups of modules that must fit within the total length L
    length_constraints = []
    
    # Groups of modules that must fit within the total height H
    height_constraints = []
    
    # Determine average number of constraints per module - scales linearly with module count
    avg_constraints_per_dim = max(2, num_modules // 8)
    
    # First, create pairs of modules (adjacency relationships)
    for i in range(num_modules):
        # Create some width (x-direction) constraints
        num_width_constraints = random.randint(1, avg_constraints_per_dim)
        possible_pairs = [j for j in range(num_modules) if j != i]
        random.shuffle(possible_pairs)
        
        for j in possible_pairs[:num_width_constraints]:
            # Check if this pair already exists (in either order)
            if [i, j] not in width_constraints and [j, i] not in width_constraints:
                width_constraints.append([i, j])
        
        # Create some length (y-direction) constraints
        num_length_constraints = random.randint(1, avg_constraints_per_dim)
        possible_pairs = [j for j in range(num_modules) if j != i]
        random.shuffle(possible_pairs)
        
        for j in possible_pairs[:num_length_constraints]:
            # Check if this pair already exists (in either order)
            if [i, j] not in length_constraints and [j, i] not in length_constraints:
                length_constraints.append([i, j])
        
        # Create some height (z-direction) constraints
        num_height_constraints = random.randint(1, avg_constraints_per_dim)
        possible_pairs = [j for j in range(num_modules) if j != i]
        random.shuffle(possible_pairs)
        
        for j in possible_pairs[:num_height_constraints]:
            # Check if this pair already exists (in either order)
            if [i, j] not in height_constraints and [j, i] not in height_constraints:
                height_constraints.append([i, j])
    
    # Now create larger groups (3+ modules)
    # Number of larger groups scales linearly with module count
    num_larger_width_groups = max(1, num_modules // 5)
    num_larger_length_groups = max(1, num_modules // 5)
    num_larger_height_groups = max(1, num_modules // 5)
    
    # Maximum group size scales linearly with module count
    max_group_size = max(3, num_modules // 10)
    
    print(f"Creating {num_larger_width_groups} larger constraint groups with max size {max_group_size}")
    
    if num_modules > 10:
        # Create some larger width constraint groups
        for _ in range(num_larger_width_groups):
            group_size = random.randint(3, max_group_size)
            group = random.sample(range(num_modules), min(group_size, num_modules))
            if len(group) > 2:  # Only add if we actually got a larger group
                width_constraints.append(group)
        
        # Create some larger length constraint groups
        for _ in range(num_larger_length_groups):
            group_size = random.randint(3, max_group_size)
            group = random.sample(range(num_modules), min(group_size, num_modules))
            if len(group) > 2:  # Only add if we actually got a larger group
                length_constraints.append(group)
        
        # Create some larger height constraint groups
        for _ in range(num_larger_height_groups):
            group_size = random.randint(3, max_group_size)
            group = random.sample(range(num_modules), min(group_size, num_modules))
            if len(group) > 2:  # Only add if we actually got a larger group
                height_constraints.append(group)
    
    # Ensure at least one constraint for each dimension
    if num_modules > 0 and len(width_constraints) == 0:
        # Ensure at least one width constraint
        width_constraints.append([0, min(1, num_modules-1)])
    
    if num_modules > 0 and len(length_constraints) == 0:
        # Ensure at least one length constraint
        length_constraints.append([0, min(1, num_modules-1)])
    
    if num_modules > 0 and len(height_constraints) == 0:
        # Ensure at least one height constraint
        height_constraints.append([0, min(1, num_modules-1)])
    
    return {
        'width_constraints': width_constraints,
        'length_constraints': length_constraints,
        'height_constraints': height_constraints,
        'module_positions': module_positions  # Keep for potential use later
    }

def optimize_3d_floorplan(module_data, arrangement_constraints, alpha=0.6, print_details=True, solver_opts=None):
    """
    Perform 3D floorplanning optimization using geometric programming.
    
    Optimizes module dimensions and positions to minimize a weighted combination
    of total volume and thermal performance metrics, subject to arrangement
    constraints and minimum dimension requirements.
    
    Parameters:
        module_data: Dictionary with module parameters
        arrangement_constraints: Dictionary with spatial constraints
        alpha: Weight between volume (alpha) and thermal performance (1-alpha)
        print_details: Whether to print optimization details
        solver_opts: Additional solver options
        
    Returns:
        tuple: (optimized_module_data, dimensions) or (None, None) if failed
    """
    print("Optimizing 3D floorplan...")
    
    # Define default solver options if none provided
    if solver_opts is None:
        solver_opts = {}
    
    num_modules = module_data['num_modules']
    orientations = module_data['orientations']
    
    # Create variables for module dimensions
    x = cp.Variable(num_modules, pos=True)  # Width
    y = cp.Variable(num_modules, pos=True)  # Length
    z = cp.Variable(num_modules, pos=True)  # Height
    
    # Variables for total floorplan dimensions
    W = cp.Variable(pos=True)  # Total width
    L = cp.Variable(pos=True)  # Total length
    H = cp.Variable(pos=True)  # Total height
    
    # Define the temperature loss metric based on the paper
    # This calculates a thermal metric: Σ(P_i * K_i^(-1) * t_i * a_i^(-1))
    temperature_loss = 0
    
    # Create string representation for printing
    temp_loss_str = "Temperature Loss = "
    temp_terms = []
    
    for i in range(num_modules):
        P_i = module_data['power_consumption'][i]  # Power consumption of module i
        K_i = module_data['thermal_conductivity'][i]  # Thermal conductivity of module i
        
        # Calculate thickness (t_i) and area (a_i) based on orientation
        orientation = orientations[i]
        if orientation == HORIZONTAL:  # xy-plane
            # t_i = z_i, a_i = x_i * y_i
            thickness = z[i]
            area = x[i] * y[i]
            thickness_str = f"z[{i}]"
            area_str = f"x[{i}] * y[{i}]"
        elif orientation == VERTICAL_YZ:  # yz-plane
            # t_i = x_i, a_i = y_i * z_i
            thickness = x[i]
            area = y[i] * z[i]
            thickness_str = f"x[{i}]"
            area_str = f"y[{i}] * z[{i}]"
        else:  # VERTICAL_XZ
            # t_i = y_i, a_i = x_i * z_i
            thickness = y[i]
            area = x[i] * z[i]
            thickness_str = f"y[{i}]"
            area_str = f"x[{i}] * z[{i}]"
        
        # Formula from paper: P_i * K_i^(-1) * t_i * a_i^(-1)
        # This represents temperature loss where higher values mean worse thermal performance
        temperature_loss += P_i * (1/K_i) * thickness * (1/area)
        
        # Add to string representation
        temp_terms.append(f"P_{i}({P_i:.2f}) * (1/K_{i}({K_i:.2f})) * {thickness_str} * (1/{area_str})")
    
    temp_loss_str += " + ".join(temp_terms)
    
    # Objective function: trade-off between volume and temperature loss
    objective_fn = alpha * W * L * H + (1-alpha) * temperature_loss
    obj_fn_str = f"Objective Function = {alpha} * W * L * H + {1-alpha} * (Temperature Loss)"
    
    # Constraints
    constraints = []
    constraint_strs = []
    
    # Minimum dimension constraints based on orientation
    for i in range(num_modules):
        orientation = orientations[i]
        if orientation == HORIZONTAL:  # xy-plane
            constraints.append(x[i] >= module_data['width_min'][i])
            constraints.append(y[i] >= module_data['height_min'][i])
            constraints.append(z[i] >= module_data['depth_min'][i])
            
            constraint_strs.append(f"x[{i}] >= {module_data['width_min'][i]:.2f}")
            constraint_strs.append(f"y[{i}] >= {module_data['height_min'][i]:.2f}")
            constraint_strs.append(f"z[{i}] >= {module_data['depth_min'][i]:.2f}")
        elif orientation == VERTICAL_YZ:  # yz-plane (standing along x)
            constraints.append(x[i] >= module_data['depth_min'][i])
            constraints.append(y[i] >= module_data['width_min'][i])
            constraints.append(z[i] >= module_data['height_min'][i])
            
            constraint_strs.append(f"x[{i}] >= {module_data['depth_min'][i]:.2f}")
            constraint_strs.append(f"y[{i}] >= {module_data['width_min'][i]:.2f}")
            constraint_strs.append(f"z[{i}] >= {module_data['height_min'][i]:.2f}")
        elif orientation == VERTICAL_XZ:  # xz-plane (standing along y)
            constraints.append(x[i] >= module_data['width_min'][i])
            constraints.append(y[i] >= module_data['depth_min'][i])
            constraints.append(z[i] >= module_data['height_min'][i])
            
            constraint_strs.append(f"x[{i}] >= {module_data['width_min'][i]:.2f}")
            constraint_strs.append(f"y[{i}] >= {module_data['depth_min'][i]:.2f}")
            constraint_strs.append(f"z[{i}] >= {module_data['height_min'][i]:.2f}")
    
    # Apply group constraints for width, length, and height
    
    # Width constraints
    for group in arrangement_constraints['width_constraints']:
        # Sum of widths of modules in this group must be <= W
        if len(group) > 0:  # Make sure the group is not empty
            constraints.append(cp.sum([x[i] for i in group]) <= W)
            if len(group) == 1:
                constraint_strs.append(f"x[{group[0]}] <= W")
            elif len(group) == 2:
                constraint_strs.append(f"x[{group[0]}] + x[{group[1]}] <= W")
            else:
                constraint_strs.append(f"sum([x[i] for i in {group}]) <= W")
    
    # Length constraints
    for group in arrangement_constraints['length_constraints']:
        # Sum of lengths of modules in this group must be <= L
        if len(group) > 0:
            constraints.append(cp.sum([y[i] for i in group]) <= L)
            if len(group) == 1:
                constraint_strs.append(f"y[{group[0]}] <= L")
            elif len(group) == 2:
                constraint_strs.append(f"y[{group[0]}] + y[{group[1]}] <= L")
            else:
                constraint_strs.append(f"sum([y[i] for i in {group}]) <= L")
    
    # Height constraints
    for group in arrangement_constraints['height_constraints']:
        # Sum of heights of modules in this group must be <= H
        if len(group) > 0:
            constraints.append(cp.sum([z[i] for i in group]) <= H)
            if len(group) == 1:
                constraint_strs.append(f"z[{group[0]}] <= H")
            elif len(group) == 2:
                constraint_strs.append(f"z[{group[0]}] + z[{group[1]}] <= H")
            else:
                constraint_strs.append(f"sum([z[i] for i in {group}]) <= H")
    
    # Print the optimization problem details if requested
    if print_details:
        print("\n" + "="*80)
        print("OPTIMIZATION PROBLEM DETAILS")
        print("="*80)
        print("\nVARIABLES:")
        print(f"  Module dimensions: x[0..{num_modules-1}], y[0..{num_modules-1}], z[0..{num_modules-1}]")
        print(f"  Total dimensions: W, L, H")
        print(f"\nOBJECTIVE FUNCTION:")
        print(f"  {obj_fn_str}")
        print(f"\nTEMPERATURE LOSS BREAKDOWN:")
        # Print temperature loss terms more readably
        for i, term in enumerate(temp_terms):
            print(f"  Term {i}: {term}")
        print(f"\nCONSTRAINTS ({len(constraints)}):")
        for i, constraint_str in enumerate(constraint_strs):
            print(f"  {i+1}: {constraint_str}")
        print("\n" + "="*80)
    
    # Define the problem
    problem = cp.Problem(cp.Minimize(objective_fn), constraints)
    
    try:
        # Solve using geometric programming
        try:
            # First try using SCS, which is more permissive with solver options
            problem.solve(gp=True, solver='SCS', verbose=False)
        except Exception as e1:
            print(f"SCS solver failed: {e1}")
            try:
                # Fall back to ECOS if SCS fails
                problem.solve(gp=True, solver='ECOS', verbose=False)
            except Exception as e2:
                print(f"ECOS solver failed: {e2}")
                # As a last resort, use Clarabel with minimal options
                problem.solve(gp=True, solver='CLARABEL', verbose=False, max_iter=10000)
        
        print(f"\nOptimization complete. Status: {problem.status}")
        print(f"Objective value: {problem.value}")
        print(f"Total volume: {W.value * L.value * H.value}")
        
        # Print optimal dimensions if requested
        if print_details:
            print("\n" + "="*80)
            print("OPTIMAL SOLUTION")
            print("="*80)
            print(f"Total dimensions: W={W.value:.4f}, L={L.value:.4f}, H={H.value:.4f}")
            print("\nModule dimensions:")
            for i in range(num_modules):
                orientation_text = "Horizontal" if orientations[i] == HORIZONTAL else \
                                  ("Vertical YZ" if orientations[i] == VERTICAL_YZ else "Vertical XZ")
                print(f"  Module {i} ({orientation_text}): x={x[i].value:.4f}, y={y[i].value:.4f}, z={z[i].value:.4f}")
            print("="*80)
        
        # Calculate module positions for visualization (not part of the optimization)
        # We'll need to extract adjacency information from the constraints for visualization
        pos_x = np.zeros(num_modules)
        pos_y = np.zeros(num_modules)
        pos_z = np.zeros(num_modules)
        
        # Create a dictionary to track which modules have been placed
        placed = set()
        
        # Extract pairs from width constraints (adjacency in x-direction)
        x_adjacencies = [group for group in arrangement_constraints['width_constraints'] if len(group) == 2]
        y_adjacencies = [group for group in arrangement_constraints['length_constraints'] if len(group) == 2]
        z_adjacencies = [group for group in arrangement_constraints['height_constraints'] if len(group) == 2]
        
        # Start with a random module
        current_module = random.randint(0, num_modules - 1)
        placed.add(current_module)
        
        # Keep placing modules until all are placed
        while len(placed) < num_modules:
            # Try to place modules adjacent in x-direction
            for idx, pair in enumerate(x_adjacencies):
                i, j = pair
                if i in placed and j not in placed:
                    # Place j to the right of i
                    pos_x[j] = pos_x[i] + x[i].value
                    pos_y[j] = pos_y[i]  # Same y-coordinate
                    pos_z[j] = pos_z[i]  # Same z-coordinate
                    placed.add(j)
                    x_adjacencies.pop(idx)
                    break
                elif j in placed and i not in placed:
                    # Place i to the right of j
                    pos_x[i] = pos_x[j] + x[j].value
                    pos_y[i] = pos_y[j]  # Same y-coordinate
                    pos_z[i] = pos_z[j]  # Same z-coordinate
                    placed.add(i)
                    x_adjacencies.pop(idx)
                    break
            
            # Try to place modules adjacent in y-direction
            for idx, pair in enumerate(y_adjacencies):
                i, j = pair
                if i in placed and j not in placed:
                    # Place j above i
                    pos_x[j] = pos_x[i]  # Same x-coordinate
                    pos_y[j] = pos_y[i] + y[i].value
                    pos_z[j] = pos_z[i]  # Same z-coordinate
                    placed.add(j)
                    y_adjacencies.pop(idx)
                    break
                elif j in placed and i not in placed:
                    # Place i above j
                    pos_x[i] = pos_x[j]  # Same x-coordinate
                    pos_y[i] = pos_y[j] + y[j].value
                    pos_z[i] = pos_z[j]  # Same z-coordinate
                    placed.add(i)
                    y_adjacencies.pop(idx)
                    break
            
            # Try to place modules adjacent in z-direction
            for idx, pair in enumerate(z_adjacencies):
                i, j = pair
                if i in placed and j not in placed:
                    # Place j on top of i
                    pos_x[j] = pos_x[i]  # Same x-coordinate
                    pos_y[j] = pos_y[i]  # Same y-coordinate
                    pos_z[j] = pos_z[i] + z[i].value
                    placed.add(j)
                    z_adjacencies.pop(idx)
                    break
                elif j in placed and i not in placed:
                    # Place i on top of j
                    pos_x[i] = pos_x[j]  # Same x-coordinate
                    pos_y[i] = pos_y[j]  # Same y-coordinate
                    pos_z[i] = pos_z[j] + z[j].value
                    placed.add(i)
                    z_adjacencies.pop(idx)
                    break
            
            # If no adjacency relationship can be used to place a module,
            # just place a random unplaced module somewhere
            if len(placed) < num_modules and all((i in placed or j in placed) for pair in x_adjacencies + y_adjacencies + z_adjacencies for i, j in [pair]):
                unplaced = list(set(range(num_modules)) - placed)
                next_module = random.choice(unplaced)
                # Place it at a random position within bounds
                pos_x[next_module] = random.uniform(0, W.value - x[next_module].value)
                pos_y[next_module] = random.uniform(0, L.value - y[next_module].value)
                pos_z[next_module] = random.uniform(0, H.value - z[next_module].value)
                placed.add(next_module)
        
        # Prepare optimized module data for visualization
        optimized_module_data = []
        for i in range(num_modules):
            orientation = orientations[i]
            
            optimized_module_data.append({
                'id': i,
                'pos_x': pos_x[i],
                'pos_y': pos_y[i],
                'pos_z': pos_z[i],
                'width': x[i].value,
                'height': y[i].value,
                'depth': z[i].value,
                'orientation': orientation
            })
        
        return optimized_module_data, (W.value, L.value, H.value)
        
    except Exception as e:
        print(f"Error solving optimization problem: {e}")
        return None, None

def run_true_3d_floorplanning(num_modules=20, alpha=0.6, seed=None, print_details=True, solver_opts=None):
    """
    Execute complete 3D floorplanning process.
    
    Parameters:
        num_modules: Number of modules to generate
        alpha: Weight between volume (alpha) and thermal performance (1-alpha)
        seed: Random seed for reproducibility
        print_details: Whether to print optimization details
        solver_opts: Additional solver options
        
    Returns:
        tuple: (optimized_data, dimensions) or (None, None) if optimization failed
    """
    
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate module data
    print(f"Generating {num_modules} modules...")
    module_data = generate_module_data(num_modules, seed)
    
    # Generate arrangement constraints
    print(f"Generating arrangement constraints...")
    arrangement_constraints = generate_arrangement_constraints(module_data)
    
    # Print the constraints
    print("\nWidth constraints (modules arranged along x-axis):")
    for group in arrangement_constraints['width_constraints']:
        if len(group) == 1:
            print(f"Module {group[0]} width <= W")
        elif len(group) == 2:
            print(f"Modules {group[0]} and {group[1]} are adjacent in x-direction: x[{group[0]}] + x[{group[1]}] <= W")
        else:
            print(f"Sum of widths of modules {group} <= W")
    
    # Display length constraints
    print("\nLength constraints (modules arranged along y-axis):")
    for group in arrangement_constraints['length_constraints']:
        if len(group) == 1:
            print(f"Module {group[0]} length <= L")
        elif len(group) == 2:
            print(f"Modules {group[0]} and {group[1]} are adjacent in y-direction: y[{group[0]}] + y[{group[1]}] <= L")
        else:
            print(f"Sum of lengths of modules {group} <= L")
    
    # Display height constraints
    print("\nHeight constraints (modules arranged along z-axis):")
    for group in arrangement_constraints['height_constraints']:
        if len(group) == 1:
            print(f"Module {group[0]} height <= H")
        elif len(group) == 2:
            print(f"Modules {group[0]} and {group[1]} are adjacent in z-direction: z[{group[0]}] + z[{group[1]}] <= H")
        else:
            print(f"Sum of heights of modules {group} <= H")
    
    # Optimize the floorplan
    optimized_data, dimensions = optimize_3d_floorplan(
        module_data, 
        arrangement_constraints, 
        alpha,
        print_details=print_details,
        solver_opts=solver_opts
    )
    
    if optimized_data:
        # Print dimensions
        total_width, total_length, total_height = dimensions
        print(f"\nOptimization successful. Final dimensions: W×L×H: {total_width:.2f}×{total_length:.2f}×{total_height:.2f}")
        return optimized_data, dimensions
    else:
        print("Optimization failed.")
        return None, None

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run true 3D floorplanning with different module orientations')
    parser.add_argument('--num_modules', type=int, default=20, help='Number of modules (default: 20)')
    parser.add_argument('--alpha', type=float, default=0.6, help='Weight between volume and performance, 0-1 (default: 0.6)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')
    parser.add_argument('--no_details', action='store_true', help='Disable printing detailed optimization information')
    
    # Add solver options
    parser.add_argument('--max_iters', type=int, default=None, help='Maximum solver iterations (default: solver default)')
    parser.add_argument('--abstol', type=float, default=None, help='Absolute tolerance for solver')
    parser.add_argument('--reltol', type=float, default=None, help='Relative tolerance for solver')
    parser.add_argument('--feastol', type=float, default=None, help='Feasibility tolerance for solver')
    
    args = parser.parse_args()
    
    # Build solver options dictionary from args
    solver_opts = {}
    if args.max_iters is not None:
        solver_opts['max_iters'] = args.max_iters
    if args.abstol is not None:
        solver_opts['abstol'] = args.abstol
    if args.reltol is not None:
        solver_opts['reltol'] = args.reltol
    if args.feastol is not None:
        solver_opts['feastol'] = args.feastol
    
    # Run the floorplanning process
    optimized_data, dimensions = run_true_3d_floorplanning(
        num_modules=args.num_modules,
        alpha=args.alpha,
        seed=args.seed,
        print_details=not args.no_details,
        solver_opts=solver_opts
    )
    
    if optimized_data:
        print(f"True 3D floorplanning with {args.num_modules} modules complete!")
    else:
        print(f"Floorplanning failed.")
