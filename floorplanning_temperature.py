import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import os
import argparse
import multiprocessing
import concurrent.futures
import pickle
import json

# Import functions from existing implementation
from floorplanning_functions import (
    generate_module_data,
    generate_arrangement_constraints,
    optimize_3d_floorplan,
    HORIZONTAL, VERTICAL_YZ, VERTICAL_XZ
)

def calculate_pre_optimization_metrics(module_data):
    """
    Calculate temperature metrics for modules using minimum dimensions.
    
    Parameters:
        module_data: Dictionary with module parameters
    
    Returns:
        List of temperature metrics for each module
    """
    num_modules = module_data['num_modules']
    temperatures = []
    
    for i in range(num_modules):
        # Extract module dimensions based on orientation
        orientation = module_data['orientations'][i]
        width = module_data['width_min'][i]
        height = module_data['height_min'][i]
        depth = module_data['depth_min'][i]
        
        # Calculate temperature metric
        P_i = module_data['power_consumption'][i]
        K_i = module_data['thermal_conductivity'][i]
        
        # Calculate thickness and area based on orientation
        if orientation == HORIZONTAL:
            thickness = depth
            area = width * height
        elif orientation == VERTICAL_YZ:
            thickness = width
            area = height * depth
        else:  # VERTICAL_XZ
            thickness = height
            area = width * depth
        
        # Temperature metric: P_i * K_i^(-1) * t_i * a_i^(-1)
        temp_metric = P_i * (1/K_i) * thickness * (1/area)
        temperatures.append(temp_metric)
    
    return temperatures

def calculate_post_optimization_metrics(module_data, optimized_data):
    """
    Calculate temperature metrics for modules using optimized dimensions.
    
    Parameters:
        module_data: Dictionary with original module parameters
        optimized_data: List of optimized module data
    
    Returns:
        List of temperature metrics for each module
    """
    num_modules = module_data['num_modules']
    temperatures = []
    
    for i in range(num_modules):
        # Extract optimized module dimensions
        module = optimized_data[i]
        width = module['width']
        height = module['height']
        depth = module['depth']
        orientation = module_data['orientations'][i]  # Original orientation
        
        # Calculate temperature metric
        P_i = module_data['power_consumption'][i]
        K_i = module_data['thermal_conductivity'][i]
        
        # Calculate thickness and area based on orientation
        if orientation == HORIZONTAL:
            thickness = depth
            area = width * height
        elif orientation == VERTICAL_YZ:
            thickness = width
            area = height * depth
        else:  # VERTICAL_XZ
            thickness = height
            area = width * depth
        
        # Temperature metric: P_i * K_i^(-1) * t_i * a_i^(-1)
        temp_metric = P_i * (1/K_i) * thickness * (1/area)
        temperatures.append(temp_metric)
    
    return temperatures

def run_experiment(seed, num_modules=150, alpha=0.6, verbose=False):
    """
    Run single 3D floorplanning optimization experiment.
    
    Parameters:
        seed: Random seed for reproducibility
        num_modules: Number of modules to generate
        alpha: Weight between volume and temperature loss
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with pre/post-optimization metrics or None if failed
    """
    try:
        # Set random seed
        np.random.seed(seed)
        random.seed(seed)
        
        if verbose:
            print(f"Starting experiment with seed {seed}")
        
        # Generate module data
        module_data = generate_module_data(num_modules, seed)
        
        # Generate arrangement constraints
        arrangement_constraints = generate_arrangement_constraints(module_data)
        
        # Calculate pre-optimization metrics
        pre_temperatures = calculate_pre_optimization_metrics(module_data)
        
        # Solver options
        solver_opts = {
            'verbose': False,
            'max_iter': 10000
        }
        
        # Optimize the floorplan
        optimized_data, dimensions = optimize_3d_floorplan(
            module_data, 
            arrangement_constraints, 
            alpha,
            print_details=False,
            solver_opts=solver_opts
        )
        
        # If optimization failed, return None
        if optimized_data is None:
            if verbose:
                print(f"Experiment with seed {seed} failed.")
            return None
        
        # Calculate post-optimization metrics
        post_temperatures = calculate_post_optimization_metrics(module_data, optimized_data)
        
        # Calculate total metrics
        total_pre_temperature = sum(pre_temperatures)
        total_post_temperature = sum(post_temperatures)
        
        # Calculate percentage improvement
        temperature_improvement = (1 - total_post_temperature/total_pre_temperature) * 100
        
        if verbose:
            print(f"Experiment with seed {seed} completed.")
            print(f"  Temperature: {total_pre_temperature:.2f} -> {total_post_temperature:.2f} ({temperature_improvement:.2f}% improvement)")
        
        return {
            'seed': seed,
            'module_data': module_data,
            'optimized_data': optimized_data,
            'pre_temperatures': pre_temperatures,
            'post_temperatures': post_temperatures,
            'total_pre_temperature': total_pre_temperature,
            'total_post_temperature': total_post_temperature,
            'temperature_improvement': temperature_improvement
        }
    except Exception as e:
        print(f"Error in experiment with seed {seed}: {str(e)}")
        return None

def run_multiple_experiments(num_experiments=1, num_modules=150, alpha=0.6, max_workers=None, verbose=False):
    """
    Execute multiple floorplanning experiments in parallel.
    
    Parameters:
        num_experiments: Number of experiments to run
        num_modules: Number of modules per experiment
        alpha: Weight between volume and temperature loss
        max_workers: Maximum number of parallel workers (None = auto)
        verbose: Whether to print detailed information
    
    Returns:
        List of dictionaries with experiment results
    """
    print(f"\nRunning {num_experiments} optimization experiments with {num_modules} modules...")
    
    # Generate seeds
    seeds = list(range(1, num_experiments + 1))
    
    # If max_workers is not specified, use number of CPU cores
    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    
    results = []
    start_time = time.time()
    
    # Run experiments in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create a list of futures
        futures = {
            executor.submit(run_experiment, seed, num_modules, alpha, verbose): seed
            for seed in seeds
        }
        
        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc="Running experiments"
        ):
            seed = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as exc:
                print(f"Experiment with seed {seed} generated an exception: {exc}")
    
    total_time = time.time() - start_time
    
    print(f"Completed {len(results)} of {num_experiments} experiments in {total_time:.2f} seconds")
    
    return results

def plot_module_temperatures(result, output_dir='results', prefix=''):
    """
    Create a bar chart comparing pre and post-optimization temperatures for each module.
    
    Parameters:
    - result: Dictionary with experiment results
    - output_dir: Directory to save the plots
    - prefix: Prefix for the output file names
    """
    if not result:
        print("No results to plot.")
        return
    
    # Extract data
    pre_temperatures = result['pre_temperatures']
    post_temperatures = result['post_temperatures']
    num_modules = len(pre_temperatures)
    
    # Create x positions for the bars
    module_indices = np.arange(num_modules)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Set larger font sizes
    plt.rcParams.update({'font.size': 14})
    
    # Plot bars - place them on the same vertical line with different colors
    # For better visibility, we'll make the bars slightly transparent
    plt.bar(module_indices, pre_temperatures, 
            color='red', alpha=0.6, label='original temperature')
    plt.bar(module_indices, post_temperatures, 
            color='blue', alpha=0.6, label='optimized temperature')
    
    # Add labels (lowercase)
    plt.xlabel('module index', fontsize=16)
    plt.ylabel('temperature metric', fontsize=16)
    
    # Add legend with larger font
    plt.legend(fontsize=16)
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # If there are many modules, only show some x-ticks
    if num_modules > 20:
        step = max(1, num_modules // 20)
        plt.xticks(module_indices[::step], fontsize=14)
    else:
        plt.xticks(module_indices, fontsize=14)
    
    plt.yticks(fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{prefix}module_temperatures.pdf"))
    plt.savefig(os.path.join(output_dir, f"{prefix}module_temperatures.png"))
    print(f"Module temperature comparison plot saved to {output_dir}")

def save_results(results, output_dir='results', prefix=''):
    """
    Save the computation results to disk for future use.
    
    Parameters:
    - results: List of dictionaries with experiment results
    - output_dir: Directory to save the results
    - prefix: Prefix for the output file names
    """
    if not results:
        print("No results to save.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the full results using pickle (for future Python use)
    pickle_path = os.path.join(output_dir, f"{prefix}full_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save a JSON-serializable version of the results (more portable)
    json_path = os.path.join(output_dir, f"{prefix}results_summary.json")
    
    # Prepare serializable data
    summary_data = []
    for result in results:
        # Extract only the serializable parts
        summary = {
            'seed': result['seed'],
            'pre_temperatures': result['pre_temperatures'],
            'post_temperatures': result['post_temperatures'],
            'total_pre_temperature': result['total_pre_temperature'],
            'total_post_temperature': result['total_post_temperature'],
            'temperature_improvement': result['temperature_improvement']
        }
        summary_data.append(summary)
    
    with open(json_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Results saved to {pickle_path} and {json_path}")
    
    # Create a CSV file with the temperature data for the first experiment
    if results:
        result = results[0]  # Use first result
        csv_path = os.path.join(output_dir, f"{prefix}module_temperatures.csv")
        with open(csv_path, 'w') as f:
            f.write("Module,PreTemperature,PostTemperature\n")
            for i, (pre, post) in enumerate(zip(result['pre_temperatures'], result['post_temperatures'])):
                f.write(f"{i},{pre},{post}\n")
        print(f"Module temperature data saved to {csv_path}")

def main():
    """
    Main function to run the optimization visualization analysis.
    """
    parser = argparse.ArgumentParser(description='Floorplanning Optimization Visualization')
    parser.add_argument('--num_experiments', type=int, default=1,
                        help='Number of experiments to run (default: 1)')
    parser.add_argument('--num_modules', type=int, default=150,
                        help='Number of modules to use (default: 150)')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='Weight between volume and temperature loss (default: 0.6)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--output_dir', type=str, default='results/optimization_visualization',
                        help='Directory to save results (default: results/optimization_visualization)')
    parser.add_argument('--prefix', type=str, default='optvis_',
                        help='Prefix for output files (default: optvis_)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')
    
    args = parser.parse_args()
    
    print(f"Starting floorplanning optimization visualization with:")
    print(f"  Number of experiments: {args.num_experiments}")
    print(f"  Number of modules: {args.num_modules}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Output directory: {args.output_dir}")
    
    # Run experiments
    results = run_multiple_experiments(
        num_experiments=args.num_experiments,
        num_modules=args.num_modules,
        alpha=args.alpha,
        max_workers=args.workers,
        verbose=args.verbose
    )
    
    if not results:
        print("No valid results to plot.")
        return
    
    # Add timestamp to prefix
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.prefix}{timestamp}_"
    
    # Save the results
    save_results(results, args.output_dir, prefix)
    
    # Create visualization - we'll just use the first experiment for the bar chart
    print("Creating visualization...")
    plot_module_temperatures(results[0], args.output_dir, prefix)
    
    print("Visualization completed!")

if __name__ == "__main__":
    main() 